import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from src.models.lcs_detr.matcher import build_matcher
from src.models.lcs_detr.transformer import build_transformer
from src.models.components.positional_encoding.base import build_position_encoding
from src.models.lcs_detr.criterion import SetCriterion

from src.models.components.transformer.encoder import (
    LocalSaliencyHead,
    Text2AudioEncoder,
    SaliencyAmplifier,
)


def inverse_sigmoid(x, eps=1e-3):
    """Applies inverse sigmoid transformation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        eps (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Inverse sigmoid of x.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class LCSDETR(nn.Module):
    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        aud_dim,
        txt_dim,
        num_queries,
        input_dropout,
        max_a_l,
        aux_loss=True,
        span_loss_type="l1",
        use_txt_pos=False,
        n_input_proj=2,
        use_saliency_conv=False,
        use_global_query_init=False,
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         LCS-DETR can detect in a single audio.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_a_l: int, maximum #clips in audio
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        span_pred_dim = 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False

        self.input_txt_proj = nn.Sequential(
            *[
                LinearLayer(
                    txt_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[0],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[1],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[2],
                ),
            ][:n_input_proj]
        )
        self.input_aud_proj = nn.Sequential(
            *[
                LinearLayer(
                    aud_dim + 2,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[0],
                ),  # add pos_embedding
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[1],
                ),
                LinearLayer(
                    hidden_dim,
                    hidden_dim,
                    layer_norm=True,
                    dropout=input_dropout,
                    relu=relu_args[2],
                ),
            ][:n_input_proj]
        )
        self.aux_loss = aux_loss

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        self.local_saliency_head = LocalSaliencyHead(
            model_dim=hidden_dim,
            use_projections=True,
            logit_mode="linear",
            use_gamma=True,
            num_aggregation_layers=1,
            use_saliency_conv=use_saliency_conv,
        )

        self.txt2aud_encoder = Text2AudioEncoder(
            d_model=hidden_dim,
            num_dummies=0,
            num_t2v_layers=2,
            dropout=input_dropout,
            droppath=0.1,
            use_cross_attn_wo_dummy=True,
            weight_attn_with_saliency=True,
        )

        self.saliency_amplifier = SaliencyAmplifier(
            d_model=hidden_dim,
            mode="sigmoid",
            use_mha=True,
        )

        self.use_global_query_init = use_global_query_init
        if use_global_query_init:
            self.query_init_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src_txt, src_txt_mask, src_aud, src_aud_mask):
        """The forward expects two tensors:
           - src_txt: [batch_size, L_txt, D_txt]
           - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer
           - src_aud: [batch_size, L_aud, D_aud]
           - src_aud_mask: [batch_size, L_aud], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer

        It returns a dict with the following elements:
           - "pred_spans": The normalized boxes coordinates for all queries, represented as
                           (center_x, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        src_aud = self.input_aud_proj(src_aud)
        src_txt = self.input_txt_proj(src_txt)

        audio_length = src_aud.shape[1]

        # Position embedding (you can still use your current one or upgrade to RoPE later)
        pos_aud = self.position_embed(src_aud, src_aud_mask)
        pos_txt = (
            self.txt_position_embed(src_txt)
            if self.use_txt_pos
            else torch.zeros_like(src_txt)
        )

        # Get saliency scores
        saliency_scores, src_sent = self.local_saliency_head(
            src_aud, src_txt, src_txt_mask
        )

        # Concat audio and text
        src = torch.cat([src_aud, src_txt], dim=1)
        mask = torch.cat([src_aud_mask, src_txt_mask], dim=1).bool()
        pos = torch.cat([pos_aud, pos_txt], dim=1)

        # Saliency-Guided Cross Attention
        src_updated, mask_updated, pos_updated, attn_weights = self.txt2aud_encoder(
            src=src,
            mask=mask,
            pos=pos,
            batch_audio_len=audio_length,
            saliency_scores=torch.sigmoid(saliency_scores),
        )

        src = src_updated
        mask = mask_updated
        pos = pos_updated

        # (#layers, bsz, #queries, d), (bsz, L_aud+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(
            src.shape[0], 1, 1
        )
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(
            pos.shape[0], 1, 1
        )
        pos = torch.cat([pos_, pos], dim=1)

        audio_length = src_aud.shape[1]

        # Decouple the transformer to inject Saliency Amplifier before the Decoder
        bs, seq_len, d = src.shape
        src_t = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed_t = pos.permute(1, 0, 2)  # (L, batch_size, d)

        # 1. T2V Encoder (processes text + audio + global)
        src_t = self.transformer.t2v_encoder(
            src_t,
            src_key_padding_mask=~mask,
            pos=pos_embed_t,
            audio_length=audio_length,
        )

        # Strip text tokens
        src_t = src_t[: audio_length + 1]
        mask_enc = (~mask)[:, : audio_length + 1]
        pos_embed_t = pos_embed_t[: audio_length + 1]

        # 2. Audio Encoder (processes audio + global)
        memory_t = self.transformer.encoder(
            src_t, src_key_padding_mask=mask_enc, pos=pos_embed_t
        )
        memory_global, memory_local_t = memory_t[0], memory_t[1:]

        # 3. Apply Saliency Amplifier on Audio Memory
        # Keep un-amplified memory for saliency score calculation
        aud_mem_unamplified = memory_local_t.transpose(0, 1)  # (batch_size, L_aud, d)

        aud_mem_amplified_t = self.saliency_amplifier(
            features=memory_local_t,  # (L_aud, bsz, d)
            saliency_scores=saliency_scores,
            pos=pos_aud.transpose(0, 1),  # (L_aud, bsz, d)
            aud_mask=src_aud_mask,
        )

        # 4. Decoder
        mask_local = mask_enc[:, 1:]
        pos_embed_local_t = pos_embed_t[1:]
        refpoint_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        if self.use_global_query_init:
            tgt = self.query_init_proj(memory_global).unsqueeze(0).repeat(refpoint_embed.shape[0], 1, 1)
        else:
            tgt = torch.zeros(refpoint_embed.shape[0], bs, d, device=src.device)

        hs, reference = self.transformer.decoder(
            tgt,
            aud_mem_amplified_t,  # Pass the amplified memory to the decoder!
            memory_key_padding_mask=mask_local,
            pos=pos_embed_local_t,
            refpoints_unsigmoid=refpoint_embed,
        )

        outputs_class = self.class_embed(
            hs
        )  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_spans": outputs_coord[-1]}

        ### Neg Pairs ###
        # Skip decoder for negative pairs to save massive compute
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_aud, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_aud_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()

        src_neg_t = src_neg.permute(1, 0, 2)
        pos_neg_t = pos_neg.permute(1, 0, 2)

        src_neg_t = self.transformer.t2v_encoder(
            src_neg_t,
            src_key_padding_mask=~mask_neg,
            pos=pos_neg_t,
            audio_length=audio_length,
        )
        src_neg_t = src_neg_t[: audio_length + 1]
        mask_neg_enc = (~mask_neg)[:, : audio_length + 1]
        pos_neg_t = pos_neg_t[: audio_length + 1]

        memory_neg_t = self.transformer.encoder(
            src_neg_t, src_key_padding_mask=mask_neg_enc, pos=pos_neg_t
        )
        memory_global_neg = memory_neg_t[0]
        aud_mem_neg_unamplified = memory_neg_t[1:].transpose(0, 1)

        out["saliency_scores"] = torch.sum(
            self.saliency_proj1(aud_mem_unamplified)
            * self.saliency_proj2(memory_global).unsqueeze(1),
            dim=-1,
        ) / np.sqrt(self.hidden_dim)

        out["saliency_scores_neg"] = torch.sum(
            self.saliency_proj1(aud_mem_neg_unamplified)
            * self.saliency_proj2(memory_global_neg).unsqueeze(1),
            dim=-1,
        ) / np.sqrt(self.hidden_dim)
        out["audio_mask"] = src_aud_mask
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    """Builds the LCS-DETR model and criterion.

    Args:
        args: Configuration arguments.

    Returns:
        tuple: model, criterion
    """
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/lcs_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = LCSDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        max_a_l=args.max_a_l,
        txt_dim=args.t_feat_dim,
        aud_dim=args.a_feat_dim,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_saliency_conv=args.get("use_saliency_conv", False),
        use_global_query_init=args.get("use_global_query_init", False),
    )

    matcher = build_matcher(args)
    use_focal_loss = args.get("use_focal_loss", False)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": 0.0 if use_focal_loss else args.label_loss_coef,
        "loss_saliency": args.lw_saliency,
    }
    if use_focal_loss:
        weight_dict["loss_focal"] = args.label_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items() if k != "loss_saliency"}
            )
        weight_dict.update(aux_weight_dict)

    losses = ["spans", "labels", "saliency"]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        span_loss_type=args.span_loss_type,
        max_a_l=args.max_a_l,
        saliency_margin=args.saliency_margin,
        use_focal_loss=use_focal_loss,
    )
    criterion.to(device)
    return model, criterion
