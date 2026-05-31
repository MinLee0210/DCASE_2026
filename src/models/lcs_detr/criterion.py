import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from src.utils.misc import accuracy
from src.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        span_loss_type,
        max_a_l,
        saliency_margin=1,
        use_focal_loss=False,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_a_l = max_a_l
        self.saliency_margin = saliency_margin
        self.use_focal_loss = use_focal_loss

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = (
            self.eos_coef
        )  # lower weight for background (index 1, foreground index 0)
        self.register_buffer("empty_weight", empty_weight)

    def _focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """Computes focal loss for classification."""
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        log_pt = log_prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        pt = prob.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Clamp pt and log_pt to guarantee numerical stability of gradients under mixed-precision (autocast)
        pt = torch.clamp(pt, min=1e-5, max=1.0 - 1e-5)
        log_pt = torch.clamp(log_pt, min=-12.0)

        alpha_t = torch.where(targets == self.foreground_label, alpha, 1.0 - alpha)
        loss = -alpha_t * ((1.0 - pt) ** gamma) * log_pt
        return loss.mean()

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert "pred_spans" in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs["pred_spans"][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat(
            [t["spans"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
            loss_giou = 1 - torch.diag(
                generalized_temporal_iou(
                    span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)
                )
            )
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction="none")
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses["loss_span"] = loss_span.mean()
        losses["loss_giou"] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.background_label,
            dtype=torch.int64,
            device=src_logits.device,
        )  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            reduction="none",
        )
        losses = {"loss_label": loss_ce.mean()}

        if self.use_focal_loss:
            losses["loss_focal"] = self._focal_loss(src_logits, target_classes)

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], self.foreground_label)[0]
            )
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        aud_token_mask = outputs["audio_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()

        loss_neg_pair = (
            (-torch.log(1.0 - torch.sigmoid(saliency_scores_neg)) * aud_token_mask)
            .sum(dim=1)
            .mean()
        )

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat(
            [saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1
        )

        aud_token_mask = aud_token_mask.repeat([1, 2])
        saliency_scores = (
            aud_token_mask * saliency_scores + (1.0 - aud_token_mask) * -1e3
        )

        tau = 0.5
        loss_rank_contrastive = 0.0

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (
                saliency_contrast_label >= rand_idx
            )  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = (
                    torch.sum(pos_mask, dim=1) > 0
                )  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e3

            # numerical stability
            logits = (
                cur_saliency_scores
                - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
            )

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * aud_token_mask).sum(1) / (
                pos_mask.sum(1) + 1e-6
            )

            loss = -mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [
                saliency_scores[batch_indices, pos_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        neg_scores = torch.stack(
            [
                saliency_scores[batch_indices, neg_indices[:, col_idx]]
                for col_idx in range(num_pairs)
            ],
            dim=1,
        )
        loss_saliency = (
            torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum()
            / (len(pos_scores) * num_pairs)
            * 2
        )  # * 2 to keep the loss the same scale

        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def _get_src_permutation_idx(self, indices):
        """Permutes predictions following the given indices.

        Args:
            indices: List of tuples (src, tgt) indices.

        Returns:
            tuple: batch_idx, src_idx
        """
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        """Permutes targets following the given indices.

        Args:
            indices: List of tuples (src, tgt) indices.

        Returns:
            tuple: batch_idx, tgt_idx
        """
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        """Retrieves the loss function for the given loss type.

        Args:
            loss (str): Type of loss.
            outputs: Model outputs.
            targets: Ground truth targets.
            indices: Matched indices.
            **kwargs: Additional arguments.

        Returns:
            dict: Loss values.
        """
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        indices = self.matcher(outputs_without_aux, targets)
        losses_target = self.losses

        # Compute all the requested losses
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                losses_target = self.losses

                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
