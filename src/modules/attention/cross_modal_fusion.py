# cross_modal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.attention.multi_head import MultiheadAttention


class CrossModalCoAttention(nn.Module):
    """
    Co-Attention between Audio and Text.
    Allows both modalities to attend to each other.
    """

    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Audio -> Text attention
        self.audio_to_text = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Text -> Audio attention
        self.text_to_audio = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, audio_feat, text_feat, audio_mask=None, text_mask=None):
        """
        audio_feat: (B, L_a, D)
        text_feat:  (B, L_t, D)
        """
        B = audio_feat.shape[0]

        # Audio attends to Text
        audio2text, _ = self.audio_to_text(
            query=audio_feat,
            key=text_feat,
            value=text_feat,
            key_padding_mask=~text_mask if text_mask is not None else None,
        )
        audio_feat = self.norm1(audio_feat + self.dropout(audio2text))

        # Text attends to Audio
        text2audio, _ = self.text_to_audio(
            query=text_feat,
            key=audio_feat,
            value=audio_feat,
            key_padding_mask=~audio_mask if audio_mask is not None else None,
        )
        text_feat = self.norm2(text_feat + self.dropout(text2audio))

        # FFN
        audio_feat = audio_feat + self.dropout(self.ffn(audio_feat))
        audio_feat = self.norm_ffn(audio_feat)

        text_feat = text_feat + self.dropout(self.ffn(text_feat))
        text_feat = self.norm_ffn(text_feat)

        return audio_feat, text_feat


class CrossModalFusionBlock(nn.Module):
    """Stack multiple co-attention layers + final fusion"""

    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalCoAttention(d_model, nhead, dropout) for _ in range(num_layers)]
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, audio_feat, text_feat, audio_mask=None, text_mask=None):
        for layer in self.layers:
            audio_feat, text_feat = layer(audio_feat, text_feat, audio_mask, text_mask)

        # Final fusion: concatenate + project
        fused = torch.cat([audio_feat, text_feat], dim=1)  # (B, L_a + L_t, D)

        return fused
