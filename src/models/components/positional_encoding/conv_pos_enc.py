# src/models/components/positional_encoding/conv_pe.py
import torch
import torch.nn as nn


class ConvPositionalEncoding(nn.Module):
    """
    1‑D depthwise convolution that adds a positional bias.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # depthwise
            bias=True,  # Enable bias for better centering
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L) for conv1d
        pos = x.transpose(1, 2)
        pos = self.conv(pos)
        pos = pos.transpose(1, 2)
        pos = self.norm(pos)
        pos = self.activation(pos)
        return self.dropout(pos)
