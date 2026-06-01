# src/models/components/positional_encoding/conv_pe.py
import torch
import torch.nn as nn


class ConvPositionalEncoding(nn.Module):
    """
    1‑D depthwise convolution that adds a positional bias.
    """

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # depthwise
            bias=False,
        )
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L) for conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)
