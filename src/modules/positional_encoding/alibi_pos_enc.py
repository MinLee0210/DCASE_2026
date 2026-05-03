import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alibi_slopes(n_heads: int):
    """Generate slopes for each head (geometric progression)"""
    start = 2 ** (-8.0 / n_heads)
    return torch.tensor(
        [start * (start**i) for i in range(n_heads)], dtype=torch.float32
    )


def create_alibi_bias(n_heads: int, seq_len: int, device: torch.device):
    """Create the ALiBi bias matrix: [n_heads, seq_len, seq_len]"""
    slopes = get_alibi_slopes(n_heads).to(device).view(n_heads, 1, 1)

    # Create distance matrix (negative for recency bias)
    pos = torch.arange(seq_len, device=device)
    distances = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq_len, seq_len]

    # Bias = -slope * distance  (only penalize past, but usually full matrix)
    alibi = distances.unsqueeze(0) * slopes  # [n_heads, seq_len, seq_len]
    return alibi  # You add this to attention scores (before softmax)


# Example integration in attention
class AttentionWithALiBi(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.alibi_slopes = get_alibi_slopes(n_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # q, k, v: [batch, heads, seq_len, head_dim]
        batch, heads, seq_len, _ = q.shape

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add ALiBi bias
        alibi = create_alibi_bias(heads, seq_len, q.device)
        scores = scores + alibi

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output
