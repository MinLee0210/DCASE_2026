# rope.py
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb.unsqueeze(0).expand(bsz, -1, -1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: (..., seq_len, head_dim)"""
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.reshape(1, 1, -1, 2) if cos.dim() == 2 else cos
    sin = sin.reshape(1, 1, -1, 2) if sin.dim() == 2 else sin

    x_out = torch.stack(
        [
            x_[..., 0] * cos[..., 0] - x_[..., 1] * sin[..., 0],
            x_[..., 1] * cos[..., 0] + x_[..., 0] * sin[..., 0],
        ],
        dim=-1,
    ).flatten(-2)
    return x_out.type_as(x)
