import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query from primary sequence
        self.to_q = nn.Linear(d_model, d_model)
        # Key, Value from context sequence
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.scale = self.head_dim**-0.5

    def forward(self, x, context):
        """
        x: [batch, seq_len, d_model] - primary sequence (query source)
        context: [batch, context_len, d_model] - context sequence (K, V source)
        """
        Q = self.to_q(x).reshape(-1, self.num_heads, x.shape[1], self.head_dim)
        K = self.to_k(context).reshape(
            -1, self.num_heads, context.shape[1], self.head_dim
        )
        V = self.to_v(context).reshape(
            -1, self.num_heads, context.shape[1], self.head_dim
        )

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return out.reshape(-1, x.shape[1], self.d_model)


class MambaWithCrossAttention(nn.Module):
    def __init__(self, d_model, d_state=16, num_heads=8):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state)
        self.cross_attn = MambaCrossAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        # Standard Mamba
        mamba_out = self.mamba(x)

        if context is not None:
            # Cross-attend to context
            cross_out = self.cross_attn(mamba_out, context)
            return self.norm(mamba_out + cross_out)  # Residual

        return mamba_out
