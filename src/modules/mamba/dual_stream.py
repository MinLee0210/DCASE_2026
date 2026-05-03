import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .cross_attention import MambaCrossAttention


class DualStreamMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_tokens = Mamba(d_model)
        self.chunk_summarizer = nn.Linear(d_model, d_model)
        self.cross_attn = MambaCrossAttention(d_model)

    def forward(self, chunks):  # [num_chunks, chunk_size, d_model]
        # Token-level processing
        token_outs = torch.stack([self.mamba_tokens(c) for c in chunks])

        # Chunk-level summaries
        chunk_summaries = self.chunk_summarizer(chunks.mean(dim=1))

        # Cross-attend tokens to chunk context
        results = []
        for i, tok_out in enumerate(token_outs):
            context = torch.cat([chunk_summaries[:i], chunk_summaries[i + 1 :]])
            cross = self.cross_attn(tok_out, context)
            results.append(tok_out + cross)

        return torch.stack(results)
