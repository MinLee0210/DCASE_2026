import torch
import torch.nn as nn
from mamba_ssm import Mamba
from .cross_attention import MambaCrossAttention


class SparseContextAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, top_k=32):
        super().__init__()
        self.top_k = top_k
        self.similarity = nn.Linear(d_model, d_model)
        self.cross_attn = MambaCrossAttention(d_model, num_heads)

    def forward(self, chunk, context):
        # Find top-k most similar context tokens
        sim = torch.matmul(chunk, self.similarity(context).transpose(-2, -1))
        _, top_indices = torch.topk(sim, self.top_k, dim=-1)

        # Gather sparse context
        sparse_context = torch.gather(
            context, 1, top_indices.expand(-1, -1, context.shape[-1])
        )

        return self.cross_attn(chunk, sparse_context)


class HierarchicalMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_local = Mamba(d_model)
        self.mamba_global = Mamba(d_model)
        self.cross_attn_local = MambaCrossAttention(d_model)
        self.cross_attn_global = MambaCrossAttention(d_model)

    def forward(self, x, chunk_size=2048):
        # Process chunks locally
        chunks = x.reshape(-1, chunk_size, x.shape[-1])
        local_out = self.mamba_local(chunks)
        local_out = local_out.reshape_as(x)

        # Downsample for global context
        global_context = x[:, ::8]  # Every 8th token
        global_out = self.mamba_global(global_context)

        # Cross-attend locally to global
        local_attended = self.cross_attn_local(
            local_out, global_out.unsqueeze(1).expand_as(local_out)
        )

        return local_attended + local_out


class TemporalBiasMamba(nn.Module):
    def __init__(self, d_model, num_heads=8, max_distance=5):
        super().__init__()
        self.cross_attn = MambaCrossAttention(d_model, num_heads)
        self.max_distance = max_distance
        self.temporal_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, chunk, context, chunk_idx):
        # Compute attention with temporal distance penalty
        attn = self.cross_attn(chunk, context)

        # Bias toward nearby chunks
        dist = abs(chunk_idx - torch.arange(context.shape[0]))
        bias = self.temporal_bias(
            torch.clamp(dist, -self.max_distance, self.max_distance)
        )

        return attn * (1 + bias.unsqueeze(-1))


class DualStreamMamba(nn.Module):
    """
    Use two streams:

    Token stream: Mamba processes token-by-token within chunk
    Chunk stream: Cross-attn to summary of other chunks
    """

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
