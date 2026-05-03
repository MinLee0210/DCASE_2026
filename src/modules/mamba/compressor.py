import torch.nn as nn
from mamba_ssm import Mamba
from .cross_attention import MambaCrossAttention


class ContextCompressor(nn.Module):
    def __init__(self, d_model, compress_ratio=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(compress_ratio)
        self.compress = nn.Linear(d_model, d_model)

    def forward(self, chunk):
        # [batch, seq, d_model] -> [batch, compress_ratio, d_model]
        compressed = self.pool(chunk.transpose(1, 2)).transpose(1, 2)
        return self.compress(compressed)


class ChunkedMambaProcessor(nn.Module):
    def __init__(self, d_model, chunk_size=2048, overlap=512):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=16)
        self.cross_attn = MambaCrossAttention(d_model)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def forward(self, x, context_window=2):
        """
        x: [batch, total_seq, d_model]
        context_window: how many neighboring chunks to attend to
        """
        outputs = []
        stride = self.chunk_size - self.overlap

        for i in range(0, x.shape[1], stride):
            chunk = x[:, i : i + self.chunk_size]

            # Gather context from neighboring chunks
            start_ctx = max(0, i - context_window * self.chunk_size)
            end_ctx = min(x.shape[1], i + (context_window + 1) * self.chunk_size)
            context = x[:, start_ctx:end_ctx]

            # Process
            mamba_out = self.mamba(chunk)
            cross_out = self.cross_attn(mamba_out, context)
            outputs.append(cross_out)

        # Merge overlapping regions (average or learnable blend)
        return self._merge_chunks(outputs, stride)
