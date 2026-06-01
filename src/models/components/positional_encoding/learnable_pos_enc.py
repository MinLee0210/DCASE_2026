import torch
import torch.nn as nn


class PositionEmbeddingLearned(nn.Module):
    """Learnable 1‑D absolute positional embedding.
    The embedding size is ``max_len`` x ``d_model`` and is added to the token features.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        # initialise close to sinusoidal (optional but helpful)
        with torch.no_grad():
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            sinusoid = torch.zeros(max_len, d_model)
            sinusoid[:, 0::2] = torch.sin(position * div_term)
            sinusoid[:, 1::2] = torch.cos(position * div_term)
            self.position_embeddings.weight.copy_(sinusoid)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Add positional embeddings to ``x``.
        Args:
            x: (B, L, D)
            mask: optional mask (ignored, kept for API compatibility)
        Returns:
            (B, L, D) with added embedding.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embeddings(positions)
        # Note: In standard DETR the pos embed is returned separately from x.
        # But we return it to match the sine embed output shape (B, L, D)
        return pos_emb.repeat(x.size(0), 1, 1)
