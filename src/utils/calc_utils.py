import torch


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
