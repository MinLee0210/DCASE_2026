import torch
import torch.nn as nn
from mamba_ssm import Mamba


class AttentionGatedMamba(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state)
        self.context_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x, context):
        mamba_out = self.mamba(x)
        gate = self.context_gate(torch.cat([mamba_out, context], dim=-1))
        return mamba_out * gate
