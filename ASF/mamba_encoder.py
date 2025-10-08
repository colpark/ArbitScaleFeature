"""
MAMBA Encoder - Bidirectional State Space Model Encoder
Part of MAMBA-GINR Innovation

This module implements a bidirectional Mamba encoder that processes sequences
in both forward and reverse directions, capturing richer contextual information.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block


class BiMamba(torch.nn.Module):
    """
    Bidirectional Mamba Layer

    Processes input sequence in both forward and reverse directions,
    then averages the outputs to capture bidirectional context.

    Args:
        dim (int): Model dimension
    """
    def __init__(self, dim=512):
        super(BiMamba, self).__init__()

        # Forward direction Mamba
        self.f_mamba = Mamba(d_model=dim)
        # Reverse direction Mamba
        self.r_mamba = Mamba(d_model=dim)

    def forward(self, x, **kwargs):
        # Forward pass
        x_f = self.f_mamba(x, **kwargs)
        # Reverse pass: flip input, process, flip output
        x_r = torch.flip(self.r_mamba(torch.flip(x, dims=[1]), **kwargs), dims=[1])
        # Average both directions
        out = (x_f + x_r) / 2

        return out


class MambaEncoder(torch.nn.Module):
    """
    Mamba Encoder Stack

    Stack of Mamba blocks with bidirectional processing and MLP layers.
    Each block contains:
    1. BiMamba mixer (bidirectional state space model)
    2. MLP with GELU activation
    3. Layer normalization

    Args:
        depth (int): Number of Mamba blocks
        dim (int): Model dimension
        ff_dim (int, optional): Feedforward dimension. Defaults to 4*dim
        dropout (float): Dropout rate
    """
    def __init__(self, depth=6, dim=768, ff_dim=None, dropout=0.):
        super(MambaEncoder, self).__init__()

        if not ff_dim:
            self.ff_dim = 4 * dim
        else:
            self.ff_dim = ff_dim

        token_dim = dim

        # Build encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=token_dim,
                mixer_cls=lambda dim: BiMamba(dim),
                mlp_cls=lambda dim: torch.nn.Sequential(
                    nn.Linear(dim, self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.ff_dim, dim),
                    nn.Dropout(dropout),
                ),
                norm_cls=nn.LayerNorm,
                fused_add_norm=False
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, D)
                B: batch size
                L: sequence length
                D: dimension

        Returns:
            Output tensor of shape (B, L, D)
        """
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual=residual)
        return x
