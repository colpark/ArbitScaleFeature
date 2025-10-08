import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm


from models import register

class BiMamba(torch.nn.Module):
    def __init__(self, dim = 512):
        super(BiMamba, self).__init__()
        
        self.f_mamba = Mamba(d_model = dim)
        self.r_mamba = Mamba(d_model = dim)
        
    def forward(self, x, **kwargs):
        x_f = self.f_mamba(x, **kwargs)
        x_r = torch.flip(self.r_mamba(torch.flip(x, dims=[1]), **kwargs), dims=[1])
        out = (x_f + x_r)/2
        
        return out

        
@register('mamba_encoder')     
class MambaEncoder(torch.nn.Module):
    def __init__(self, depth = 6, dim = 768, ff_dim = None, dropout=0.):
        super(MambaEncoder, self).__init__()
        if not ff_dim:
            self.ff_dim = 4*dim
        else: 
            self.ff_dim = ff_dim
        token_dim = dim
        self.blocks = nn.ModuleList([
            Block(
                dim=token_dim,
                mixer_cls= lambda dim: BiMamba(dim),
                mlp_cls= lambda dim: torch.nn.Sequential(
                    nn.Linear(dim, self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.ff_dim, dim),
                    nn.Dropout(dropout),
                ),
                norm_cls= nn.LayerNorm,  # or RMSNorm, 
                fused_add_norm=False
            )
            for _ in range(depth)
        ])
    
    def forward(self, x):
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual=residual)
        return x
