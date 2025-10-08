import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
#from mamba_ssm import Mamba
from mamba_repo.mamba_ssm.modules.mamba_simple import Mamba
from mamba_repo.mamba_ssm.modules.block import Block
import pickle

from models import register

class BiMamba(torch.nn.Module):
    def __init__(self, dim = 512):
        super(BiMamba, self).__init__()
        
        self.f_mamba = Mamba(d_model = dim)
        self.r_mamba = Mamba(d_model = dim)
        
    def forward(self, x, return_dt = True, **kwargs):
        x_f, dt_forward, dt_f_lr = self.f_mamba(x, **kwargs)
        x_r, dt_backward, dt_b_lr = self.r_mamba(torch.flip(x, dims=[1]), **kwargs)
        dts = torch.stack([dt_forward, dt_backward], dim = 0)
        dts_lr = torch.stack([dt_f_lr, dt_b_lr], dim = 0)
        x_r = torch.flip(x_r, dims=[1])
        out = (x_f + x_r)/2

        '''if dt_forward is not None and dt_backward is not None:
            dts = [dt_forward, dt_backward]
            print(f"This is the len of dts: {len(dts)}")
            with open('./dts.pkl', 'wb') as file:
                pickle.dump((dt_forward, dt_backward), file)'''
        
            
        return out, dts, dts_lr
        
@register('mamba_encoder_dt')     
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
                norm_cls=nn.LayerNorm,  # or RMSNorm
                fused_add_norm=False
            )
            for _ in range(depth)
        ])
    
    def forward(self, x):
        residual = None
        xs = []
        dt_list = []
        dt_lr_list = []
        for block in self.blocks:
            result = block(x, residual=residual)
            if len(result) == 4:
                x, residual, dts, dts_lr = result
                dt_list.append(dts)
                dt_lr_list.append(dts_lr)
            else:
                x, residual = result
            xs.append(x)
        return x, xs, dt_list, dt_lr_list
