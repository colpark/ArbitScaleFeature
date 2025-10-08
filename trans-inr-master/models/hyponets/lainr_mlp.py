import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import einops
import ssl
import math
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from math import pi, log
from functools import wraps
import torch.nn.functional as F
from models import register

@register('lainr_mlp')
class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim, input_dim, output_dim, sigma_q, sigma_ls, hidden_dim = 256, context_dim = None):
        super().__init__()
        self.layer_num = len(sigma_ls)
        self.n = feature_dim//(2*input_dim)
        self.omegas = torch.logspace(1, math.log10(sigma_q), self.n)
        
        '''self.omegas_l = {str(sig_l):torch.logspace(1, math.log10(sig_l), self.n) for sig_l in sigma_ls}
        self.bandwidth_lins_lins = nn.ModuleDict({
            str(sig_l): nn.Linear(feature_dim, hidden_dim) for sig_l in sigma_ls
        })
        self.modulation_lins = nn.ModuleDict({
            str(sig_l): nn.Linear(hidden_dim, hidden_dim) for sig_l in sigma_ls
        })'''

        self.omegas_l = [torch.logspace(1, math.log10(sigma_ls[i]), self.n) for i in range(self.layer_num)]

        self.query_lin = nn.Linear(feature_dim, hidden_dim)

        #self.modulation_ca = PerPixelCrossAttention(query_dim = hidden_dim, heads=2)
        self.modulation_ca = SharedTokenCrossAttention(query_dim = hidden_dim, heads=2)
        
        self.bandwidth_lins = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for i in range(self.layer_num)
                                       ])
        
        self.modulation_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for i in range(self.layer_num)
        ])

        self.hv_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(len(sigma_ls) - 1)
        ])

        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(len(sigma_ls))
        ])
        self.act = nn.ReLU()
        

    def calc_gamma(self, x, omegas):
        #x is passed as H*W, D
        L = x.shape[0]
        coords = x.unsqueeze(-1)  # (H*W, 2, 1)
        omegas = omegas.view(1, 1, -1).to(x.device)  # (1, 1, F)
        
        
        arg = torch.pi * coords * omegas  # shape: (B, 2, F)
        sin_part = torch.sin(arg)
        cos_part = torch.cos(arg)
        
        gamma = torch.cat([sin_part, cos_part], dim=-1).view(L, -1)  
        
        return gamma

    def get_patch_index(x, y, H, W):
        row = int(y * H)
        col = int(x * W)
        return row * W + col  # index in [0, N-1]

    def get_patch_index(x, y, H, W):
        row = int(y * H)
        col = int(x * W)
        return row * W + col  # index in [0, N-1]

        
    def forward(self, x, tokens):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1]) #B, HW, 2
        
        x_q = einops.repeat(self.calc_gamma(x[0], self.omegas), 'l d -> b l d', b=B) #B, HW, input_dim

        x_q = self.act(self.query_lin(x_q))


        #(B, HW, 1, D)
        #(B, HW, L, D)

        
        '''tokens = einops.repeat(tokens, 'b l d -> b p l d', p=query_shape[0]*query_shape[1])
        modulation_vector = self.modulation_ca(x_q.unsqueeze(2), context = tokens).squeeze(2)'''
        modulation_vector = self.modulation_ca(x_q, context = tokens)

        modulations_l = []
        h_f = []


        for k in range(self.layer_num):
            x_l = einops.repeat(self.calc_gamma(x[0], self.omegas_l[k]), 'l d -> b l d', b=B)
            h_l = self.act(self.bandwidth_lins[k](x_l))
            h_f.append(h_l)
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        h_v = [modulations_l[0]]

        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]

        out = sum(outs)
        out = out.view(B, *query_shape, -1)  # (B, H, W, output_dim)

        return out

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)


'''This cross attention class takes in coordinates that have been fourier expanded of shape (B, HW, D) 
where for each pixel of each image in the batch (B*H*W pixels), it calculates the cross attention between the (D, ) shape query for 
that pixel location and the (L, D) shaped set of tokens extracted from the Mamba hypernetwork for that image to get the 
modulation vector. We unsqueeze on dim = 2 to make the fourier embedded coordinates of shape (B, HW, 1, D) and the tokens
of shape (B, HW, L, D) by repeating HW times along dim = 1 to make sure each set of tokens for an image is attending to each of 
the HW pixels from that image'''
class PerPixelCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: (B, HW, 1, D)     (1-query per pixel)
        # context: (B, HW, L, D) (L tokens per pixel)

        B, HW, _, D = x.shape
        L = context.shape[2]
        H = self.heads

        # Project
        q = self.to_q(x)                           # (B, HW, 1, H*Dh)
        kv = self.to_kv(context)                   # (B, HW, L, 2*H*Dh)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for batched multihead attention
        q = q.view(B, HW, 1, H, self.dim_head)     # (B, HW, 1, H, Dh)
        k = k.view(B, HW, L, H, self.dim_head)     # (B, HW, L, H, Dh)
        v = v.view(B, HW, L, H, self.dim_head)

        # Move heads to batch dimension for einsum
        q = q.permute(0, 3, 1, 2, 4).reshape(B * H * HW, 1, self.dim_head)   # (B*H*HW, 1, Dh)
        k = k.permute(0, 3, 1, 2, 4).reshape(B * H * HW, L, self.dim_head)   # (B*H*HW, L, Dh)
        v = v.permute(0, 3, 1, 2, 4).reshape(B * H * HW, L, self.dim_head)   # (B*H*HW, L, Dh)

        # Attention
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale       # (B*H*HW, 1, L)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)                 # (B*H*HW, 1, Dh)

        # Reshape back
        out = out.view(B, H, HW, 1, self.dim_head).permute(0, 2, 3, 1, 4)    # (B, HW, 1, H, Dh)
        out = out.reshape(B, HW, 1, H * self.dim_head)                       # (B, HW, 1, D)
        return self.to_out(out)                                             # (B, HW, 1, D)

class SharedTokenCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: (B, HW, D)         ← 1 query per pixel (you can squeeze that 1)
        # context: (B, L, D)       ← shared tokens

        B, HW, D = x.shape

        H = self.heads
        Dh = self.dim_head
        D_inner = H * Dh

        q = self.to_q(x)              # (B, HW, H*Dh)
        kv = self.to_kv(context)      # (B, L, 2*H*Dh)
        k, v = kv.chunk(2, dim=-1)    # (B, L, H*Dh)

        # Reshape
        q = q.view(B, HW, H, Dh).transpose(1, 2)   # (B, H, HW, Dh)
        k = k.view(B, -1, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        v = v.view(B, -1, H, Dh).transpose(1, 2)   # (B, H, L, Dh)

        # Attention
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, HW, L)
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)                              # (B, H, HW, Dh)

        out = out.transpose(1, 2).contiguous().view(B, HW, D_inner)  # (B, HW, H*Dh)
        out = self.to_out(out)                                        # (B, HW, D)
        return out                                   # (B, HW, 1, D)
