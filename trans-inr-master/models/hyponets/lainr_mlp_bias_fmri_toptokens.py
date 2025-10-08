
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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseSharedTokenCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context, attn_indices, bias=None, debug=False):
        """
        x: (B, HW, D)         ← queries
        context: (B, L, D)    ← keys/values  
        attn_indices: (B, HW, k)  ← sparse indices for each query
        bias: (B, HW, k) or None  ← optional bias term
        """
        B, HW, D = x.shape
        _, L, _ = context.shape
        H, Dh = self.heads, self.dim_head
        k_neighbors = attn_indices.shape[-1]
        
        if debug:
            print(f"Input shapes:")
            print(f"  x: {x.shape}")
            print(f"  context: {context.shape}")
            print(f"  attn_indices: {attn_indices.shape}")
            if bias is not None:
                print(f"  bias: {bias.shape}")
        
        # Get dimensions and validate
        actual_HW = attn_indices.shape[1]  # Get the actual sequence length from attn_indices
        
        # Debug check - ensure x and attn_indices have compatible dimensions
        if HW != actual_HW:
            raise ValueError(f"Mismatch: x has {HW} tokens but attn_indices has {actual_HW} tokens")
        
        if debug: print(f"Memory-efficient gather: B={B}, H={H}, actual_HW={actual_HW}, k={k_neighbors}, L={L}, Dh={Dh}")
        
        # Project queries
        q = self.to_q(x).view(B, HW, H, Dh).transpose(1, 2)  # (B, H, HW, Dh)
        if debug: print(f"q shape: {q.shape}")
        
        # Project keys and values
        kv = self.to_kv(context)  # (B, L, 2*H*Dh)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        v = v.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        if debug: print(f"k shape: {k.shape}, v shape: {v.shape}")
        
        # Process attention in chunks to avoid memory explosion
        # Never materialize the full k_sparse/v_sparse tensors
        
        chunk_size = 1024  # Process 1024 queries at a time
        output_chunks = []
        
        for i in range(0, actual_HW, chunk_size):
            end_idx = min(i + chunk_size, actual_HW)
            chunk_len = end_idx - i
            
            # Get queries for this chunk
            q_chunk = q[:, :, i:end_idx, :]  # (B, H, chunk_len, Dh)
            attn_indices_chunk = attn_indices[:, i:end_idx, :]  # (B, chunk_len, k)
            
            # Gather k,v for this chunk only
            k_chunk_sparse = torch.empty(B, H, chunk_len, k_neighbors, Dh, device=k.device, dtype=k.dtype)
            v_chunk_sparse = torch.empty(B, H, chunk_len, k_neighbors, Dh, device=v.device, dtype=v.dtype)
            
            for b in range(B):
                for h in range(H):
                    for hw_local in range(chunk_len):
                        indices = attn_indices_chunk[b, hw_local, :]  # (k_neighbors,)
                        k_chunk_sparse[b, h, hw_local, :, :] = k[b, h, indices, :]
                        v_chunk_sparse[b, h, hw_local, :, :] = v[b, h, indices, :]
            
            # Compute attention for this chunk
            sim_chunk = torch.einsum('bhqd,bhqkd->bhqk', q_chunk, k_chunk_sparse) * self.scale
            
            # Handle bias if present
            if bias is not None:
                bias_chunk = bias[:, i:end_idx, :]  # (B, chunk_len, k)
                bias_exp = bias_chunk.unsqueeze(1)  # (B, 1, chunk_len, k)
                sim_chunk = sim_chunk + bias_exp
            
            attn_chunk = F.softmax(sim_chunk, dim=-1)  # (B, H, chunk_len, k)
            
            # Weighted sum of values for this chunk
            out_chunk = torch.einsum('bhqk,bhqkd->bhqd', attn_chunk, v_chunk_sparse)  # (B, H, chunk_len, Dh)
            output_chunks.append(out_chunk)
            
            if debug and i == 0:
                print(f"Processing chunks of size {chunk_len}, total chunks: {(actual_HW + chunk_size - 1) // chunk_size}")
                print(f"Chunk memory usage: {k_chunk_sparse.numel() * 4 / 1e9:.2f} GB")
        
        # Concatenate all chunks
        out = torch.cat(output_chunks, dim=2)  # (B, H, actual_HW, Dh)
        if debug: print(f"out before reshape: {out.shape}")
        
        # Reshape back to original format
        out = out.transpose(1, 2).contiguous().view(B, actual_HW, H * Dh)  # (B, actual_HW, H*Dh)
        out = self.to_out(out)  # (B, actual_HW, D)
        if debug: print(f"final out shape: {out.shape}")
        
        return out


@register('lainr_mlp_bias_fmri_toptokens')
class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim, input_dim, output_dim, sigma_q, sigma_ls, n_patches, 
                 hidden_dim=256, context_dim=None, top_k=512):
        super().__init__()
        self.layer_num = len(sigma_ls)
        self.n = feature_dim//(2*input_dim)
        self.omegas = torch.logspace(1, math.log10(sigma_q), self.n)
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0
        self.top_k = top_k  # Store top_k parameter

        self.omegas_l = [torch.logspace(1, math.log10(sigma_ls[i]), self.n) for i in range(self.layer_num)]

        self.query_lin = nn.Linear(feature_dim, hidden_dim)

        # Pass top_k to the cross-attention module
        self.modulation_ca = SparseSharedTokenCrossAttention(
            query_dim = hidden_dim, heads=2
        )
        
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
        L = x.shape[0]
        coords = x.unsqueeze(-1)  # (H*W, 2, 1)
        omegas = omegas.view(1, 1, -1).to(x.device)  # (1, 1, F)
        
        arg = torch.pi * coords * omegas  # shape: (B, 2, F)
        sin_part = torch.sin(arg)
        cos_part = torch.cos(arg)
        
        gamma = torch.cat([sin_part, cos_part], dim=-1).view(L, -1)  
        return gamma

    def get_patch_index(self, grid, D, H, W, T):
        z = grid[:, 0]  # D dimension
        y = grid[:, 1]  # H dimension
        x = grid[:, 2]  # W dimension
        t = grid[:, 3]  # T dimension
    
        # Convert normalized coords to integer indices
        z_idx = (z * D).to(torch.int32)
        y_idx = (y * H).to(torch.int32)
        x_idx = (x * W).to(torch.int32)
        t_idx = (t * T).to(torch.int32)
    
        # Flatten: W → H → D → T
        return (((t_idx * D + z_idx) * H + y_idx) * W + x_idx)
    
    def approximate_relative_distances(self, target_index, D, H, W, T, m):
        alpha = self.alpha
        N = D * H * W * T
        t = target_index / N
        token_positions = torch.tensor([(i + 0.5) / m for i in range(m)],
                                        device=target_index.device)
    
        rel_distances = alpha * torch.stack([
            torch.abs((t - s) ** 2) for s in token_positions
        ], dim=0)
        return rel_distances

    def forward(self, x, tokens, shape):
        k = 512
        D, H, W, T = shape
        B, query_shape = x.shape[0], x.shape[1]
        x = x.view(B, -1, x.shape[-1])  # B, HW, 2
        
        grid = x[0]
        indexes = self.get_patch_index(grid, D, H, W, T)
        rel_distances = self.approximate_relative_distances(indexes, D, H, W, T, len(tokens[0]))
        bias = rel_distances.transpose(1, 0)
        bias = einops.repeat(bias, 'l n -> b l n', b=B)  # B, HW, L
        _, attn_indices = torch.topk(bias, self.top_k, dim=-1, largest=False)
        
        x_q = einops.repeat(self.calc_gamma(x[0], self.omegas), 'l d -> b l d', b=B)  # B, HW, input_dim
        x_q = self.act(self.query_lin(x_q))

        # The cross-attention will now automatically select top-k tokens if top_k is set
        #modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)
        modulation_vector = self.modulation_ca(x_q, context=tokens, attn_indices = attn_indices, bias=None, debug = True)

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
        out = out.view(B, query_shape, -1)  # (B, H, W, output_dim)

        return out
        
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

