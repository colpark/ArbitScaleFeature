# Fixed LAINRDecoderSCENT - corrects spatial bias dimension mismatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math
from math import log, pi

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def gaussian_fourier_encode(coords, B_matrix):
    """Gaussian Fourier Feature encoding"""
    if coords.dim() == 3:
        coords = coords.view(-1, coords.shape[-1])
    proj = 2 * np.pi * coords @ B_matrix.T
    features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    return features

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, bias=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(bias):
            # Bias is (b, l, n) or (b*h, l, n)
            if bias.dim() == 3 and bias.shape[0] == x.shape[0]:  # (b, l, n)
                bias = repeat(bias, 'b l n -> (b h) l n', h=h)
            sim = sim + bias

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LAINRDecoderSCENT(nn.Module):
    """LAINR Decoder with SCENT-style architecture - FIXED SPATIAL BIAS"""

    def __init__(self, feature_dim=64, input_dim=2, output_dim=3,
                 sigma_q=16, sigma_ls=[128, 32], n_patches=256,
                 hidden_dim=512,
                 context_dim=256,
                 learnable_frequencies=True,
                 num_layers=3,
                 heads=8,
                 dim_head=64):
        super().__init__()

        self.layer_num = len(sigma_ls)
        self.n_features = feature_dim // 2
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0
        self.num_layers = num_layers

        # Initialize Gaussian Fourier frequency matrices
        B_q_init = torch.randn(self.n_features, input_dim) / sigma_q
        B_ls_init = [torch.randn(self.n_features, input_dim) / sigma_ls[i]
                     for i in range(self.layer_num)]

        if learnable_frequencies:
            self.B_q = nn.Parameter(B_q_init)
            self.B_ls = nn.ParameterList([
                nn.Parameter(B_ls_init[i]) for i in range(self.layer_num)
            ])
        else:
            self.register_buffer('B_q', B_q_init)
            for i in range(self.layer_num):
                self.register_buffer(f'B_l_{i}', B_ls_init[i])
            self.B_ls = [getattr(self, f'B_l_{i}') for i in range(self.layer_num)]

        # Query encoding
        self.query_lin = nn.Linear(feature_dim, hidden_dim)

        # Cross-attention for modulation extraction
        self.modulation_ca = PreNorm(
            hidden_dim,
            Attention(hidden_dim, context_dim, heads=2, dim_head=64),
            context_dim=context_dim
        )

        # Bandwidth encoders - SCENT-STYLE (Attention + FF)
        self.bandwidth_lins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                *[
                    nn.ModuleList([
                        PreNorm(hidden_dim, Attention(hidden_dim, heads=heads, dim_head=dim_head)),
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
                    ])
                    for _ in range(num_layers - 1)
                ]
            )
            for _ in range(self.layer_num)
        ])

        # Modulation projections - SCENT-STYLE
        self.modulation_lins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                *[
                    nn.ModuleList([
                        PreNorm(hidden_dim, Attention(hidden_dim, heads=heads, dim_head=dim_head)),
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
                    ])
                    for _ in range(num_layers - 1)
                ]
            )
            for _ in range(self.layer_num)
        ])

        # Hidden value layers - SCENT-STYLE
        self.hv_lins = nn.ModuleList([
            nn.ModuleList([
                PreNorm(hidden_dim, Attention(hidden_dim, heads=heads, dim_head=dim_head)),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
            ])
            for _ in range(self.layer_num - 1)
        ])

        # Skip connection projections
        self.fourier_skip_projs = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        # Output layers
        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(self.layer_num)
        ])

        self.act = nn.ReLU()

    def get_patch_index(self, grid, H, W):
        """Convert coordinates to patch indices"""
        y = grid[:, 0]
        x = grid[:, 1]
        row = (y * H).to(torch.int32).clamp(0, H-1)
        col = (x * W).to(torch.int32).clamp(0, W-1)
        return row * W + col

    def approximate_relative_distances(self, target_index, H, W, m):
        """Compute spatial bias"""
        alpha = self.alpha
        N = H * W
        t = target_index.float() / N

        token_positions = torch.tensor(
            [(i + 0.5) / m for i in range(m)],
            device=target_index.device
        )

        t_expanded = t.unsqueeze(0)
        tokens_expanded = token_positions.unsqueeze(1)
        rel_distances = -alpha * torch.abs(t_expanded - tokens_expanded)**2

        return rel_distances

    def apply_block_sequence(self, x, block_seq):
        """Apply sequence of Attention + FeedForward blocks"""
        # First layer is Linear + ReLU
        if isinstance(block_seq[0], nn.Linear):
            x = block_seq[1](block_seq[0](x))  # Linear + ReLU
            start_idx = 2
        else:
            start_idx = 0

        # Rest are Attention + FF blocks
        for i in range(start_idx, len(block_seq)):
            block_list = block_seq[i]
            if isinstance(block_list, nn.ModuleList):
                attn, ff = block_list[0], block_list[1]
                x = attn(x) + x
                x = ff(x) + x
        return x

    def forward(self, coords_decoding, tokens, coords_modulation=None):
        """
        Forward pass with SCENT-style architecture - FIXED

        Args:
            coords_decoding: (B, H, W, 2) - where to predict RGB
            tokens: (B, L, D) - LP token features
            coords_modulation: (B, H, W, 2) - where to extract modulation (None = test mode)
        """
        B, query_shape = coords_decoding.shape[0], coords_decoding.shape[1:-1]
        coords_dec = coords_decoding.view(B, -1, coords_decoding.shape[-1])

        if coords_modulation is not None:
            coords_mod = coords_modulation.view(B, -1, coords_modulation.shape[-1])
        else:
            coords_mod = coords_dec

        # === MODULATION EXTRACTION ===

        # CRITICAL FIX: Use coords_mod (not coords_dec) for spatial bias computation
        # This ensures bias dimensions match the actual query length
        grid_mod = coords_mod[0]  # (HW, 2) where HW = coords_mod length
        num_queries = grid_mod.shape[0]  # This is the actual query count

        # Compute indices based on actual grid size
        H_mod = W_mod = int(math.sqrt(num_queries))
        indexes = self.get_patch_index(grid_mod, H_mod, W_mod)

        rel_distances = self.approximate_relative_distances(
            indexes, H_mod, W_mod, tokens.shape[1]
        )
        bias = repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding - use coords_mod for consistency
        x_q = repeat(
            gaussian_fourier_encode(coords_mod[0], self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        # Extract modulation with properly sized bias
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # === MULTI-SCALE DECODING (SCENT-STYLE) ===

        modulations_l = []
        fourier_encodings = []

        for k in range(self.layer_num):
            # Bandwidth encoding using coords_dec (decoding positions)
            x_l_fourier = gaussian_fourier_encode(coords_dec[0], self.B_ls[k])
            x_l_fourier_batch = repeat(x_l_fourier, 'l d -> b l d', b=B)
            fourier_encodings.append(x_l_fourier_batch)

            # Process through Attention + FF blocks
            h_l = self.apply_block_sequence(x_l_fourier_batch, self.bandwidth_lins[k])

            # Modulation projection (Attention + FF blocks)
            m_proj = self.apply_block_sequence(modulation_vector, self.modulation_lins[k])

            # Combine
            m_l = self.act(h_l + m_proj)
            modulations_l.append(m_l)

        # Residual connections between scales
        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            x_combined = modulations_l[i+1] + h_v[i]

            # Apply Attention + FF
            attn, ff = self.hv_lins[i][0], self.hv_lins[i][1]
            x_combined = attn(x_combined) + x_combined
            x_combined = ff(x_combined) + x_combined

            h_v.append(x_combined)

        # Output with SKIP CONNECTIONS (like SCENT!)
        outs = []
        for i in range(self.layer_num):
            # Add skip connection from original Fourier encoding
            fourier_skip = self.fourier_skip_projs[i](fourier_encodings[i])
            out_with_skip = self.out_lins[i](h_v[i] + fourier_skip)

            outs.append(out_with_skip)

        out = sum(outs)
        out = out.view(B, *query_shape, -1)

        return out


print("✓ Fixed LAINRDecoderSCENT defined")
print("  Fix: Spatial bias now computed using coords_mod dimensions")
print("  This ensures bias shape matches actual query count")
