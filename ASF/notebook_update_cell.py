# ============================================================================
# ADD THIS CELL TO YOUR NOTEBOOK TO REPLACE CELL 4
# This implements the CORRECT specifications:
# (1) Gaussian Fourier features everywhere
# (2) Sub-pixel jittering constraint
# (3) Separate modulation/decoding coordinates
# (4) No jittering at test time
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np


# ============================================================================
# Gaussian Fourier Feature Encoding
# ============================================================================

def gaussian_fourier_encode(coords, B_matrix):
    """
    Gaussian Fourier Feature encoding

    Args:
        coords: (HW, 2) - coordinates in [0, 1]
        B_matrix: (n_features, 2) - random frequency matrix

    Returns:
        features: (HW, 2*n_features) - [cos(2πx·B), sin(2πx·B)]
    """
    # Project coordinates to random frequencies
    proj = 2 * np.pi * coords @ B_matrix.T  # (HW, n_features)

    # Compute cos and sin
    features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    return features


# ============================================================================
# Shared Token Cross Attention (unchanged)
# ============================================================================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class SharedTokenCrossAttention(nn.Module):
    """Cross-attention with spatial bias"""
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

    def forward(self, x, context, bias=None):
        B, HW, D = x.shape
        H = self.heads
        Dh = self.dim_head
        D_inner = H * Dh

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, HW, H, Dh).transpose(1, 2)
        k = k.view(B, -1, H, Dh).transpose(1, 2)
        v = v.view(B, -1, H, Dh).transpose(1, 2)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if bias is not None:
            bias = einops.repeat(bias, 'b l n -> b h l n', h=H)
            bias = bias.transpose(-2, -1)
            sim = sim + bias

        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, HW, D_inner)
        out = self.to_out(out)
        return out


# ============================================================================
# LAINR Decoder with Gaussian Fourier Features
# ============================================================================

class LAINRDecoderGaussian(nn.Module):
    """
    LAINR with Gaussian Fourier features and correct jittering handling

    Key changes:
    1. Gaussian Fourier features (not deterministic logspace)
    2. Separate coords for modulation and decoding
    3. Learnable frequency matrices
    """

    def __init__(self, feature_dim=64, input_dim=2, output_dim=3,
                 sigma_q=16, sigma_ls=[128, 32], n_patches=256,
                 hidden_dim=256, context_dim=256,
                 learnable_frequencies=True):
        super().__init__()

        self.layer_num = len(sigma_ls)
        self.n_features = feature_dim // 2
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0

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

        # Architecture layers
        self.query_lin = nn.Linear(feature_dim, hidden_dim)
        self.modulation_ca = SharedTokenCrossAttention(
            query_dim=hidden_dim, context_dim=context_dim, heads=2
        )

        self.bandwidth_lins = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        self.modulation_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        self.hv_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layer_num - 1)
        ])

        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(self.layer_num)
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

    def forward(self, coords_decoding, tokens, coords_modulation=None):
        """
        Forward pass with separate modulation and decoding coordinates

        Args:
            coords_decoding: (B, H, W, 2) - where to predict RGB
            tokens: (B, L, D) - LP token features
            coords_modulation: (B, H, W, 2) - where to extract modulation
                              If None, use coords_decoding (test mode)
        """
        B, query_shape = coords_decoding.shape[0], coords_decoding.shape[1:-1]
        coords_dec = coords_decoding.view(B, -1, coords_decoding.shape[-1])

        # Determine modulation coordinates
        if coords_modulation is not None:
            # Training mode: use separate jittered coordinates
            coords_mod = coords_modulation.view(B, -1, coords_modulation.shape[-1])
        else:
            # Test mode: use same coordinates
            coords_mod = coords_dec

        # === MODULATION EXTRACTION (at coords_modulation) ===

        # Spatial bias
        grid_mod = coords_mod[0]
        indexes = self.get_patch_index(grid_mod, self.patch_num, self.patch_num)
        rel_distances = self.approximate_relative_distances(
            indexes, self.patch_num, self.patch_num, tokens.shape[1]
        )
        bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding with Gaussian Fourier features
        x_q = einops.repeat(
            gaussian_fourier_encode(coords_mod[0], self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        # Extract modulation
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # === DECODING (at coords_decoding) ===

        modulations_l = []
        for k in range(self.layer_num):
            # Bandwidth encoding with Gaussian Fourier features
            x_l = einops.repeat(
                gaussian_fourier_encode(coords_dec[0], self.B_ls[k]), 'l d -> b l d', b=B
            )
            h_l = self.act(self.bandwidth_lins[k](x_l))

            # Add modulation
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        # Residual connections
        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        # Multi-scale outputs
        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]
        out = sum(outs)
        out = out.view(B, *query_shape, -1)

        return out


print("✓ Gaussian Fourier LAINR decoder defined")
print("  - Uses Gaussian random features (not deterministic)")
print("  - Supports separate modulation/decoding coordinates")
print("  - Learnable frequency matrices")
