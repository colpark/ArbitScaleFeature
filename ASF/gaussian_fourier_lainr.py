# LAINR Decoder with Gaussian Fourier Features
# This replaces deterministic Fourier features with random Gaussian features
# for better continuous position encoding and super-resolution

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


# ============================================================================
# Attention Module (unchanged)
# ============================================================================

class SharedTokenCrossAttention(nn.Module):
    """Cross-attention with spatial bias (unchanged from original)"""
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
    LAINR Hyponet with Gaussian Fourier Features

    Key changes from original:
    1. Random Gaussian frequency matrices instead of deterministic logspace
    2. Optional learnable frequency matrices
    3. Better continuous position encoding for super-resolution
    """

    def __init__(self, feature_dim=64, input_dim=2, output_dim=3,
                 sigma_q=16, sigma_ls=[128, 32], n_patches=256,
                 hidden_dim=256, context_dim=256,
                 learnable_frequencies=True,
                 frequency_init_scale=1.0):
        """
        Args:
            feature_dim: Dimension of Fourier features (must be divisible by 2)
            sigma_q: Scale parameter for query Gaussian features
            sigma_ls: Scale parameters for bandwidth Gaussian features (multi-scale)
            learnable_frequencies: If True, frequency matrices are learnable parameters
            frequency_init_scale: Multiplier for initial frequency variance
        """
        super().__init__()

        self.layer_num = len(sigma_ls)
        self.n_features = feature_dim // 2  # Number of random frequencies
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0
        self.learnable_frequencies = learnable_frequencies

        # Initialize Gaussian Fourier frequency matrices
        # B_q: (n_features, input_dim) - for query encoding
        # B_ls: List of (n_features, input_dim) - for multi-scale bandwidth encoding

        # Query frequency matrix
        B_q_init = torch.randn(self.n_features, input_dim) * frequency_init_scale / sigma_q

        # Multi-scale frequency matrices
        B_ls_init = [
            torch.randn(self.n_features, input_dim) * frequency_init_scale / sigma_ls[i]
            for i in range(self.layer_num)
        ]

        if learnable_frequencies:
            # Learnable frequency matrices (optimized during training)
            self.B_q = nn.Parameter(B_q_init)
            self.B_ls = nn.ParameterList([
                nn.Parameter(B_ls_init[i]) for i in range(self.layer_num)
            ])
        else:
            # Fixed random features
            self.register_buffer('B_q', B_q_init)
            for i in range(self.layer_num):
                self.register_buffer(f'B_l_{i}', B_ls_init[i])
            # Create list for convenient access
            self.B_ls = [getattr(self, f'B_l_{i}') for i in range(self.layer_num)]

        # Query encoding
        self.query_lin = nn.Linear(feature_dim, hidden_dim)

        # Cross-attention for modulation
        self.modulation_ca = SharedTokenCrossAttention(
            query_dim=hidden_dim, context_dim=context_dim, heads=2
        )

        # Bandwidth encoders (per frequency scale)
        self.bandwidth_lins = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        # Modulation projections
        self.modulation_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        # Hidden value layers (residual connections)
        self.hv_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layer_num - 1)
        ])

        # Output layers (one per scale)
        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(self.layer_num)
        ])

        self.act = nn.ReLU()

    def calc_gamma_gaussian(self, x, B_matrix):
        """
        Gaussian Fourier Features

        Args:
            x: (HW, input_dim) - coordinates
            B_matrix: (n_features, input_dim) - random frequency matrix

        Returns:
            gamma: (HW, 2*n_features) - Fourier features [cos(2πx·B), sin(2πx·B)]
        """
        # Project coordinates to random frequencies
        # (HW, input_dim) @ (input_dim, n_features) = (HW, n_features)
        proj = 2 * np.pi * x @ B_matrix.T

        # Compute cos and sin
        cos_features = torch.cos(proj)
        sin_features = torch.sin(proj)

        # Concatenate [cos, sin]
        gamma = torch.cat([cos_features, sin_features], dim=-1)  # (HW, 2*n_features)

        return gamma

    def get_patch_index(self, grid, H, W):
        """Convert coordinates to patch indices (unchanged)"""
        y = grid[:, 0]
        x = grid[:, 1]
        row = (y * H).to(torch.int32).clamp(0, H-1)
        col = (x * W).to(torch.int32).clamp(0, W-1)
        return row * W + col

    def approximate_relative_distances(self, target_index, H, W, m):
        """
        Compute spatial bias based on distance (unchanged)

        Returns:
            rel_distances: (m, HW) - bias matrix [LP_tokens × pixels]
        """
        alpha = self.alpha
        N = H * W

        t = target_index.float() / N  # (HW,)

        token_positions = torch.tensor(
            [(i + 0.5) / m for i in range(m)],
            device=target_index.device
        )  # (m,)

        # Broadcast to create (m, HW) matrix
        t_expanded = t.unsqueeze(0)  # (1, HW)
        tokens_expanded = token_positions.unsqueeze(1)  # (m, 1)

        # Distance-based bias
        rel_distances = -alpha * torch.abs(t_expanded - tokens_expanded)**2

        return rel_distances

    def forward(self, x, tokens):
        """
        Args:
            x: (B, H, W, 2) or (B, HW, 2) - coordinate grid
            tokens: (B, L, D) - LP token features

        Returns:
            out: (B, H, W, 3) - RGB output
        """
        B, query_shape = x.shape[0], x.shape[1:-1]
        x = x.view(B, -1, x.shape[-1])  # (B, HW, 2)

        # Get first batch item for spatial computations
        grid = x[0]
        indexes = self.get_patch_index(grid, self.patch_num, self.patch_num)

        # Compute spatial bias
        rel_distances = self.approximate_relative_distances(
            indexes, self.patch_num, self.patch_num, tokens.shape[1]
        )
        # rel_distances is already (L, HW), don't transpose!
        bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding with Gaussian Fourier features
        x_q = einops.repeat(
            self.calc_gamma_gaussian(x[0], self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        # Extract modulation via cross-attention with spatial bias
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # Multi-scale processing with Gaussian Fourier features
        modulations_l = []

        for k in range(self.layer_num):
            # Encode at each frequency scale using Gaussian features
            if self.learnable_frequencies:
                B_matrix = self.B_ls[k]
            else:
                B_matrix = getattr(self, f'B_l_{k}')

            x_l = einops.repeat(
                self.calc_gamma_gaussian(x[0], B_matrix), 'l d -> b l d', b=B
            )
            h_l = self.act(self.bandwidth_lins[k](x_l))

            # Add modulation
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        # Residual connections across scales
        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        # Multi-scale outputs (summed)
        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]
        out = sum(outs)

        out = out.view(B, *query_shape, -1)
        return out


# ============================================================================
# Hybrid Version: Deterministic + Gaussian
# ============================================================================

class LAINRDecoderHybrid(nn.Module):
    """
    Hybrid approach: Combine deterministic and Gaussian Fourier features

    Benefits:
    - Deterministic: Ensures specific frequencies are present
    - Gaussian: Fills gaps with continuous coverage
    """

    def __init__(self, feature_dim=64, input_dim=2, output_dim=3,
                 sigma_q=16, sigma_ls=[128, 32], n_patches=256,
                 hidden_dim=256, context_dim=256,
                 deterministic_ratio=0.25):
        """
        Args:
            deterministic_ratio: Fraction of features that are deterministic
                                (rest are Gaussian random)
        """
        super().__init__()

        self.layer_num = len(sigma_ls)
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0

        # Split features between deterministic and random
        n_total = feature_dim // 4  # Per coord, per sin/cos
        self.n_deterministic = int(n_total * deterministic_ratio)
        self.n_random = n_total - self.n_deterministic

        # Deterministic frequencies (logspace)
        self.register_buffer('omegas_q',
                           torch.logspace(1, math.log10(sigma_q), self.n_deterministic))
        self.omegas_ls = []
        for i in range(self.layer_num):
            omega = torch.logspace(1, math.log10(sigma_ls[i]), self.n_deterministic)
            self.register_buffer(f'omegas_l_{i}', omega)
            self.omegas_ls.append(getattr(self, f'omegas_l_{i}'))

        # Gaussian frequencies
        self.register_buffer('B_q', torch.randn(self.n_random, input_dim) / sigma_q)
        for i in range(self.layer_num):
            B = torch.randn(self.n_random, input_dim) / sigma_ls[i]
            self.register_buffer(f'B_l_{i}', B)

        # Rest of architecture (same as Gaussian version)
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

    def calc_gamma_hybrid(self, x, omegas_det, B_gauss):
        """
        Hybrid Fourier features: deterministic + Gaussian

        Args:
            x: (HW, 2) - coordinates
            omegas_det: (n_det,) - deterministic frequencies
            B_gauss: (n_random, 2) - Gaussian frequency matrix

        Returns:
            gamma: (HW, feature_dim) - hybrid features
        """
        # Deterministic component
        coords_expanded = x.unsqueeze(-1)  # (HW, 2, 1)
        omegas_expanded = omegas_det.view(1, 1, -1).to(x.device)  # (1, 1, n_det)

        arg_det = torch.pi * coords_expanded * omegas_expanded  # (HW, 2, n_det)
        gamma_det = torch.cat([torch.sin(arg_det), torch.cos(arg_det)], dim=-1)
        gamma_det = gamma_det.view(x.shape[0], -1)  # (HW, 4*n_det)

        # Gaussian component
        proj_gauss = 2 * np.pi * x @ B_gauss.T  # (HW, n_random)
        gamma_gauss = torch.cat([torch.cos(proj_gauss), torch.sin(proj_gauss)],
                               dim=-1)  # (HW, 2*n_random)

        # Concatenate
        gamma = torch.cat([gamma_det, gamma_gauss], dim=-1)

        return gamma

    def get_patch_index(self, grid, H, W):
        y = grid[:, 0]
        x = grid[:, 1]
        row = (y * H).to(torch.int32).clamp(0, H-1)
        col = (x * W).to(torch.int32).clamp(0, W-1)
        return row * W + col

    def approximate_relative_distances(self, target_index, H, W, m):
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

    def forward(self, x, tokens):
        B, query_shape = x.shape[0], x.shape[1:-1]
        x = x.view(B, -1, x.shape[-1])

        grid = x[0]
        indexes = self.get_patch_index(grid, self.patch_num, self.patch_num)
        rel_distances = self.approximate_relative_distances(
            indexes, self.patch_num, self.patch_num, tokens.shape[1]
        )
        bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding with hybrid features
        x_q = einops.repeat(
            self.calc_gamma_hybrid(x[0], self.omegas_q, self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        modulations_l = []
        for k in range(self.layer_num):
            B_matrix = getattr(self, f'B_l_{k}')
            omegas = getattr(self, f'omegas_l_{k}')

            x_l = einops.repeat(
                self.calc_gamma_hybrid(x[0], omegas, B_matrix), 'l d -> b l d', b=B
            )
            h_l = self.act(self.bandwidth_lins[k](x_l))
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]
        out = sum(outs)
        out = out.view(B, *query_shape, -1)

        return out


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Gaussian Fourier Features (learnable)
    decoder_gaussian = LAINRDecoderGaussian(
        feature_dim=64,
        sigma_q=16,
        sigma_ls=[128, 32],
        n_patches=256,
        hidden_dim=256,
        learnable_frequencies=True
    )

    # Example 2: Gaussian Fourier Features (fixed random)
    decoder_gaussian_fixed = LAINRDecoderGaussian(
        feature_dim=64,
        sigma_q=16,
        sigma_ls=[128, 32],
        n_patches=256,
        hidden_dim=256,
        learnable_frequencies=False
    )

    # Example 3: Hybrid (deterministic + Gaussian)
    decoder_hybrid = LAINRDecoderHybrid(
        feature_dim=64,
        sigma_q=16,
        sigma_ls=[128, 32],
        n_patches=256,
        hidden_dim=256,
        deterministic_ratio=0.25  # 25% deterministic, 75% Gaussian
    )

    # Test forward pass
    batch_size = 4
    coords = torch.rand(batch_size, 32, 32, 2)  # Random coordinates
    tokens = torch.randn(batch_size, 256, 256)  # LP tokens

    output_gaussian = decoder_gaussian(coords, tokens)
    print(f"Gaussian output shape: {output_gaussian.shape}")  # (4, 32, 32, 3)

    output_hybrid = decoder_hybrid(coords, tokens)
    print(f"Hybrid output shape: {output_hybrid.shape}")  # (4, 32, 32, 3)

    print("\n✓ Gaussian Fourier Features implemented successfully!")
    print("\nKey benefits:")
    print("  1. Continuous frequency coverage (no gaps)")
    print("  2. Better interpolation for super-resolution")
    print("  3. Can learn optimal frequency distribution")
    print("  4. No computational overhead vs deterministic")
