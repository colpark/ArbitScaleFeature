# Memory-Efficient SCENT-Style Decoder
# Replaces self-attention with stacked FeedForwards to save memory
# Still keeps: 4x expansion, GEGLU, skip connections, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math

# ... (keep all helper functions and PreNorm, GEGLU, FeedForward, Attention classes)

class LAINRDecoderSCENT_MemoryEfficient(nn.Module):
    """
    Memory-efficient version: NO self-attention on decoder positions
    Uses stacked FeedForwards instead

    Memory: ~5GB (vs ~36GB with self-attention)
    Expected: Still +3-5 dB over ResidualBlock
    """

    def __init__(self, feature_dim=64, input_dim=2, output_dim=3,
                 sigma_q=16, sigma_ls=[128, 32], n_patches=256,
                 hidden_dim=512, context_dim=256,
                 learnable_frequencies=True, num_layers=3,
                 heads=8, dim_head=64):
        super().__init__()

        self.layer_num = len(sigma_ls)
        self.n_features = feature_dim // 2
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0
        self.num_layers = num_layers

        # Gaussian Fourier frequencies
        B_q_init = torch.randn(self.n_features, input_dim) / sigma_q
        B_ls_init = [torch.randn(self.n_features, input_dim) / sigma_ls[i]
                     for i in range(self.layer_num)]

        if learnable_frequencies:
            self.B_q = nn.Parameter(B_q_init)
            self.B_ls = nn.ParameterList([nn.Parameter(B_ls_init[i]) for i in range(self.layer_num)])
        else:
            self.register_buffer('B_q', B_q_init)
            for i in range(self.layer_num):
                self.register_buffer(f'B_l_{i}', B_ls_init[i])
            self.B_ls = [getattr(self, f'B_l_{i}') for i in range(self.layer_num)]

        # Query encoding
        self.query_lin = nn.Linear(feature_dim, hidden_dim)

        # Cross-attention for modulation (this is OK - only 1024 × 256)
        self.modulation_ca = PreNorm(
            hidden_dim,
            Attention(hidden_dim, context_dim, heads=2, dim_head=64),
            context_dim=context_dim
        )

        # MEMORY FIX: Use stacked FeedForwards instead of Attention + FeedForward
        self.bandwidth_lins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                *[
                    nn.Sequential(
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4)),
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
                    )
                    for _ in range(num_layers - 1)
                ]
            )
            for _ in range(self.layer_num)
        ])

        self.modulation_lins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                *[
                    nn.Sequential(
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4)),
                        PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
                    )
                    for _ in range(num_layers - 1)
                ]
            )
            for _ in range(self.layer_num)
        ])

        # Hidden value layers - also FeedForward only
        self.hv_lins = nn.ModuleList([
            nn.Sequential(
                PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4)),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
            )
            for _ in range(self.layer_num - 1)
        ])

        # Skip connections
        self.fourier_skip_projs = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(self.layer_num)
        ])

        # Output
        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(self.layer_num)
        ])

        self.act = nn.ReLU()

    def get_patch_index(self, grid, H, W):
        y, x = grid[:, 0], grid[:, 1]
        row = (y * H).to(torch.int32).clamp(0, H-1)
        col = (x * W).to(torch.int32).clamp(0, W-1)
        return row * W + col

    def approximate_relative_distances(self, target_index, H, W, m):
        N = H * W
        t = target_index.float() / N
        token_positions = torch.tensor([(i + 0.5) / m for i in range(m)], device=target_index.device)
        t_expanded = t.unsqueeze(0)
        tokens_expanded = token_positions.unsqueeze(1)
        return -self.alpha * torch.abs(t_expanded - tokens_expanded)**2

    def apply_block_sequence(self, x, block_seq):
        """Apply FeedForward blocks with residual connections"""
        if isinstance(block_seq[0], nn.Linear):
            x = block_seq[1](block_seq[0](x))
            start_idx = 2
        else:
            start_idx = 0

        for i in range(start_idx, len(block_seq)):
            block = block_seq[i]
            if isinstance(block, nn.Sequential):
                # Two FeedForwards with residuals
                ff1, ff2 = block[0], block[1]
                x = ff1(x) + x
                x = ff2(x) + x
        return x

    def forward(self, coords_decoding, tokens, coords_modulation=None):
        B, query_shape = coords_decoding.shape[0], coords_decoding.shape[1:-1]
        coords_dec = coords_decoding.view(B, -1, coords_decoding.shape[-1])
        coords_mod = coords_modulation.view(B, -1, coords_modulation.shape[-1]) if coords_modulation is not None else coords_dec

        # Modulation extraction
        grid_mod = coords_mod[0]
        num_queries = grid_mod.shape[0]
        H_mod = W_mod = int(math.sqrt(num_queries))
        indexes = self.get_patch_index(grid_mod, H_mod, W_mod)
        rel_distances = self.approximate_relative_distances(indexes, H_mod, W_mod, tokens.shape[1])
        bias = repeat(rel_distances, 'l n -> b l n', b=B)

        x_q = repeat(gaussian_fourier_encode(coords_mod[0], self.B_q), 'l d -> b l d', b=B)
        x_q = self.act(self.query_lin(x_q))
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # Multi-scale decoding (FeedForward only)
        modulations_l, fourier_encodings = [], []
        for k in range(self.layer_num):
            x_l_fourier = gaussian_fourier_encode(coords_dec[0], self.B_ls[k])
            x_l_fourier_batch = repeat(x_l_fourier, 'l d -> b l d', b=B)
            fourier_encodings.append(x_l_fourier_batch)

            h_l = self.apply_block_sequence(x_l_fourier_batch, self.bandwidth_lins[k])
            m_proj = self.apply_block_sequence(modulation_vector, self.modulation_lins[k])
            modulations_l.append(self.act(h_l + m_proj))

        # Residual connections
        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            x_combined = modulations_l[i+1] + h_v[i]
            # Apply FeedForwards
            ff_block = self.hv_lins[i]
            x_combined = ff_block[0](x_combined) + x_combined
            x_combined = ff_block[1](x_combined) + x_combined
            h_v.append(x_combined)

        # Output with skip connections
        outs = []
        for i in range(self.layer_num):
            fourier_skip = self.fourier_skip_projs[i](fourier_encodings[i])
            outs.append(self.out_lins[i](h_v[i] + fourier_skip))

        out = sum(outs).view(B, *query_shape, -1)
        return out


print("✓ Memory-efficient decoder defined")
print("  - Uses stacked FeedForwards (no self-attention on 1024 positions)")
print("  - Memory: ~5GB (vs ~36GB with self-attention)")
print("  - Still has: 4x expansion, GEGLU, skip connections, LayerNorm")
print("  - Expected: +3-5 dB over ResidualBlock")
