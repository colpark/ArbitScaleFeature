"""
Fixed LAINRDecoder - Import this instead of using the notebook cell
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat


def fourier_encode(coords, n_features=32, std=10.0):
    """Fourier feature encoding for coordinates"""
    B = torch.randn(n_features, 2, device=coords.device) * std
    proj = 2 * math.pi * coords @ B.T
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Simple residual block from original LAINR"""
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class LAINRDecoder(nn.Module):
    """
    FIXED LAINR-style decoder with ResidualBlocks
    """
    def __init__(self, n_features=32, input_dim=2, output_dim=3,
                 hidden_dim=512, context_dim=256, n_patches=256):
        super().__init__()

        self.n_features = n_features
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0  # Spatial bias coefficient

        # Fourier encoding frequencies
        self.register_buffer('B', torch.randn(n_features, input_dim) * 10.0)
        feature_dim = 2 * n_features

        # Query encoding
        self.query_proj = nn.Linear(feature_dim, hidden_dim)

        # Cross-attention for modulation extraction
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_kv = nn.Linear(context_dim, hidden_dim * 2)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // 2) ** -0.5

        # Decoder processing
        self.decoder_blocks = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def get_patch_index(self, coords, H, W):
        """Convert coordinates to patch indices"""
        y, x = coords[:, 0], coords[:, 1]
        row = (y * H).long().clamp(0, H-1)
        col = (x * W).long().clamp(0, W-1)
        return row * W + col

    def compute_spatial_bias(self, target_index, H, W, num_tokens):
        """
        Compute spatial bias for attention
        Returns: (num_tokens, num_queries)
        """
        N = H * W
        t = target_index.float() / N
        token_positions = torch.linspace(0.5/num_tokens, 1 - 0.5/num_tokens,
                                        num_tokens, device=target_index.device)
        distances = torch.abs(t.unsqueeze(0) - token_positions.unsqueeze(1))
        return -self.alpha * distances**2

    def cross_attention(self, queries, context, bias=None):
        """
        FIXED: Cross-attention with optional spatial bias

        Args:
            queries: (B, num_queries, D)
            context: (B, num_tokens, D)
            bias: (num_tokens, num_queries) spatial bias
        """
        B, N, D = queries.shape

        q = self.to_q(queries)  # (B, num_queries, D)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # (B, num_tokens, D)

        # sim shape: (B, num_queries, num_tokens)
        sim = torch.einsum('bnd,bld->bnl', q, k) * self.scale

        if bias is not None:
            # CRITICAL FIX: bias is (num_tokens, num_queries) but sim is (B, num_queries, num_tokens)
            # Must transpose from (256, 1024) to (1024, 256) then add batch dim
            bias_corrected = bias.transpose(0, 1).unsqueeze(0)  # (1, num_queries, num_tokens)
            sim = sim + bias_corrected

        attn = sim.softmax(dim=-1)
        out = torch.einsum('bnl,bld->bnd', attn, v)
        return self.attn_out(out)

    def forward(self, coords, tokens):
        """
        Args:
            coords: (B, H, W, 2) query coordinates
            tokens: (B, L, D) LP token features
        Returns:
            rgb: (B, H, W, 3) predicted RGB values
        """
        B, H, W, _ = coords.shape
        coords_flat = coords.reshape(B, -1, 2)

        # Fourier encoding
        fourier_features = fourier_encode(coords_flat[0], self.n_features)
        fourier_features = repeat(fourier_features, 'n d -> b n d', b=B)

        # Query projection
        queries = F.relu(self.query_proj(fourier_features))

        # Spatial bias - use actual query grid dimensions
        grid_coords = coords_flat[0]
        num_queries = grid_coords.shape[0]
        H_query = W_query = int(math.sqrt(num_queries))
        indices = self.get_patch_index(grid_coords, H_query, W_query)
        bias = self.compute_spatial_bias(indices, H_query, W_query, tokens.shape[1])

        # Extract modulation via cross-attention
        modulation = self.cross_attention(queries, tokens, bias)

        # Decode
        decoder_input = torch.cat([fourier_features, modulation], dim=-1)
        features = self.decoder_blocks(decoder_input)
        rgb = self.output_proj(features)

        return rgb.reshape(B, H, W, 3)
