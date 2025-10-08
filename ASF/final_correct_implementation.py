# FINAL CORRECT IMPLEMENTATION
# Based on your exact specifications:
# (1) Gaussian Fourier features everywhere
# (2) Sub-pixel jittering (max < 1/(2*resolution))
# (3) Separate modulation/decoding coordinates
# (4) No jittering at test time

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np
from tqdm import tqdm


# ============================================================================
# Gaussian Fourier Feature Encoding
# ============================================================================

def gaussian_fourier_encode(coords, B_matrix):
    """
    Gaussian Fourier Feature encoding

    Args:
        coords: (HW, 2) or (B, HW, 2) - coordinates in [0, 1]
        B_matrix: (n_features, 2) - random frequency matrix

    Returns:
        features: (HW, 2*n_features) or (B, HW, 2*n_features)
    """
    # Handle batched or unbatched input
    if coords.ndim == 2:
        # (HW, 2) @ (2, n_features) = (HW, n_features)
        proj = 2 * np.pi * coords @ B_matrix.T
    else:
        # (B, HW, 2) @ (2, n_features) = (B, HW, n_features)
        proj = 2 * np.pi * coords @ B_matrix.T

    # [cos(proj), sin(proj)]
    features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    return features


# ============================================================================
# LAINR Decoder with Gaussian Fourier Features
# ============================================================================

class LAINRDecoderGaussianCorrected(nn.Module):
    """
    LAINR with Gaussian Fourier features and correct jittering handling
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

        from gaussian_fourier_lainr import SharedTokenCrossAttention
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

        Returns:
            out: (B, H, W, 3) - predicted RGB
        """
        B, query_shape = coords_decoding.shape[0], coords_decoding.shape[1:-1]
        coords_dec = coords_decoding.view(B, -1, coords_decoding.shape[-1])  # (B, HW, 2)

        # Determine modulation coordinates
        if coords_modulation is not None:
            # Training mode: use separate jittered coordinates
            coords_mod = coords_modulation.view(B, -1, coords_modulation.shape[-1])
        else:
            # Test mode: use same coordinates
            coords_mod = coords_dec

        # === MODULATION EXTRACTION (at coords_modulation) ===

        # Spatial bias (based on modulation coordinates)
        grid_mod = coords_mod[0]
        indexes = self.get_patch_index(grid_mod, self.patch_num, self.patch_num)
        rel_distances = self.approximate_relative_distances(
            indexes, self.patch_num, self.patch_num, tokens.shape[1]
        )
        bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding (at modulation coordinates)
        x_q = einops.repeat(
            gaussian_fourier_encode(coords_mod[0], self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        # Extract modulation via cross-attention
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # === DECODING (at coords_decoding) ===

        modulations_l = []
        for k in range(self.layer_num):
            # Bandwidth encoding (at DECODING coordinates)
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


# ============================================================================
# Training with Correct Jittering
# ============================================================================

def create_coordinate_grid(H, W, device='cpu'):
    """Create normalized coordinate grid [0, 1]"""
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
    return coords


def train_epoch_final(model, loader, optimizer, device, epoch,
                      resolution=32,
                      jitter_std=None,      # Auto-computed if None
                      offset_std=0.0):      # Prediction offset (default: 0)
    """
    Training with correct jittering specification

    Args:
        jitter_std: Std of modulation jittering. If None, set to 1/(6*resolution)
                    to ensure max jitter < 1/(2*resolution) with 3-sigma rule
        offset_std: Std of prediction offset (default: 0)
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    # Auto-compute jittering std if not provided
    if jitter_std is None:
        pixel_size = 1.0 / resolution
        max_allowed_jitter = pixel_size / 2
        jitter_std = max_allowed_jitter / 3  # 3-sigma rule

    # Validate constraint
    pixel_size = 1.0 / resolution
    max_allowed_jitter = pixel_size / 2
    if jitter_std > max_allowed_jitter / 3:
        print(f"WARNING: jitter_std={jitter_std:.6f} may violate sub-pixel constraint!")
        print(f"         Recommended max: {max_allowed_jitter/3:.6f}")

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Create base coordinate grid INSIDE loop (critical!)
        base_coords = create_coordinate_grid(resolution, resolution, device)

        # (1) Apply small jittering for modulation
        jitter_small = torch.randn_like(base_coords) * jitter_std
        coords_modulation = (base_coords + jitter_small).clamp(0, 1)

        # (2) Apply prediction offset (default: 0)
        if offset_std > 0:
            prediction_offset = torch.randn_like(base_coords) * offset_std
            coords_decoding = (coords_modulation + prediction_offset).clamp(0, 1)
        else:
            coords_decoding = coords_modulation

        # Repeat for batch
        coords_mod_batch = einops.repeat(coords_modulation, 'h w d -> b h w d', b=B)
        coords_dec_batch = einops.repeat(coords_decoding, 'h w d -> b h w d', b=B)

        # Forward pass
        lp_features = model.encode(images)
        pred = model.hyponet(coords_dec_batch, lp_features,
                            coords_modulation=coords_mod_batch)

        # Ground truth
        gt = einops.rearrange(images, 'b c h w -> b h w c')

        # Loss
        mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'psnr': f"{psnr.item():.2f}",
            'jitter': f"σ={jitter_std:.5f}",
            'offset': f"σ={offset_std:.5f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


def validate_final(model, loader, device, resolution=32):
    """
    Validation with NO jittering
    """
    model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            B = images.shape[0]

            # Exact coordinates (no jittering)
            coords = create_coordinate_grid(resolution, resolution, device)
            coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

            # Forward pass (coords_modulation=None → test mode)
            lp_features = model.encode(images)
            pred = model.hyponet(coords_batch, lp_features,
                                coords_modulation=None)

            # Ground truth
            gt = einops.rearrange(images, 'b c h w -> b h w c')

            # Metrics
            mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses)).mean()

            total_loss += loss.item()
            total_psnr += psnr.item()

    return total_loss / len(loader), total_psnr / len(loader)


def super_resolve_final(model, images, target_resolution=128, device='cpu'):
    """
    Super-resolution with NO jittering
    """
    model.eval()

    with torch.no_grad():
        B = images.shape[0]

        # Encode at training resolution
        lp_features = model.encode(images)

        # Decode at target resolution (no jittering)
        coords = create_coordinate_grid(target_resolution, target_resolution, device)
        coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

        # Predict (coords_modulation=None → test mode)
        pred = model.hyponet(coords_batch, lp_features, coords_modulation=None)

        # Rearrange to image format
        pred_images = einops.rearrange(pred, 'b h w c -> b c h w')

    return pred_images


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FINAL CORRECT IMPLEMENTATION")
    print("="*70)

    print("\nSpecifications:")
    print("  (1) Gaussian Fourier features for ALL position encoding")
    print("  (2) Jittering constraint: max < 1/(2*resolution)")
    print("  (3) Separate modulation/decoding coordinates")
    print("  (4) No jittering at test time")

    print("\nDefault jittering parameters for 32×32:")
    resolution = 32
    pixel_size = 1/resolution
    max_jitter = pixel_size / 2
    jitter_std = max_jitter / 3
    print(f"  Pixel size: {pixel_size:.6f}")
    print(f"  Max jitter: {max_jitter:.6f}")
    print(f"  Jitter std: {jitter_std:.6f}")
    print(f"  This ensures 99.7% of jitter stays within ±{max_jitter:.6f}")

    print("\n" + "="*70)
    print("✓ Implementation ready!")
    print("="*70)
