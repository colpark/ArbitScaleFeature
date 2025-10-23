"""
Masked Image Modeling with MAMBA-GINR (Proper Implementation)

Key corrections:
1. Use actual BiMamba encoder (not simplified Transformer)
2. Use LAINRDecoder for reconstruction
3. Extract pixel-level modulation features (32×32×hidden_dim)
4. Train CNN classifier on spatial modulation maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat
import math

from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from decoder_fix import LAINRDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Helper Functions
# ============================================================================

def fourier_encode(coords, n_features=32, std=10.0):
    """Fourier feature encoding for coordinates"""
    B = torch.randn(n_features, 2, device=coords.device) * std
    proj = 2 * math.pi * coords @ B.T
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def create_coordinate_grid(H, W, device='cpu'):
    """Create normalized coordinate grid in [0,1]"""
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([yy, xx], dim=-1)


def get_sinusoidal_embeddings(n, d):
    """Sinusoidal positional embeddings"""
    assert d % 2 == 0
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ============================================================================
# BiMamba Components
# ============================================================================

class BiMamba(nn.Module):
    """Bidirectional Mamba from MAMBA-GINR"""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.f_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.r_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x, inference_params=None):
        x_forward = self.f_mamba(x, inference_params=inference_params)
        x_backward = self.r_mamba(torch.flip(x, dims=[1]), inference_params=inference_params)
        x_backward = torch.flip(x_backward, dims=[1])
        x = torch.cat([x_forward, x_backward], dim=-1)
        return self.proj(x)


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks"""
    def __init__(self, depth=6, dim=256, ff_dim=1024, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                mixer_cls=lambda d: BiMamba(d_model=d),
                mlp_cls=lambda d: nn.Sequential(
                    nn.Linear(d, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d),
                    nn.Dropout(dropout),
                ),
                norm_cls=nn.LayerNorm,
                fused_add_norm=False
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual=residual, inference_params=None)
        return x


class LearnablePositionTokens(nn.Module):
    """
    LP tokens - FIXED for variable-length sequences

    Simplified version: LP tokens concatenated at END (not interleaved)
    This avoids index out of bounds errors with variable-length visible patches.
    """
    def __init__(self, num_tokens=256, dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim

        # Initialize with sinusoidal embeddings
        init_tokens = get_sinusoidal_embeddings(num_tokens, dim)
        self.tokens = nn.Parameter(init_tokens, requires_grad=True)

    def add_lp(self, x):
        """
        Add LP tokens to END of sequence

        Args:
            x: (B, L, D) input tokens (variable length L)
        Returns:
            (B, L+num_tokens, D) tokens with LP appended
        """
        B = x.shape[0]
        lps = repeat(self.tokens, 'n d -> b n d', b=B)
        return torch.cat([x, lps], dim=1)  # Concatenate at end

    def extract_lp(self, x):
        """
        Extract LP tokens from END of sequence

        Args:
            x: (B, L+num_tokens, D) encoded tokens
        Returns:
            (B, num_tokens, D) LP features
        """
        return x[:, -self.num_tokens:]  # Last num_tokens positions


# ============================================================================
# Modulation Feature Extractor
# ============================================================================

class ModulationExtractor(nn.Module):
    """
    Extract pixel-level modulation features from LAINRDecoder

    This wraps LAINRDecoder to expose the intermediate modulation features
    that are normally hidden inside the forward pass.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, coords, tokens, return_modulation=False):
        """
        Args:
            coords: (B, H, W, 2) query coordinates
            tokens: (B, L, D) LP token features
            return_modulation: if True, return modulation features instead of RGB

        Returns:
            if return_modulation:
                modulation: (B, H, W, hidden_dim) spatial modulation features
            else:
                rgb: (B, H, W, 3) predicted RGB values
        """
        B, H, W, _ = coords.shape
        coords_flat = coords.reshape(B, -1, 2)

        # Fourier encoding
        fourier_features = fourier_encode(coords_flat[0], self.decoder.n_features)
        fourier_features = repeat(fourier_features, 'n d -> b n d', b=B)

        # Query projection
        queries = F.relu(self.decoder.query_proj(fourier_features))

        # Spatial bias
        grid_coords = coords_flat[0]
        num_queries = grid_coords.shape[0]
        H_query = W_query = int(math.sqrt(num_queries))
        indices = self.decoder.get_patch_index(grid_coords, H_query, W_query)
        bias = self.decoder.compute_spatial_bias(indices, H_query, W_query, tokens.shape[1])

        # Extract modulation via cross-attention
        modulation = self.decoder.cross_attention(queries, tokens, bias)

        if return_modulation:
            # Return spatial modulation features for classification
            return modulation.reshape(B, H, W, -1)
        else:
            # Continue with RGB reconstruction
            decoder_input = torch.cat([fourier_features, modulation], dim=-1)
            features = self.decoder.decoder_blocks(decoder_input)
            rgb = self.decoder.output_proj(features)
            return rgb.reshape(B, H, W, 3)


# ============================================================================
# Masked MAMBA-GINR Model
# ============================================================================

class MaskedMAMBAGINR(nn.Module):
    """
    Masked Image Modeling with original MAMBA-GINR architecture

    Encoder: BiMamba (only processes visible patches)
    Decoder: LAINRDecoder (reconstructs all pixels)
    """
    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_channels=3,
                 dim=256,
                 num_lp=256,
                 mamba_depth=6,
                 ff_dim=1024,
                 hidden_dim=512,
                 n_features=32,
                 mask_ratio=0.5):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_num = img_size // patch_size
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, dim)

        # Patch positional encoding
        self.register_buffer('pos_freq', torch.randn(dim // 2, 2) * 10.0)
        self.pos_proj = nn.Linear(dim, dim)

        # Learnable position tokens (works with variable-length sequences)
        self.lp_module = LearnablePositionTokens(
            num_tokens=num_lp,
            dim=dim
        )

        # BiMamba encoder
        self.encoder = MambaEncoder(
            depth=mamba_depth,
            dim=dim,
            ff_dim=ff_dim
        )

        # LAINRDecoder for reconstruction
        self.decoder = LAINRDecoder(
            n_features=n_features,
            input_dim=2,
            output_dim=3,
            hidden_dim=hidden_dim,
            context_dim=dim,
            n_patches=self.num_patches
        )

        # Modulation extractor (wrapper around decoder)
        self.modulation_extractor = ModulationExtractor(self.decoder)

    def patchify(self, images):
        """Convert images to patches"""
        B, C, H, W = images.shape
        p = self.patch_size
        patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return patches

    def unpatchify(self, patches):
        """Convert patches back to images"""
        B, N, _ = patches.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        patches = patches.reshape(B, h, w, 3, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        images = patches.reshape(B, 3, h*p, w*p)
        return images

    def get_patch_positions(self, indices, device):
        """Get normalized positions for specific patch indices"""
        h = w = self.patch_num
        y_coords = (indices // w).float() / h + 0.5 / h
        x_coords = (indices % w).float() / w + 0.5 / w
        return torch.stack([y_coords, x_coords], dim=-1)

    def fourier_pos_encoding(self, positions):
        """Fourier positional encoding for patches"""
        proj = 2 * math.pi * positions @ self.pos_freq.T
        encoding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.pos_proj(encoding)

    def random_masking(self, B, device):
        """Generate random binary masks"""
        num_masked = int(self.num_patches * self.mask_ratio)
        masks = []
        visible_indices = []

        for _ in range(B):
            indices = torch.randperm(self.num_patches, device=device)
            mask = torch.ones(self.num_patches, device=device)
            mask[indices[:num_masked]] = 0  # 0 = masked, 1 = visible
            masks.append(mask)
            visible_indices.append(indices[num_masked:])  # Indices of visible patches

        return torch.stack(masks, dim=0), visible_indices

    def encode_visible_patches(self, patches, mask):
        """
        Encode ONLY visible patches with BiMamba

        Critical: This is the information bottleneck!
        """
        B = patches.shape[0]
        device = patches.device

        # Embed all patches first
        tokens = self.patch_embed(patches)  # (B, num_patches, dim)

        # Get visible patch indices
        _, visible_indices = self.random_masking(B, device)

        # Extract ONLY visible patches
        visible_tokens = []
        for i in range(B):
            visible_idx = mask[i].nonzero(as_tuple=True)[0]
            visible_tokens.append(tokens[i, visible_idx])

        # Stack (variable length per sample)
        max_visible = max(v.shape[0] for v in visible_tokens)
        tokens_visible = torch.zeros(B, max_visible, tokens.shape[-1], device=device)

        for i, v_tokens in enumerate(visible_tokens):
            tokens_visible[i, :v_tokens.shape[0]] = v_tokens

            # Add positional encoding for visible patches
            visible_idx = mask[i].nonzero(as_tuple=True)[0]
            positions = self.get_patch_positions(visible_idx, device)
            pos_encoding = self.fourier_pos_encoding(positions)
            tokens_visible[i, :v_tokens.shape[0]] = tokens_visible[i, :v_tokens.shape[0]] + pos_encoding

        # Add LP tokens (adapt to visible sequence length)
        # For simplicity, use fixed LP placement based on full sequence
        # In practice, could adapt LP insertion based on visible patches
        tokens_with_lp = self.lp_module.add_lp(tokens_visible)

        # Encode with BiMamba
        encoded = self.encoder(tokens_with_lp)

        # Extract LP features
        lp_features = self.lp_module.extract_lp(encoded)

        return lp_features

    def forward(self, images, return_modulation=False):
        """
        Forward pass

        Args:
            images: (B, 3, H, W)
            return_modulation: if True, return modulation features for classification

        Returns:
            if training (return_modulation=False):
                loss: reconstruction loss on masked regions
                pred_rgb: (B, 3, H, W) reconstructed images
                mask: (B, num_patches) binary mask
            if inference (return_modulation=True):
                modulation: (B, H, W, hidden_dim) spatial features for classification
        """
        B = images.shape[0]
        device = images.device

        if return_modulation:
            # Feature extraction mode (no masking)
            patches = self.patchify(images)
            tokens = self.patch_embed(patches)

            # Add positional encoding for all patches
            all_indices = torch.arange(self.num_patches, device=device).unsqueeze(0).expand(B, -1)
            positions = torch.stack([
                self.get_patch_positions(all_indices[i], device)
                for i in range(B)
            ], dim=0)
            pos_encoding = self.fourier_pos_encoding(positions.reshape(B * self.num_patches, 2))
            pos_encoding = pos_encoding.reshape(B, self.num_patches, -1)
            tokens = tokens + pos_encoding

            # Add LP tokens and encode
            tokens_with_lp = self.lp_module.add_lp(tokens)
            encoded = self.encoder(tokens_with_lp)
            lp_features = self.lp_module.extract_lp(encoded)

            # Extract modulation features at 32×32 grid
            coords = create_coordinate_grid(self.img_size, self.img_size, device)
            coords_batch = repeat(coords, 'h w d -> b h w d', b=B)
            modulation = self.modulation_extractor(coords_batch, lp_features, return_modulation=True)

            return modulation

        else:
            # Training mode (with masking)
            patches = self.patchify(images)  # (B, num_patches, patch_dim)

            # Random masking
            mask, _ = self.random_masking(B, device)  # (B, num_patches)

            # Encode ONLY visible patches → LP features
            lp_features = self.encode_visible_patches(patches, mask)

            # Decode at pixel grid
            coords = create_coordinate_grid(self.img_size, self.img_size, device)
            coords_batch = repeat(coords, 'h w d -> b h w d', b=B)
            pred_rgb = self.modulation_extractor(coords_batch, lp_features, return_modulation=False)

            # Compute loss on MASKED pixels only
            pred_rgb_recon = rearrange(pred_rgb, 'b h w c -> b c h w')
            loss = self.compute_masked_loss(images, pred_rgb_recon, mask)

            return loss, pred_rgb_recon, mask

    def compute_masked_loss(self, target, pred, patch_mask):
        """
        Compute reconstruction loss on masked pixels only

        Args:
            target: (B, 3, H, W) ground truth images
            pred: (B, 3, H, W) predicted images
            patch_mask: (B, num_patches) - 1 = visible, 0 = masked
        """
        B, C, H, W = target.shape
        p = self.patch_size

        # Convert patch mask to pixel mask
        patch_mask_spatial = patch_mask.reshape(B, self.patch_num, self.patch_num)
        pixel_mask = repeat(patch_mask_spatial, 'b h w -> b (h p1) (w p2)', p1=p, p2=p)
        pixel_mask = repeat(pixel_mask, 'b h w -> b c h w', c=C)

        # Compute loss only on masked pixels
        loss = ((pred - target) ** 2) * (1 - pixel_mask)
        loss = loss.sum() / ((1 - pixel_mask).sum() + 1e-8)

        return loss


# ============================================================================
# CNN Classifier for Spatial Modulation Features
# ============================================================================

class CNNClassifier(nn.Module):
    """
    CNN classifier for spatial modulation features

    Input: (B, H, W, hidden_dim) modulation features
    Output: (B, num_classes) logits
    """
    def __init__(self, hidden_dim=512, num_classes=10):
        super().__init__()

        # Rearrange to (B, C, H, W) for CNN processing
        # Input: (B, 32, 32, 512) → treat as (B, 512, 32, 32)

        self.conv_blocks = nn.Sequential(
            # Block 1: 512 → 256
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16

            # Block 2: 256 → 128
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 → 8

            # Block 3: 128 → 64
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # → 1×1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, modulation_features):
        """
        Args:
            modulation_features: (B, H, W, hidden_dim)
        Returns:
            logits: (B, num_classes)
        """
        # Rearrange to (B, C, H, W)
        x = modulation_features.permute(0, 3, 1, 2)

        # CNN processing
        x = self.conv_blocks(x)
        logits = self.classifier(x)

        return logits


print("✓ Masked MAMBA-GINR modules defined!")
print("  - BiMamba encoder (original architecture)")
print("  - LAINRDecoder for reconstruction")
print("  - Modulation extractor for pixel-level features")
print("  - CNN classifier for spatial modulation maps")
