# MAMBA-GINR on CIFAR-10 - Standalone Implementation

## Overview

This folder contains a clean, self-contained implementation of MAMBA-GINR for CIFAR-10 image reconstruction.

**Target Performance: 60 PSNR**

## Architecture Details

### 1. Encoder (BiMamba)
```
Input: 32×32×3 images
  ↓
Patchify: 2×2 patches → 256 patch tokens
  ↓
Patch Embedding: Linear(12 → 256)
  ↓
Fourier Positional Encoding: patch center positions
  ↓
Add LP Tokens: 256 learnable position tokens (equidistant)
  ↓ Total: 256 patch + 256 LP = 512 tokens
BiMamba Encoder: 6 layers
  ↓
Extract LP Tokens: last 256 tokens
  ↓
Output: (B, 256, 256) latent features
```

### 2. Decoder (LAINR)
```
Query Coordinates: (H, W, 2) in [0, 1]
  ↓
Fourier Encoding: 32 frequency components
  ↓
Query Projection: Linear(64 → 512)
  ↓
Cross-Attention: queries attend to LP tokens (with spatial bias)
  ↓
Decoder: 3 ResidualBlocks (512 → 512 → 512)
  ↓
Output: Linear(512 → 3) RGB
```

## Key Components

### BiMamba
- Forward + Backward Mamba processing
- Concatenate outputs → Linear projection
- O(L) complexity vs O(L²) for attention

### Learnable Position Tokens (LP)
- **Count**: 256 tokens
- **Initialization**: Sinusoidal embeddings
- **Placement**: Equidistant insertion among patch tokens
- **Purpose**: Implicit sequential bias for position information

### LAINR Decoder
- **Fourier Features**: 32 frequency components, std=10.0
- **Spatial Bias**: `-10.0 × |distance|²` for attention
- **Architecture**: ResidualBlock-based (NOT Attention+FeedForward)
- **Hidden Dim**: 512

## Hyperparameters

### Model
- `img_size=32`
- `patch_size=2`
- `dim=256` (encoder hidden dim)
- `num_lp=256` (learnable position tokens)
- `mamba_depth=6` (encoder layers)
- `ff_dim=1024` (feedforward expansion)
- `hidden_dim=512` (decoder hidden dim)
- `n_features=32` (Fourier frequency components)

### Training
- `batch_size=64`
- `lr=5e-4` (base learning rate)
- `weight_decay=1e-4`
- `num_epochs=40`
- `warmup_epochs=5`
- `scheduler`: Cosine annealing after warmup
- `optimizer`: AdamW
- `grad_clip=1.0`

### Loss
- MSE loss on RGB values
- PSNR = -10 × log10(MSE)

## Files

- `mamba_ginr_cifar10.ipynb` - Complete training notebook (13 cells)
  - Cell 1: Imports
  - Cell 2: Helper functions
  - Cell 3: BiMamba
  - Cell 4: MambaEncoder
  - Cell 5: Learnable Position Tokens
  - Cell 6: LAINR Decoder (ResidualBlock-based)
  - Cell 7: Complete MAMBA-GINR model
  - Cell 8: Training functions
  - Cell 9: Data loading
  - Cell 10: Model initialization
  - Cell 11: Training loop (40 epochs)
  - Cell 12: Super-resolution test
  - Cell 13: Final analysis

## Usage

```bash
# Install dependencies
pip install torch torchvision mamba-ssm einops tqdm matplotlib

# Run notebook
jupyter notebook mamba_ginr_cifar10.ipynb
```

## Expected Results

### Training Convergence
- **Epoch 5**: ~25 dB PSNR
- **Epoch 10**: ~30 dB PSNR
- **Epoch 20**: ~35 dB PSNR
- **Epoch 40**: ~40-50 dB PSNR

**Target: 60 PSNR** requires near-perfect reconstruction (MSE < 1e-6)

### Super-Resolution
- Train at 32×32
- Generate at 64×64, 128×128, 256×256
- Continuous coordinate decoding

## Implementation Notes

### Why 60 PSNR is Challenging

60 dB PSNR means MSE ≈ 1e-6, which requires:
1. Perfect pixel-level reconstruction
2. No quantization errors
3. Exact coordinate mapping
4. High decoder capacity

### Original vs This Implementation

**Preserved from Original:**
- ✅ BiMamba encoder with LP tokens
- ✅ Equidistant LP placement
- ✅ Fourier positional encoding
- ✅ LAINR-style decoder
- ✅ Spatial bias in cross-attention
- ✅ ResidualBlocks (not SCENT-style)
- ✅ AdamW + cosine annealing

**Simplified:**
- Uses standard Fourier encoding (not Gaussian)
- Fixed frequency std (not learnable)
- No jittering (exact pixel centers)

### Comparison with Other Notebooks

| Notebook | Decoder | Fourier | Jittering | Target PSNR |
|----------|---------|---------|-----------|-------------|
| `mamba_ginr_cifar10.ipynb` | ResidualBlock | Standard | No | 60 dB |
| `mamba_ginr_gaussian_fourier.ipynb` | ResidualBlock | Gaussian | Yes | ~40 dB |
| `mamba_ginr_scent_decoder.ipynb` | SCENT-style | Gaussian | Yes | ~30 dB |
| `mamba_scent_hybrid.ipynb` | SCENT-style | Gaussian | Yes | ~32 dB |

## Key Differences Explained

### Why ResidualBlock for 60 PSNR?

For **exact reconstruction** (60 dB), the decoder needs:
- Local, precise processing (ResidualBlock)
- No stochastic components (no dropout, no jittering)
- Deterministic mapping (fixed Fourier frequencies)

For **super-resolution** (32 dB), different decoder needed:
- Global receptive field (Attention)
- Stochastic training (jittering, Gaussian frequencies)
- High-capacity blocks (SCENT-style)

This notebook focuses on **exact reconstruction**, not super-resolution.

## Model Size

- **Total parameters**: ~15-20M
  - Encoder: ~8-10M
  - Decoder: ~5-8M
  - LP tokens: ~65K (256 × 256)

## Memory Requirements

- **Training**: ~6-8 GB GPU memory (batch_size=64)
- **Inference**: ~2 GB

## Outputs

- `mamba_ginr_best.pth` - Best model checkpoint
- `mamba_ginr_super_resolution.png` - Super-resolution visualization

## Citation

If you use this implementation, please cite the original MAMBA-GINR paper.

## Notes

This implementation prioritizes:
1. **Clarity**: Clean, readable code
2. **Exactness**: Faithful to original architecture
3. **Simplicity**: Minimal dependencies
4. **Performance**: Targets 60 PSNR benchmark
