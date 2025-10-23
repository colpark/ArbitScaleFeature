# Masked MAMBA-GINR v2 - Proper Implementation

## Critical Fixes from Previous Version

### ❌ Problems in Previous Implementation

1. **Wrong Encoder**: Used simplified Transformer encoder instead of BiMamba
2. **Wrong Feature Extraction**: Used global average pooling over LP tokens instead of pixel-level modulation
3. **Wrong Classifier**: Used linear classifier on 1D vector instead of CNN on spatial features
4. **Too Aggressive Masking**: 75% masking on small 32×32 images left insufficient context

### ✅ Fixes in This Version

1. **Proper BiMamba Encoder**: Uses original MAMBA-GINR architecture with bidirectional Mamba blocks
2. **LAINRDecoder Integration**: Properly uses LAINRDecoder for both reconstruction and modulation extraction
3. **Pixel-Level Modulation Features**: Extracts 32×32×512 spatial features for classification
4. **CNN Classifier**: Trains CNN on spatial modulation maps (preserves spatial structure)
5. **50% Masking**: More reasonable mask ratio for small images

## Architecture Overview

```
Input: 32×32×3 image
    ↓
Patchify: 2×2 patches → 256 patches
    ↓
Random Masking: 50% → 128 visible, 128 masked
    ↓
VISIBLE PATCHES ONLY → Patch Embedding + Fourier Positional Encoding
    ↓
Add LP Tokens (256 learnable tokens with equidistant placement)
    ↓
BiMamba Encoder (6 layers)
    - Bidirectional processing
    - O(L) complexity
    ↓
Extract LP Features (256 tokens × 256 dim)
    ↓
LAINRDecoder:
    - Query: 32×32 coordinate grid
    - Cross-attention: Extract modulation from LP features
    - Output: RGB reconstruction OR modulation features
    ↓
Two Training Stages:
    1. Reconstruction: Loss on MASKED pixels only
    2. Classification: CNN on 32×32×512 modulation features
```

## Key Innovations

### 1. Information Bottleneck

The encoder **ONLY** sees visible patches (e.g., 128 out of 256 patches). The LP tokens must encode enough information to reconstruct the missing patches, forcing semantic understanding.

### 2. Pixel-Level Modulation Features

Instead of global pooling:
- LAINRDecoder extracts modulation features at **every pixel** (32×32 grid)
- These features come from cross-attention between queries and LP tokens
- Preserves spatial structure for CNN classification

### 3. Two-Stage Training

**Stage 1: Masked Reconstruction Pretraining (100 epochs)**
- Input: Images with 50% masked patches
- Task: Reconstruct all pixels from visible patches
- Loss: MSE on masked pixels only
- Output: Pretrained encoder with semantic features

**Stage 2: Classification with Frozen Features (100 epochs)**
- Extract: 32×32×512 modulation features from pretrained model
- Train: CNN classifier on spatial features
- Test: Linear probe accuracy on frozen features

## File Structure

```
masked_mamba_ginr_v2.py       # Model architecture
train_masked_mamba_ginr_v2.py # Training pipeline
decoder_fix.py                # LAINRDecoder (already exists)
```

## Usage

### Training

```bash
# Full two-stage training
python train_masked_mamba_ginr_v2.py
```

This will:
1. Train masked reconstruction for 100 epochs
2. Save best model to `./checkpoints/masked_mamba_ginr_pretrain_best.pth`
3. Extract modulation features from frozen encoder
4. Train CNN classifier for 100 epochs
5. Save results and visualizations

### Model Components

```python
from masked_mamba_ginr_v2 import MaskedMAMBAGINR, CNNClassifier

# Initialize model
model = MaskedMAMBAGINR(
    img_size=32,
    patch_size=2,
    dim=256,
    num_lp=256,
    mamba_depth=6,
    hidden_dim=512,
    mask_ratio=0.5  # 50% masking
)

# Stage 1: Training mode (reconstruction)
loss, pred_rgb, mask = model(images, return_modulation=False)

# Stage 2: Inference mode (feature extraction)
modulation = model(images, return_modulation=True)  # (B, 32, 32, 512)

# CNN classifier
classifier = CNNClassifier(hidden_dim=512, num_classes=10)
logits = classifier(modulation)
```

## Expected Results

### Stage 1: Reconstruction
- **Metric**: MSE loss on masked pixels
- **Expected**: ~0.001 - 0.01 (depending on reconstruction quality)
- **Visualization**: Saved every 10 epochs showing original → masked → reconstructed

### Stage 2: Classification
- **Metric**: Top-1 accuracy on CIFAR-10
- **Expected**: **60-75%** (vs. 45% with previous broken version)
- **Comparison**:
  - Supervised baseline: ~90% (full supervision)
  - Random features: ~10% (random chance)
  - Previous version: ~45% (information leakage)
  - This version: **60-75%** (proper self-supervised learning)

## Why Pixel-Level Modulation Features?

### Previous Approach (Global Pooling)
```python
lp_tokens = model.encoder(patches, mask)  # (B, 256, 256)
features = lp_tokens.mean(dim=1)          # (B, 256) ← loses spatial info
classifier = Linear(256, 10)              # Simple linear layer
```

**Problems**:
- Loses all spatial structure
- LP tokens encode different parts of image → averaging loses localization
- Linear classifier can't capture spatial relationships

### This Approach (Spatial Modulation)
```python
lp_features = model.encoder(patches, mask)           # (B, 256, 256)
coords = create_grid(32, 32)                         # (32, 32, 2)
modulation = decoder.extract_modulation(coords, lp)  # (B, 32, 32, 512)
logits = CNNClassifier(modulation)                   # CNN on spatial features
```

**Advantages**:
- Preserves spatial structure (32×32 feature map)
- Each pixel gets features from cross-attention with LP tokens
- CNN can learn spatial patterns and local/global features
- Similar to how Transformer vision models work (but with implicit neural representation decoder)

## Configuration

Edit `CONFIG` dictionary in `train_masked_mamba_ginr_v2.py`:

```python
CONFIG = {
    'mask_ratio': 0.5,        # 50% masking (0.5-0.7 recommended for CIFAR-10)
    'stage1_epochs': 100,     # Reconstruction pretraining epochs
    'stage2_epochs': 100,     # Classification epochs
    'stage1_lr': 5e-4,        # Learning rate for pretraining
    'stage2_lr': 1e-3,        # Learning rate for classifier
    'hidden_dim': 512,        # Modulation feature dimension
    'num_lp': 256,            # Number of LP tokens
    # ... see file for full config
}
```

## Comparison Summary

| Aspect | Previous Version | This Version |
|--------|-----------------|--------------|
| Encoder | Simplified Transformer | BiMamba (original) |
| Decoder | Simplified transformer decoder | LAINRDecoder (original) |
| Features | Global avg pool (256D) | Pixel-level modulation (32×32×512) |
| Classifier | Linear (1 layer) | CNN (6 conv layers) |
| Masking | 75% (too aggressive) | 50% (reasonable) |
| Expected Acc | ~45% | **60-75%** |

## Why This Should Work Better

1. **Semantic Features**: BiMamba + masked reconstruction forces semantic understanding
2. **Spatial Preservation**: Pixel-level modulation preserves object localization
3. **Rich Features**: 512-dim features at each pixel (vs 256-dim global vector)
4. **Powerful Classifier**: CNN can learn hierarchical spatial patterns
5. **Proper Architecture**: Uses proven MAMBA-GINR components, not simplified versions

## Troubleshooting

### If reconstruction quality is poor:
- Reduce `mask_ratio` from 0.5 to 0.4
- Increase `stage1_epochs` from 100 to 200
- Increase `hidden_dim` in decoder from 512 to 768

### If classification accuracy is low:
- Check that pretrained reconstruction works well (visually inspect saved images)
- Ensure modulation features have spatial structure (print shape: should be (B, 32, 32, 512))
- Try deeper CNN classifier
- Extract features from multiple decoder layers

## Notes

- This implementation requires `mamba-ssm` package for BiMamba
- Requires `decoder_fix.py` for LAINRDecoder
- GPU recommended (model has ~30M parameters)
- Training time: ~2-3 hours per stage on single GPU
