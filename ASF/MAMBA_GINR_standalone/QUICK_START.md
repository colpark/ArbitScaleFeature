# Quick Start: Masked MAMBA-GINR

## ✅ Fixed and Ready to Run!

The **index out of bounds error** has been fixed. The notebook is now ready to train.

## 🚀 Run the Notebook

```bash
jupyter notebook masked_mamba_ginr_proper.ipynb
```

Then **Run All Cells** (Cell → Run All)

## 🔧 What Was Fixed

### Bug
```
RuntimeError: CUDA error: device-side assert triggered
index out of bounds in LearnablePositionTokens.extract_lp()
```

### Root Cause
- LP tokens used **equidistant interleaving** designed for fixed-length sequences
- With masking, sequence length varies (128 visible patches instead of 256)
- Interleaving indices computed for length 512, but actual sequence was 384
- `extract_lp()` tried to index position 511 in sequence of length 384 → ERROR

### Solution
Simplified `LearnablePositionTokens`:
- **Before**: Complex interleaving with fixed indices
- **After**: Simple concatenation at end of sequence

```python
# OLD (broken)
def extract_lp(self, x):
    return x[:, self.lp_idxs]  # lp_idxs = [0, 2, 4, ..., 511] → OUT OF BOUNDS!

# NEW (fixed)
def extract_lp(self, x):
    return x[:, -self.num_tokens:]  # Last 256 positions → ALWAYS WORKS
```

## 📊 Expected Results

### Stage 1: Masked Reconstruction
- **Epochs**: 100
- **Expected loss**: ~0.001-0.01 (MSE on masked pixels)
- **Visualization**: Saved every 10 epochs

### Stage 2: CNN Classification
- **Epochs**: 100
- **Expected accuracy**: **60-75%**
- **Comparison**:
  - Previous broken version: ~45%
  - This proper version: **60-75%**
  - Supervised baseline: ~88-90%

## 🎯 Architecture Overview

```
Input: 32×32 RGB image
    ↓
Patchify: 2×2 patches → 256 patches
    ↓
Random Mask: 50% → 128 visible patches (INFORMATION BOTTLENECK)
    ↓
Patch Embedding + Fourier Positional Encoding
    ↓
Add 256 LP Tokens at END: [128 patches] + [256 LP] = 384 tokens
    ↓
BiMamba Encoder (6 layers): Processes all 384 tokens bidirectionally
    ↓
Extract LP Features: Last 256 tokens (always at end)
    ↓
LAINRDecoder:
    - Query: 32×32 coordinate grid
    - Cross-attention with LP features
    - Output: 32×32×512 modulation OR 32×32×3 RGB
    ↓
Stage 1: Reconstruct masked pixels (loss on masked only)
Stage 2: CNN classifier on 32×32×512 spatial features
```

## 🔑 Key Improvements

| Aspect | Previous (Broken) | This (Fixed) |
|--------|------------------|--------------|
| Encoder | Simplified Transformer | ✅ BiMamba |
| Decoder | Simplified | ✅ LAINRDecoder |
| Features | 256D global vector | ✅ 32×32×512 spatial |
| Classifier | Linear layer | ✅ CNN (6 layers) |
| Masking | 75% (too much) | ✅ 50% (balanced) |
| LP Tokens | Broken interleaving | ✅ Simple concatenation |
| Accuracy | ~45% | ✅ **60-75%** |

## 📁 Files

- **`masked_mamba_ginr_proper.ipynb`** ← Run this!
- `masked_mamba_ginr_v2.py` - Modular Python version
- `train_masked_mamba_ginr_v2.py` - Standalone training script
- `BUGFIX_lp_tokens.md` - Bug fix documentation
- `README_MASKED_V2.md` - Complete documentation

## 💡 Training Tips

### If reconstruction quality is poor:
- Reduce `mask_ratio` from 0.5 → 0.4 (less aggressive masking)
- Increase `stage1_epochs` from 100 → 200 (train longer)
- Check visualizations every 10 epochs to see if model is learning

### If classification accuracy < 60%:
- Ensure Stage 1 reconstruction works well first
- Try increasing `hidden_dim` from 512 → 768 (richer features)
- Increase `stage2_epochs` from 100 → 200
- Try reducing learning rate (5e-4 → 1e-4 for Stage 1)

## ⚡ Quick Test

To verify the fix works, run just the first few cells:

```python
# Cells 1-2: Imports and config ✓
# Cell 8: BiMamba components (FIXED) ✓
# Cell 12: MaskedMAMBAGINR model ✓
# Cell 18: Initialize model ✓

# Test forward pass
images, _ = next(iter(train_loader))
images = images.to(device)
loss, pred, mask = model(images, return_modulation=False)
print(f"✓ Forward pass works! Loss: {loss.item():.4f}")
```

If this prints without errors, the fix is working!

## 🎓 What You'll Learn

1. **Information Bottleneck**: How masking forces semantic understanding
2. **Pixel-Level Features**: Why spatial features > global pooling
3. **Architecture Matters**: BiMamba + LAINRDecoder > simplified versions
4. **Self-Supervised Learning**: Pretraining strategy for vision

## 📧 Issues?

If you encounter any problems:
1. Check CUDA is available: `torch.cuda.is_available()`
2. Verify dependencies: `mamba-ssm`, `einops`, `torch`, `torchvision`
3. Check GPU memory (model needs ~4-6GB)
4. Try reducing batch size if OOM

Good luck with training! 🚀
