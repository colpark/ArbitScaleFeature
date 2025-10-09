# Super-Resolution Failure Analysis

## Your Observation

✅ **Correct observation**: Higher resolution images lack high-frequency details - just smooth upsampling.

## Potential Reasons (Ranked by Likelihood)

### 1. ⭐⭐⭐ DECODER IS TOO SMALL (Your Hypothesis - VERY LIKELY!)

**Current decoder capacity**:
```python
hidden_dim = 256
layers_per_scale = 1  # Only ONE linear layer per frequency scale!

# Total decoder parameters:
# - Query encoding: 64 → 256 (16K params)
# - Cross-attention: ~200K params
# - Bandwidth encoding (×2 scales): 64 → 256 each (32K params)
# - Modulation projection (×2): 256 → 256 each (128K params)
# - Hidden value (×1): 256 → 256 (64K params)
# - Output (×2): 256 → 3 each (1.5K params)
# Total: ~450K params for entire decoder
```

**Why this is insufficient for super-resolution**:

The decoder must learn to:
1. **Extract semantic features** from modulation (edges, textures, patterns)
2. **Combine** semantics with high-frequency Fourier basis
3. **Generate realistic high-frequency patterns** (texture synthesis)

**With only 256 hidden dims and 1 layer per scale**, the decoder has:
- ❌ Insufficient capacity to learn complex texture generators
- ❌ Limited ability to combine semantics + high-freq effectively
- ❌ Can only do smooth interpolation (linear combinations)

**Evidence from other papers**:
- **SIREN** (super-resolution INR): Uses 5+ layers of 256 dims = ~1M params
- **NeRF** (3D synthesis): Uses 8 layers of 256 dims = ~2M params
- **Your decoder**: 2 scales × 1 layer × 256 dims = ~450K params ⚠️

### Solution 1: Increase Decoder Capacity

```python
# CURRENT (insufficient):
hidden_dim = 256
layers_per_scale = 1

# RECOMMENDED:
hidden_dim = 512  # Double the width
num_layers = 3    # Add depth (residual blocks)

# Or even more aggressive:
hidden_dim = 512
num_layers = 4
```

**Expected improvement**: +3-5 dB PSNR at 128×128

---

### 2. ⭐⭐⭐ TRAINING ON LOW RESOLUTION ONLY

**Current training**:
```python
# Train ONLY on 32×32
coord = create_coordinate_grid(32, 32, device)
```

**The problem**:
- Network sees only **32×32 images** (Nyquist ≈ 16 cycles/image)
- LP tokens encode only **low-frequency information**
- Even with σ=128 Fourier features, the **modulation vectors** are band-limited

**Why this matters**:
```
Training at 32×32:
  LP tokens ← Extract features from 32×32 patches
            ← Maximum frequency ≈ 16 cycles/image

  At 128×128 test:
    Query: High-freq Fourier basis (σ=128) ✓
    Modulation: Band-limited to ~16 cycles ✗
    Output: modulation × high-freq = BAND-LIMITED ✗
```

**Even with large decoder**, if modulation is band-limited, output will be smooth!

### Solution 2A: Multi-Resolution Training (STRONG RECOMMENDATION)

```python
def train_epoch_multires(model, loader, optimizer, device, epoch):
    for images, _ in loader:
        # Randomly sample resolution per batch
        res = random.choice([32, 48, 64])

        # Upsample image to target resolution
        images_upsampled = F.interpolate(images, size=res, mode='bicubic')

        # Train at this resolution
        coords = create_coordinate_grid(res, res, device)
        pred = model(images_upsampled, coords)
        ...
```

**Why this works**:
- Network sees **64×64 images** (Nyquist ≈ 32 cycles)
- LP tokens learn to encode **higher frequencies**
- Modulation vectors become **less band-limited**

**Expected improvement**: +5-8 dB at 128×128 🔥

### Solution 2B: Patch-Based High-Res Training

```python
def train_epoch_patches(model, loader, optimizer, device):
    for images, _ in loader:
        # Extract random 16×16 patches from 32×32 images
        # But query them at HIGHER internal resolution

        # Upsample to 64×64
        images_64 = F.interpolate(images, size=64, mode='bicubic')

        # Random crop 32×32 patches
        start_h = random.randint(0, 32)
        start_w = random.randint(0, 32)
        patch = images_64[:, :, start_h:start_h+32, start_w:start_w+32]

        # Train on high-res patch
        coords = create_coordinate_grid(32, 32, device)
        pred = model(patch, coords)
        ...
```

---

### 3. ⭐⭐ INSUFFICIENT JITTERING VARIANCE

**Current jittering**:
```python
jitter_std = 1/(6*32) ≈ 0.00521  # Very small!
```

**The problem**:
- Training coordinates are **almost identical** to pixel centers
- Network learns **discrete lookup table**, not continuous function
- No incentive to generate sub-pixel details

**Why it matters for SR**:
```
Without enough jittering:
  Training coords: [0.0, 0.03125, 0.0625, ...]  # Grid-aligned
  Network learns: "At x=0.03125, output rgb_memorized"

  At 128×128 test:
    Query at x=0.0039 (between pixels)
    Network: "Interpolate between x=0.0 and x=0.03125"
    Result: SMOOTH interpolation
```

### Solution 3: Increase Jittering

```python
# CURRENT:
jitter_std = 1/(6*32) ≈ 0.00521

# RECOMMENDED (still sub-pixel):
jitter_std = 1/(4*32) ≈ 0.0078  # Larger variance
# This ensures network sees more diverse positions
```

**OR** add explicit sub-pixel training:

```python
def train_epoch_subpixel(model, loader, optimizer, device):
    for images, _ in loader:
        # Create grid with random sub-pixel offsets
        base_coords = create_coordinate_grid(32, 32, device)

        # Add larger jitter (but interpolate ground truth)
        jitter = torch.randn_like(base_coords) * 0.015  # Half pixel
        coords_jittered = (base_coords + jitter).clamp(0, 1)

        # Interpolate ground truth at jittered positions
        gt_interpolated = F.grid_sample(
            images,
            coords_jittered.unsqueeze(0) * 2 - 1,  # Normalize to [-1,1]
            align_corners=True
        )

        pred = model(images, coords_jittered)
        loss = mse(pred, gt_interpolated)
```

---

### 4. ⭐⭐ NO ADVERSARIAL/PERCEPTUAL LOSS

**Current training**:
```python
loss = MSE(pred, gt)  # Pixel-wise L2 loss only
```

**The problem**:
- MSE loss encourages **smooth solutions** (minimizes variance)
- No penalty for blurry outputs
- No incentive to generate sharp edges/textures

**Why it fails for SR**:
```
MSE optimal solution:
  pred = E[gt | coords]  (expected value)

For super-resolution:
  E[high_res | low_res] = BLURRED high_res

MSE will prefer: Smooth blur (low MSE)
NOT: Sharp details (higher MSE but perceptually better)
```

### Solution 4: Add Perceptual/Adversarial Loss

```python
# Option A: Perceptual loss (VGG features)
import torchvision.models as models

vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)

def perceptual_loss(pred, gt):
    pred_feat = vgg(pred)
    gt_feat = vgg(gt)
    return F.mse_loss(pred_feat, gt_feat)

# Combined loss:
loss = mse_loss + 0.1 * perceptual_loss(pred, gt)
```

```python
# Option B: Adversarial loss (GAN)
discriminator = Discriminator().to(device)

# Generator loss:
loss_gen = mse_loss + 0.01 * adversarial_loss(discriminator(pred))

# Discriminator loss:
loss_disc = bce_loss(discriminator(gt), real) + bce_loss(discriminator(pred.detach()), fake)
```

**Expected improvement**: +2-4 dB + much sharper visuals

---

### 5. ⭐ GAUSSIAN FOURIER FEATURES NOT DENSE ENOUGH

**Current implementation**:
```python
n_features = feature_dim // 2 = 32
B_matrix = torch.randn(32, 2) / sigma
```

**Only 32 random frequencies!**

**The problem**:
- 32 frequencies may not densely cover the frequency space
- Gaps in frequency coverage → poor interpolation
- Especially problematic for high frequencies (σ=128)

### Solution 5: Increase Number of Fourier Features

```python
# CURRENT:
feature_dim = 64  # → 32 random frequencies

# RECOMMENDED:
feature_dim = 128  # → 64 random frequencies
# Or even:
feature_dim = 256  # → 128 random frequencies
```

**Trade-off**: More features = more parameters, but better frequency coverage

---

### 6. ⭐ LEARNABLE FREQUENCIES NOT ACTUALLY LEARNING

**Current setup**:
```python
learnable_frequencies = True
```

**Potential issues**:
1. **Learning rate too low** for frequency parameters
2. **Frequencies collapsing** to same values
3. **Gradients vanishing** through Fourier encoding

### Solution 6: Diagnose Frequency Learning

```python
# Check if frequencies are actually changing:
B_init = model.hyponet.B_ls[0].clone()

# After training:
B_final = model.hyponet.B_ls[0]

print(f"Frequency change: {torch.norm(B_final - B_init):.4f}")
print(f"Frequency std: {torch.norm(B_final, dim=1).std():.4f}")

# If change is tiny → frequencies not learning!
```

**Fix**: Use separate learning rate for frequencies:

```python
param_groups = [
    {'params': [p for n, p in model.named_parameters() if 'B_' not in n],
     'lr': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'B_' in n],
     'lr': 1e-3}  # Higher LR for frequencies
]
optimizer = torch.optim.AdamW(param_groups)
```

---

## Summary: Most Likely Causes (Ranked)

| Rank | Issue | Impact | Difficulty | Recommendation |
|------|-------|--------|------------|----------------|
| 1 | **Decoder too small** | ⭐⭐⭐⭐⭐ | Easy | Increase to 512 dims, 3-4 layers |
| 2 | **Training only at 32×32** | ⭐⭐⭐⭐⭐ | Medium | Multi-resolution training |
| 3 | **MSE loss only** | ⭐⭐⭐⭐ | Medium | Add perceptual loss |
| 4 | **Insufficient jittering** | ⭐⭐⭐ | Easy | Increase jitter_std to 0.01 |
| 5 | **Too few Fourier features** | ⭐⭐ | Easy | Increase feature_dim to 128 |
| 6 | **Frequencies not learning** | ⭐⭐ | Easy | Separate LR for frequencies |

---

## Recommended Action Plan

### Quick Wins (Try First):

1. **Increase decoder size**: `hidden_dim=512, num_layers=3`
2. **Increase jittering**: `jitter_std=0.01`
3. **Increase Fourier features**: `feature_dim=128`

**Expected improvement**: +3-5 dB at 128×128

### Major Improvements (More Work):

4. **Multi-resolution training**: Train at [32, 48, 64]
5. **Perceptual loss**: Add VGG feature loss

**Expected improvement**: +5-8 dB at 128×128 🔥

### Full Solution (Best Results):

Combine all of the above + adversarial loss

**Expected**: Near state-of-the-art super-resolution

---

## Diagnostic Experiments

### Experiment 1: Check Decoder Capacity

Train two models:
- Model A: `hidden_dim=256, layers=1` (current)
- Model B: `hidden_dim=512, layers=3` (larger)

Compare PSNR at 128×128. If Model B >> Model A → **decoder was bottleneck**

### Experiment 2: Check Training Resolution

Train two models:
- Model A: Train only at 32×32 (current)
- Model B: Train at [32, 48, 64] (multi-res)

If Model B >> Model A → **training resolution was bottleneck**

### Experiment 3: Check Modulation Quality

Extract and visualize modulation vectors:
```python
# At 32×32 and 128×128
modulation_32 = extract_modulation(coords_32)
modulation_128 = extract_modulation(coords_128)

# Check spatial variation
grad_32 = compute_gradient(modulation_32)
grad_128 = compute_gradient(modulation_128)

# If grad_128 ≈ 0 → modulation is constant → no high-freq info
```

---

## My Hypothesis (Most Likely Cause)

**Combination of #1 and #2**:

1. **Decoder too small** → Can't learn complex texture generators
2. **Training only 32×32** → Modulation vectors band-limited

Even with perfect Gaussian Fourier features, if:
- Decoder has low capacity → Can only do linear combinations
- Modulation is band-limited → No high-freq information to combine

Result: **Smooth upsampling, no new details**

**Recommended fix**: Increase decoder to 512 dims + add multi-res training
