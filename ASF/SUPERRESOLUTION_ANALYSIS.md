# Super-Resolution Analysis: How Queries Are Formed in LAINR

## The Issue You Identified

You're absolutely correct - the current implementation is **resampling, not true super-resolution**. Let me show you exactly why:

## Detailed Query Formation Process

### Step-by-Step Breakdown:

```python
def forward(self, x, tokens):
    """
    x: (B, H, W, 2) - coordinate grid (can be ANY resolution)
    tokens: (B, L, D) - LP features encoded from 32×32 image
    """
    B, query_shape = x.shape[0], x.shape[1: -1]  # query_shape = (H, W)
    x = x.view(B, -1, x.shape[-1])  # (B, HW, 2)

    # === QUERY FORMATION (Line 103-105) ===
    # 1. Fourier encoding with FIXED frequency sigma_q
    x_q = einops.repeat(
        self.calc_gamma(x[0], self.omegas), 'l d -> b l d', b=B
    )
    # x_q shape: (B, HW, feature_dim)

    # 2. Project to hidden dimension
    x_q = self.act(self.query_lin(x_q))
    # x_q shape: (B, HW, hidden_dim)

    # === MODULATION EXTRACTION (Line 115) ===
    # 3. Cross-attention: queries attend to LP tokens
    modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)
    # modulation_vector shape: (B, HW, hidden_dim)

    # === MULTI-SCALE DECODING (Line 121-126) ===
    for k in range(self.layer_num):
        # 4. Bandwidth encoding at each frequency scale
        x_l = einops.repeat(
            self.calc_gamma(x[0], self.omegas_l[k]), 'l d -> b l d', b=B
        )
        h_l = self.act(self.bandwidth_lins[k](x_l))

        # 5. Add modulation (THIS IS WHERE FEATURES ARE INJECTED)
        m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
        modulations_l.append(m_l)
```

## The Problem: Resampling vs Super-Resolution

### What's Happening Now (RESAMPLING):

1. **Query encoding** (`x_q`):
   - Encodes coordinate positions using Fourier features
   - Frequency: `sigma_q = 16` (FIXED)
   - For 128×128 grid: encodes 16,384 positions with SAME frequency basis

2. **Bandwidth encoding** (`x_l`):
   - Two scales: `sigma_ls = [128, 32]`
   - These are the MAXIMUM frequencies the network can represent
   - **Critical**: These are the SAME frequencies used during training!

3. **Modulation**:
   - Extracts features from LP tokens via cross-attention
   - BUT: LP tokens were encoded from 32×32 patches
   - They contain NO information beyond the Nyquist limit of 32×32

### Why This Is Just Resampling:

```
Training (32×32):
  - Coordinates: 1024 points in [0,1]²
  - Fourier encoding: sin/cos(π * coord * ω), ω ∈ [10, 128]
  - LP tokens learn to reconstruct THESE frequencies

Super-Resolution (128×128):
  - Coordinates: 16,384 points in [0,1]²
  - Fourier encoding: sin/cos(π * coord * ω), ω ∈ [10, 128] ← SAME!
  - LP tokens interpolate between known points
  - NO NEW HIGH FREQUENCIES!
```

## Mathematical Analysis

### Fourier Feature Encoding:

```python
def calc_gamma(self, x, omegas):
    # x: (HW, 2) - coordinates in [0, 1]²
    # omegas: (F,) - frequencies, e.g., logspace(10, 128, n)

    arg = π * x * ω  # shape: (HW, 2, F)
    γ = [sin(arg), cos(arg)]  # shape: (HW, 2*2*F)

    return γ
```

### Frequency Content:

For `sigma_ls = [128, 32]`:

**Scale 1 (coarse)**: ω ∈ [10, 128]
- Maximum spatial frequency: **128 cycles per unit**
- At 32×32: Nyquist = 16 cycles → ω_max = 16π ≈ 50
- **We're already OVER-specifying** (128 > 50)

**Scale 2 (fine)**: ω ∈ [10, 32]
- Maximum spatial frequency: **32 cycles per unit**
- Perfectly captures 32×32 content

**At 128×128 super-resolution**:
- Nyquist = 64 cycles → ω_max = 64π ≈ 200
- But we're still using ω_max = 128
- **Missing frequencies: [128, 200]** ← THIS IS THE GAP!

## Why LP Tokens Can't Generate New Frequencies

### Information-Theoretic Limit:

1. **Training**: LP tokens see 32×32 images
   - Maximum frequency content: ≈50 cycles/unit
   - LP tokens learn to encode THIS frequency band

2. **Testing**: Query at 128×128
   - Requesting frequency content: ≈200 cycles/unit
   - LP tokens have NO information about [50, 200]
   - Result: Smooth interpolation (band-limited upsampling)

### What Modulation Actually Does:

```python
# At position (x, y), the modulation is:
modulation(x,y) = CrossAttention(
    query = FourierEncode(x,y, σ=16),
    keys/values = LP_tokens
)

# This extracts:
# - Which image content is at position (x,y)
# - But ONLY frequencies seen during training!
```

## The Core Architecture Issue

### Current Flow:

```
32×32 image
  ↓ [Patch embed]
256 patches
  ↓ [Mamba encoder]
256 LP tokens ← Contains band-limited info (σ ≤ 128)
  ↓ [Cross-attention with queries at ANY resolution]
modulation_vector(x,y) ← Interpolates existing frequencies
  ↓ [Fourier decode with σ ∈ [10, 128]]
RGB(x,y) ← Same frequency band as training!
```

### What's Missing for True Super-Resolution:

1. **High-frequency priors**: Need to hallucinate details beyond training resolution
2. **Texture synthesis**: Generate plausible high-frequency patterns
3. **Learned upsampling**: Network should learn to add details, not just interpolate

## Solutions for True Super-Resolution

### Option 1: Scale-Adaptive Frequency Encoding

Increase `sigma_ls` based on target resolution:

```python
def forward(self, x, tokens):
    target_resolution = x.shape[1]  # H of target grid

    # Adapt frequencies to target resolution
    scale_factor = target_resolution / 32  # e.g., 128/32 = 4
    sigma_ls_adapted = [s * scale_factor for s in self.sigma_ls]
    # For 128×128: sigma_ls = [512, 128] instead of [128, 32]

    omegas_l_adapted = [
        torch.logspace(1, math.log10(s), self.n)
        for s in sigma_ls_adapted
    ]

    # Use adapted frequencies for bandwidth encoding
    for k in range(self.layer_num):
        x_l = self.calc_gamma(x[0], omegas_l_adapted[k])
        ...
```

**Problem**: LP tokens still don't contain high-freq info!

### Option 2: Multi-Scale Encoder (Better!)

Encode image at MULTIPLE resolutions:

```python
class MultiScaleEncoder(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        # scales = [32×32, 64×64, 128×128]
        self.encoders = nn.ModuleList([
            MambaEncoder(...) for _ in scales
        ])

    def encode(self, image):
        lp_features_list = []

        for scale, encoder in zip(self.scales, self.encoders):
            # Upsample image
            img_scaled = F.interpolate(image, scale_factor=scale)

            # Encode at this resolution
            lp_scaled = encoder(img_scaled)
            lp_features_list.append(lp_scaled)

        # Concatenate multi-scale features
        return torch.cat(lp_features_list, dim=1)
```

**Benefit**: LP tokens now contain multi-scale information!

### Option 3: Implicit Neural Super-Resolution Layer

Add a learned high-frequency generator:

```python
class HighFreqGenerator(nn.Module):
    """Generate high-frequency residual"""
    def __init__(self, hidden_dim):
        self.hf_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for coords
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # RGB residual
        )

    def forward(self, modulation, coords):
        # Concatenate modulation with high-res coordinates
        features = torch.cat([modulation, coords], dim=-1)
        hf_residual = self.hf_net(features)
        return hf_residual

# In LAINRDecoder.forward():
out_base = sum(outs)  # Base reconstruction

# Add high-frequency details
hf_residual = self.hf_generator(modulation_vector, x)
out = out_base + hf_residual
```

## Current Notebook's Super-Resolution

The current implementation is effectively doing:

```python
def super_resolve(model, images, target_size=128):
    # Encode at 32×32
    lp_features = model.encode(images)  # Contains frequencies up to σ=128

    # Decode at 128×128
    hr_coords = create_coordinate_grid(128, 128)
    hr_pixels = model.decode(lp_features, hr_coords)

    # Result: Band-limited upsampling
    # Equivalent to: bicubic interpolation + learned filtering
```

## Verification Experiment

To confirm this is resampling, check frequency spectrum:

```python
import numpy as np
from scipy.fft import fft2, fftshift

# Original 32×32
img_32 = test_images[0].cpu().numpy()
spectrum_32 = np.abs(fftshift(fft2(img_32[0])))

# Super-resolved 128×128
img_128 = sr_128[0].cpu().numpy()
spectrum_128 = np.abs(fftshift(fft2(img_128[0])))

# Plot frequency spectrum
fig, axes = plt.subplots(1, 2)
axes[0].imshow(np.log(spectrum_32 + 1))
axes[0].set_title('32×32 Frequency')
axes[1].imshow(np.log(spectrum_128 + 1))
axes[1].set_title('128×128 Frequency')

# If super-res is just resampling:
# - Center will be bright (low freq)
# - Edges will be DARK (missing high freq)
```

## Summary

**Current behavior**:
- ✅ Arbitrary-scale *resampling*
- ❌ NOT true super-resolution

**Root cause**:
1. LP tokens encode band-limited info (σ_max = 128)
2. Fourier decoder uses SAME frequencies at all resolutions
3. No mechanism to generate frequencies beyond training resolution

**To achieve true super-resolution**, you need:
1. Multi-scale encoding OR
2. High-frequency hallucination network OR
3. Adaptive frequency encoding + learned priors

The architecture is fundamentally limited by the **information bottleneck** at the LP tokens!
