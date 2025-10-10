# MAMBA-GINR Encoder + SCENT Processor/Decoder Architecture

## Your Goal:

Create a hybrid model that combines:
1. **MAMBA-GINR encoder** (with all MAMBA-GINR innovations)
2. **SCENT processor + decoder** (pure SCENT after encoding)

## Architecture Breakdown:

### Part 1: MAMBA-GINR Encoder (Kept from MAMBA-GINR)
```
Input Image (32×32)
    ↓
[Patch Encoder] (2×2 patches → 256 patch tokens)
    ↓
[Concatenate with LP Tokens] (256 learnable position tokens)
    ↓ (256 patch + 256 LP = 512 tokens)
[BiMamba] (forward + backward Mamba processing)
    ↓
[Extract LP Tokens] (last 256 tokens)
    ↓
Encoded Latents (B, 256, 256)
```

**Key MAMBA-GINR innovations preserved:**
- ✅ BiMamba (bidirectional Mamba)
- ✅ Learnable Position (LP) tokens
- ✅ Patch + LP concatenation
- ✅ O(L) complexity from Mamba

### Part 2: SCENT Processor (Pure SCENT)
```
Encoded Latents (B, 256, 256)
    ↓
[Self-Attention Blocks] × 4
    Each block:
    - Self-Attention (256 latents, global mixing)
    - FeedForward (256 → 1024 → 256, 4x expansion)
    - Residual connections
    ↓
Processed Latents (B, 256, 256)
```

**From SCENT:**
- ✅ 4 layers of self-attention on latents
- ✅ Global receptive field
- ✅ FeedForward with 4x expansion
- ✅ GEGLU gating

### Part 3: SCENT Decoder (Pure SCENT)
```
Processed Latents (B, 256, 256)
+ Query Coordinates (B, H×W, 2)
    ↓
[Gaussian Fourier Encoding] (queries)
    ↓
[Cross-Attention] (queries attend to latents)
    ↓
[Decoder Self-Attention] × 3  ← Optional if memory allows
[OR Stacked FeedForwards] × 3  ← Memory efficient
    ↓
[Skip Connection] (add Fourier encoding)
    ↓
[Output Projection] → RGB
    ↓
Output (B, H, W, 3)
```

**From SCENT:**
- ✅ Gaussian Fourier features for queries
- ✅ Cross-attention to extract modulation
- ✅ Decoder processing (attention or FF)
- ✅ Skip connections from Fourier encoding

---

## Comparison: What Changes vs Pure Architectures?

### vs Pure MAMBA-GINR:
| Component | MAMBA-GINR | This Hybrid |
|-----------|------------|-------------|
| Encoder | BiMamba + LP tokens | ✅ Same |
| Processor | None (direct to decoder) | ➕ SCENT self-attention (4 layers) |
| Decoder | LAINR (weak ResidualBlocks) | ➕ SCENT (strong FF/Attention) |

### vs Pure SCENT:
| Component | SCENT | This Hybrid |
|-----------|-------|-------------|
| Encoder | Cascaded Perceiver (hierarchical) | ➕ BiMamba + LP tokens |
| Processor | Self-attention (4 layers) | ✅ Same |
| Decoder | Cross-attn + Local refinement | ✅ Same |

---

## Why This Makes Sense:

### 1. **BiMamba Encoder Advantages:**
- ✅ **O(L) complexity** vs O(L²) in SCENT's Perceiver
- ✅ **Sequential inductive bias** from Mamba SSM
- ✅ **LP tokens** for implicit sequential structure
- ✅ **Proven in MAMBA-GINR** to work well

### 2. **SCENT Processor Advantages:**
- ✅ **Global receptive field** (self-attention on 256 latents)
- ✅ **High capacity** (4 layers, 4x FF expansion)
- ✅ **Better feature mixing** than single Mamba layer
- ✅ **Memory efficient** (256 tokens, not 1024 queries)

### 3. **SCENT Decoder Advantages:**
- ✅ **4x expansion** in FeedForward (512 → 2048 → 512)
- ✅ **GEGLU gating** (more expressive than ReLU)
- ✅ **Skip connections** preserve high-freq Fourier
- ✅ **Proven in SCENT** for super-resolution

---

## Expected Benefits:

### 1. **Best of Both Worlds:**
- Mamba's efficiency + SCENT's capacity
- Sequential bias + global mixing
- O(L) encoding + strong decoding

### 2. **Super-Resolution Performance:**
- Should match or exceed SCENT
- Better than MAMBA-GINR's weak decoder
- Expected: +5-10 dB PSNR at 128×128 vs MAMBA-GINR

### 3. **Computational Efficiency:**
- Encoder: O(L) from Mamba (vs O(L²) Perceiver)
- Processor: O(L²) but only 256 tokens (manageable)
- Decoder: O(N) with FeedForward (vs O(N²) if using attention)

---

## Memory Analysis:

### Encoder (BiMamba):
- Patch features: (B, 256, 256)
- LP tokens: (256, 256) params
- Mamba states: O(L) memory
- **Total: ~0.5GB**

### Processor (SCENT self-attention on 256 latents):
- 4 layers × attention matrices
- Each: (B*8, 256, 256) = 64 × 256 × 256 × 4 bytes = 16MB
- **Total: ~0.1GB** ✓ Very manageable

### Decoder (FeedForward, no self-attention):
- Cross-attention: (B*2, 1024, 256) = 0.5GB
- FeedForwards: ~0.5GB
- **Total: ~1GB**

### Grand Total: ~2GB for forward pass ✓
- Plus gradients: ~4GB
- Plus optimizer states: ~6GB
- **Fits easily in 40GB GPU!**

---

## Implementation Plan:

### Step 1: Encoder (from MAMBA-GINR)
```python
class MambaEncoder(nn.Module):
    def __init__(self, dim=256, num_lp_tokens=256):
        self.patch_encoder = PatchEncoder(patch_size=2, dim=dim)
        self.lp_tokens = LearnablePositionTokens(num_lp_tokens, dim)
        self.mamba = BiMamba(d_model=dim)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_encoder(x)
        lp_tokens = self.lp_tokens(B)
        combined = torch.cat([patches, lp_tokens], dim=1)
        features = self.mamba(combined)
        return features[:, -lp_tokens.shape[1]:, :]  # Extract LP tokens
```

### Step 2: Processor (from SCENT)
```python
class SCENTProcessor(nn.Module):
    def __init__(self, dim=256, num_layers=4):
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=8, dim_head=64)),
                PreNorm(dim, FeedForward(dim, mult=4))
            ])
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
```

### Step 3: Decoder (from SCENT, memory-efficient)
```python
class SCENTDecoder(nn.Module):
    def __init__(self, ...):
        # Gaussian Fourier encoding
        # Cross-attention for modulation
        # Stacked FeedForwards (no self-attention on queries)
        # Skip connections
        # Output projection

    def forward(self, coords, latents):
        # Encode queries with Gaussian Fourier
        # Extract modulation via cross-attention
        # Process through FeedForwards
        # Add skip connection
        # Output RGB
```

### Step 4: Complete Model
```python
class MambaGINR_SCENT_Hybrid(nn.Module):
    def __init__(self):
        self.encoder = MambaEncoder()
        self.processor = SCENTProcessor()
        self.decoder = SCENTDecoder()

    def forward(self, x, coords):
        latents = self.encoder(x)
        latents = self.processor(latents)
        rgb = self.decoder(coords, latents)
        return rgb
```

---

## Next Steps:

1. ✅ Create the hybrid implementation
2. ✅ Train on CIFAR-10 (32×32)
3. ✅ Test super-resolution (128×128, 256×256)
4. ✅ Compare with:
   - Pure MAMBA-GINR (weak decoder)
   - Pure SCENT (slower encoder)
   - This hybrid (best of both?)

---

## Expected Result:

**Hypothesis:** This hybrid should outperform both pure models because:
- Better encoder than SCENT (O(L) vs O(L²))
- Better processor than MAMBA-GINR (4 layers vs 0)
- Better decoder than MAMBA-GINR (SCENT-style vs ResidualBlock)
- More efficient than pure SCENT (Mamba encoder)

**Expected super-resolution PSNR at 128×128:**
- MAMBA-GINR (original): ~22 dB (smooth, no high-freq)
- SCENT (pure): ~32 dB (good but slow encoder)
- This hybrid: ~32-34 dB (best of both!)
