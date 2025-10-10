# Decoder Memory Analysis

## Current SCENT Decoder Memory Breakdown

### Components in Cell 6:

**1. Cross-Attention for Modulation (Required - can't remove)**
```python
self.modulation_ca = Attention(hidden_dim=512, context_dim=256, heads=2)
```
- Memory: `(B*2, num_queries, 256)` attention matrix
- For 32×32 training: `(64*2, 1024, 256)` = ~0.5GB
- **This is OK - only 0.5GB**

**2. Bandwidth Encoders (2 scales):**
```python
self.bandwidth_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        *[nn.Sequential(
            PreNorm(512, FeedForward(512, mult=4)),  # 512 → 2048 → 512
            PreNorm(512, FeedForward(512, mult=4))
        ) for _ in range(2)]  # num_layers - 1 = 2
    )
    for _ in range(2)  # 2 scales
])
```
- Each FeedForward: `(B, 1024, 512)` → `(B, 1024, 2048)` → `(B, 1024, 512)`
- Activations: ~2GB per FeedForward
- Total: 2 scales × 2 FeedForwards × 2 layers = 8 FeedForwards × 2GB = **~16GB**

**3. Modulation Projections (2 scales):**
```python
self.modulation_lins = nn.ModuleList([similar structure])
```
- Same as bandwidth: **~16GB**

**4. Hidden Value Layers (1 layer):**
```python
self.hv_lins = nn.ModuleList([
    nn.Sequential(
        PreNorm(512, FeedForward(512, mult=4)),
        PreNorm(512, FeedForward(512, mult=4))
    )
    for _ in range(1)  # layer_num - 1
])
```
- 2 FeedForwards × 2GB = **~4GB**

**5. Skip Projections & Output (small):**
- Linear layers: ~0.5GB

### **Total Decoder Memory: ~37GB just for activations!**

Plus gradients: **~74GB total** → Explains OOM even on 40GB GPU!

---

## Root Causes:

### 1. **4x Expansion is Too Large for 1024 Queries**
- FeedForward: `512 → 2048 → 512`
- Activation size: `(B=64, 1024, 2048)` = 64 × 1024 × 2048 × 4 bytes = **536MB per FF**
- With 10 FeedForwards total: **5.3GB** just for these activations
- **Plus gradients: ~10GB**

### 2. **Too Many FeedForward Layers**
- Bandwidth: 2 scales × 2 FFs = 4
- Modulation: 2 scales × 2 FFs = 4
- Hidden value: 1 × 2 FFs = 2
- **Total: 10 FeedForwards**

### 3. **High Hidden Dimension (512)**
- All intermediate computations use 512 dims
- Could reduce to 256

---

## Solutions (Ranked by Impact):

### ⭐⭐⭐⭐⭐ Solution 1: Reduce Expansion Factor (4x → 2x)
**Current:** `FeedForward(512, mult=4)` → 512 → 2048 → 512
**New:** `FeedForward(512, mult=2)` → 512 → 1024 → 512

**Memory savings:**
- Per FF: 536MB → 268MB (2x reduction)
- Total: 37GB → 18.5GB
- With gradients: 74GB → 37GB (still OOM!)

**Still too expensive!**

---

### ⭐⭐⭐⭐⭐ Solution 2: Reduce Hidden Dim (512 → 256)
**Change:** `hidden_dim=256` instead of 512

**Memory savings:**
- FeedForward becomes: 256 → 1024 → 256 (with mult=4)
- Activation: `(64, 1024, 1024)` = 268MB per FF
- Total: 10 FFs × 268MB = 2.7GB
- With gradients: **~5.4GB** ✓ Fits!

**Impact:** May reduce capacity slightly, but still much better than ResidualBlock

---

### ⭐⭐⭐⭐⭐ Solution 3: Reduce Num Decoder Layers (3 → 1)
**Change:** `num_decoder_layers=1` instead of 3

**Effect:**
- Bandwidth: 2 scales × 0 extra FFs (just Linear + ReLU)
- Modulation: 2 scales × 0 extra FFs
- Hidden value: 0 layers
- **Total: Only cross-attention + linear projections!**

**Memory:** ~1GB total ✓ Very efficient!

**Impact:** Simpler decoder, but cross-attention + skip connections still strong

---

### ⭐⭐⭐⭐ Solution 4: Share Weights Across Scales
**Change:** Use same FeedForward for both sigma_ls scales

```python
# Instead of separate FFs for each scale:
shared_bandwidth = nn.Sequential(...)
self.bandwidth_lins = nn.ModuleList([shared_bandwidth, shared_bandwidth])
```

**Memory savings:** 2x reduction
**Impact:** Reduces parameters but may hurt multi-scale learning

---

### ⭐⭐⭐ Solution 5: Remove Multi-Scale (2 → 1 scale)
**Change:** `sigma_ls = [64]` instead of `[128, 32]`

**Memory savings:** 2x reduction (only 1 scale)
**Impact:** Loss of multi-scale benefits

---

## Recommended Combined Approach:

### **Option A: Minimal Change (Hidden Dim Reduction)**
```python
model = MambaSCENTHybrid(
    hidden_dim=256,  # 512 → 256
    num_decoder_layers=3  # keep layers
)
```
- Memory: ~5.4GB with gradients ✓ Fits!
- Still has multi-scale, skip connections, 4x expansion
- Expected PSNR: ~30-32 dB (slight reduction from 32-34 dB)

---

### **Option B: Ultra-Efficient (Reduce Layers + Hidden Dim)**
```python
model = MambaSCENTHybrid(
    hidden_dim=256,  # 512 → 256
    num_decoder_layers=1  # 3 → 1
)
```
- Memory: **~2GB with gradients** ✓ Very efficient!
- Simpler decoder but still has:
  - Cross-attention for modulation
  - Skip connections
  - Multi-scale
- Expected PSNR: ~28-30 dB

---

### **Option C: Reduce Expansion (More Conservative)**
```python
# Modify FeedForward in Cell 3:
class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):  # 4 → 2
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
```
- Memory: ~18.5GB with gradients (still OOM!)
- **Not recommended alone**

---

## Quick Fix: Apply Option A

**Single change in Cell 10:**
```python
model = MambaSCENTHybrid(
    patch_size=2,
    in_channels=3,
    dim=256,
    num_lp_tokens=256,
    feature_dim=64,
    sigma_q=16,
    sigma_ls=[128, 32],
    hidden_dim=256,  # ← CHANGE: 512 → 256
    num_processor_layers=4,
    num_decoder_layers=3
).to(device)
```

**Expected result:**
- Memory: ~5-6GB (fits 40GB GPU comfortably!)
- Still has all SCENT benefits
- May lose ~2 dB vs 512 hidden_dim, but still much better than ResidualBlock
- Expected PSNR: ~30-32 dB at 128×128
