# Memory Bottleneck Analysis: SCENT-Style Decoder

## Memory Explosion: Self-Attention on 32×32 = 1024 Tokens

### Problem Identified:

**Current implementation:**
- Training resolution: 32×32 = **1,024 query positions**
- Each position goes through **self-attention** in decoder
- Self-attention memory: O(N²) where N = 1,024

### Memory Breakdown:

#### 1. **Bandwidth Encoders** (per scale, 2 scales):
```python
# Each has 2 self-attention layers
self.bandwidth_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        PreNorm(512, Attention(512, heads=8, dim_head=64)),  # Self-attention on 1024 tokens!
        PreNorm(512, FeedForward(512, mult=4))
    )
    for _ in range(2)  # 2 scales
])
```

**Memory per self-attention:**
- Query: (B, 1024, 512) → (B*8, 1024, 64)
- Key: (B, 1024, 512) → (B*8, 1024, 64)
- Value: (B, 1024, 512) → (B*8, 1024, 64)
- Attention matrix: **(B*8, 1024, 1024)** ← **MASSIVE!**
  - For B=64, heads=8: (512, 1024, 1024) = **512M float32** = **2GB per attention!**

**Total for bandwidth_lins:**
- 2 scales × 2 self-attention layers = 4 self-attentions
- 4 × 2GB = **8GB**

#### 2. **Modulation Projections** (per scale, 2 scales):
```python
self.modulation_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        PreNorm(512, Attention(512, heads=8)),  # Self-attention on 1024 tokens!
        PreNorm(512, FeedForward(512, mult=4))
    )
    for _ in range(2)
])
```

**Memory:** 2 scales × 2 self-attentions = 4 × 2GB = **8GB**

#### 3. **Hidden Value Layers**:
```python
self.hv_lins = nn.ModuleList([
    nn.ModuleList([
        PreNorm(512, Attention(512, heads=8)),  # Self-attention on 1024 tokens!
        PreNorm(512, FeedForward(512, mult=4))
    ])
    for _ in range(1)  # layer_num - 1 = 1
])
```

**Memory:** 1 × 2GB = **2GB**

#### 4. **Total Decoder Self-Attention Memory:**
- Bandwidth: 8GB
- Modulation: 8GB
- Hidden value: 2GB
- **Total: ~18GB** just for attention matrices in decoder!

#### 5. **Plus Gradients (backprop):**
- Each attention matrix needs gradients: **×2**
- **Total with gradients: ~36GB**

#### 6. **Plus Model Parameters, Activations, Optimizer States:**
- BiMamba encoder: ~2GB
- Other activations: ~2GB
- **Grand total: ~40GB** ✓ Explains the OOM!

---

## Root Cause:

**Self-attention on 1,024 positions is killing memory!**

The original MAMBA-GINR didn't have this problem because:
- It used **ResidualBlocks** (local, no attention)
- ResidualBlock memory: O(N) not O(N²)

SCENT works because:
- It processes **fewer query positions** per forward pass
- Or uses **gradient checkpointing**

---

## Solutions (Ranked by Effectiveness):

### Solution 1: Remove Self-Attention from Decoder ⭐⭐⭐⭐⭐
**Change:** Use FeedForward only (no self-attention in decoder)

```python
# CURRENT (OOM):
self.bandwidth_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        PreNorm(512, Attention(512, heads=8)),  # ← REMOVE THIS
        PreNorm(512, FeedForward(512, mult=4))
    )
    for _ in range(2)
])

# NEW (MEMORY EFFICIENT):
self.bandwidth_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        PreNorm(512, FeedForward(512, mult=4)),  # Only FeedForward
        PreNorm(512, FeedForward(512, mult=4))   # Stack more FFs instead
    )
    for _ in range(2)
])
```

**Memory savings:** 18GB → 0.5GB (36x reduction!)

**Expected impact:** Still much better than ResidualBlock because:
- 4x expansion in FeedForward (512 → 2048 → 512)
- GEGLU gating (more expressive than ReLU)
- Skip connections from Fourier encoding
- Keep self-attention in latent processor (256 tokens, not 1024)

---

### Solution 2: Use Cross-Attention Instead ⭐⭐⭐⭐
**Change:** Instead of self-attention on 1024 positions, use cross-attention to LP tokens (256)

```python
# Instead of:
Attention(512, heads=8)  # Self-attention on 1024

# Use:
Attention(512, context_dim=256, heads=8)  # Cross-attention to LP tokens
# Then pass lp_features as context
```

**Memory:** (B*8, 1024, 256) instead of (B*8, 1024, 1024)
- Saves: 1024/256 = 4x per attention
- Total: 18GB → 4.5GB

---

### Solution 3: Reduce Batch Size ⭐⭐
**Change:** B = 64 → 16

**Memory savings:** 40GB → 10GB

**Downside:** 4x slower training

---

### Solution 4: Gradient Checkpointing ⭐⭐⭐
**Change:** Trade compute for memory

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x, block):
    return checkpoint(block, x, use_reentrant=False)
```

**Memory savings:** ~50% (stores only some activations)

**Downside:** ~30% slower

---

### Solution 5: Reduce Resolution or Num Layers ⭐
**Not recommended** - defeats the purpose

---

## Recommended Fix: Solution 1

**Replace self-attention with stacked FeedForwards in decoder:**

This gives you:
1. ✅ 4x expansion (512 → 2048 → 512) - better than ResidualBlock
2. ✅ GEGLU gating - better than ReLU
3. ✅ Skip connections - preserves high-freq
4. ✅ Memory efficient - O(N) not O(N²)
5. ✅ Keep self-attention in latent processor (256 tokens)

**Expected result:**
- Memory: ~5GB (fits easily in 40GB GPU)
- Still much better than ResidualBlock
- May lose ~1-2 dB vs full self-attention, but still +3-5 dB vs ResidualBlock

---

## Quick Fix Code:

Replace the decoder blocks with this pattern:

```python
# Bandwidth encoders - NO SELF-ATTENTION
self.bandwidth_lins = nn.ModuleList([
    nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        *[
            nn.Sequential(
                PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4)),
                PreNorm(hidden_dim, FeedForward(hidden_dim, mult=4))
            )
            for _ in range(num_layers - 1)
        ]
    )
    for _ in range(self.layer_num)
])

# Same for modulation_lins and hv_lins
```

This stacks **2 FeedForwards per layer** instead of **1 Attention + 1 FeedForward**.

**Memory:** Each FF is ~0.1GB vs Attention ~2GB → 20x savings!
