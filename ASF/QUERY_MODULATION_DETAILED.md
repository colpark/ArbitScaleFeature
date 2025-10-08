# Detailed Query and Modulation Formation in LAINR

## Complete Forward Pass Trace

Let's trace through EXACTLY what happens when you call `model.decode(lp_features, coords)` for super-resolution:

### Input Example:
- **Training**: 32×32 image → 256 LP tokens
- **Super-resolution**: Query at 128×128 coordinates

---

## Step 1: Coordinate Input

```python
# coords shape: (B, 128, 128, 2)
# Each coord is (y, x) in [0, 1]²
# Example coordinates:
coords[0, 0, 0] = [0.00390625, 0.00390625]  # Top-left pixel
coords[0, 0, 1] = [0.00390625, 0.01171875]  # Second pixel
coords[0, 63, 63] = [0.4921875, 0.4921875]  # Middle pixel
```

**Key point**: These are DENSE coordinates (16,384 points for 128×128)

---

## Step 2: Query Fourier Encoding

```python
def calc_gamma(self, x, omegas):
    """
    x: (HW, 2) - flattened coordinates
    omegas: (n,) - frequency basis
    """
    # For 128×128: x.shape = (16384, 2)

    # Expand dimensions
    coords = x.unsqueeze(-1)  # (16384, 2, 1)
    omegas = omegas.view(1, 1, -1)  # (1, 1, 16)

    # Broadcast: arg.shape = (16384, 2, 16)
    arg = π * coords * omegas

    # Compute sin/cos
    sin_part = torch.sin(arg)  # (16384, 2, 16)
    cos_part = torch.cos(arg)  # (16384, 2, 16)

    # Concatenate: [sin(πx·ω), cos(πx·ω), sin(πy·ω), cos(πy·ω)]
    gamma = torch.cat([sin_part, cos_part], dim=-1)  # (16384, 2, 32)
    gamma = gamma.view(16384, -1)  # (16384, 64)

    return gamma
```

### What `omegas` Contains (Line 24):

```python
self.omegas = torch.logspace(1, math.log10(sigma_q), self.n)
# sigma_q = 16
# n = 16 (for feature_dim=64, input_dim=2)

# omegas = [10.00, 10.32, 10.65, ..., 15.08, 15.56, 16.00]
# These are the QUERY frequencies
```

### Example Encoding for One Pixel:

For pixel at coords `(0.5, 0.5)` (center):

```python
ω = [10, 10.32, 10.65, ..., 16]

# X-coordinate encoding:
sin(π·0.5·10)  = sin(5π)   = 0.0
cos(π·0.5·10)  = cos(5π)   = -1.0
sin(π·0.5·10.32) = sin(5.16π) = 0.498
cos(π·0.5·10.32) = cos(5.16π) = -0.867
...
sin(π·0.5·16)  = sin(8π)   = 0.0
cos(π·0.5·16)  = cos(8π)   = 1.0

# Y-coordinate encoding: (same, since y=0.5)
[same pattern]

# Final gamma: (64,) vector
γ = [0.0, -1.0, 0.498, -0.867, ..., 0.0, 1.0, ...]
```

**Result**: Each of 16,384 pixels gets a unique 64-dim positional encoding

---

## Step 3: Query Projection

```python
# Line 103-105
x_q = einops.repeat(
    self.calc_gamma(x[0], self.omegas), 'l d -> b l d', b=B
)
# x_q shape: (B, 16384, 64)

x_q = self.act(self.query_lin(x_q))
# x_q shape: (B, 16384, 256)
```

**What this does**:
- Projects 64-dim Fourier features → 256-dim query space
- Each pixel has a unique query vector based on its POSITION
- Linear layer learns to map "position encoding" → "what to ask for"

### Example Query Vector:

```python
# For pixel at (0.5, 0.5):
gamma = [0.0, -1.0, 0.498, ...]  # (64,)

# After linear projection:
query = ReLU(W_q @ gamma + b_q)  # (256,)

# This query encodes: "I'm at the CENTER position, give me relevant features"
```

---

## Step 4: Spatial Bias Computation

```python
# Line 167-170
rel_distances = self.approximate_relative_distances(
    indexes, self.patch_num, self.patch_num, tokens.shape[1]
)
# rel_distances shape: (256, 16384)

bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)
# bias shape: (B, 256, 16384)
```

### What `approximate_relative_distances` Does:

```python
def approximate_relative_distances(self, target_index, H, W, m):
    """
    target_index: (16384,) - which PATCH each pixel belongs to
    H, W: patch grid (16, 16) for 256 patches
    m: num LP tokens (256)
    """
    N = H * W  # 256 patches

    # Normalize patch indices to [0, 1]
    t = target_index.float() / N
    # t[0] = 0/256 = 0.0        (pixel 0 in patch 0)
    # t[1023] = 0/256 = 0.0     (pixel 1023 in patch 0)
    # t[1024] = 1/256 = 0.0039  (pixel 1024 in patch 1)
    # ...

    # LP token positions (evenly spaced in [0, 1])
    token_positions = [(i + 0.5) / m for i in range(m)]
    # [0.00195, 0.00586, 0.00977, ..., 0.99805]

    # Broadcast to (256, 16384)
    t_expanded = t.unsqueeze(0)  # (1, 16384)
    tokens_expanded = token_positions.unsqueeze(1)  # (256, 1)

    # Distance-based bias: -α * |Δ|²
    rel_distances = -10.0 * torch.abs(t_expanded - tokens_expanded)**2
    # Shape: (256, 16384)

    return rel_distances
```

### Example Bias Values:

For pixel at position `(64, 64)` in 128×128 grid:
- Belongs to patch `(8, 8)` in 16×16 grid
- Patch index = 8*16 + 8 = 136
- Normalized position: `t = 136/256 = 0.531`

```python
# Distance to each LP token:
LP[0]   at 0.00195: |0.531 - 0.00195|² = 0.280  → bias = -2.80
LP[136] at 0.531:   |0.531 - 0.531|²   = 0.0    → bias = 0.0    ← CLOSEST!
LP[255] at 0.998:   |0.531 - 0.998|²   = 0.218  → bias = -2.18
```

**Result**: Bias encourages attention to spatially nearby LP tokens

---

## Step 5: Cross-Attention Modulation Extraction

```python
# Line 177
modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

# Inside SharedTokenCrossAttention:
def forward(self, x, context, bias):
    """
    x: (B, 16384, 256) - queries (one per pixel)
    context: (B, 256, 256) - LP tokens
    bias: (B, 256, 16384) - spatial bias
    """
    B, HW, D = x.shape  # HW = 16384

    # Project to Q, K, V
    q = self.to_q(x)  # (B, 16384, 128) [heads=2, dim_head=64 → 128]
    kv = self.to_kv(context)  # (B, 256, 256)
    k, v = kv.chunk(2, dim=-1)  # Each: (B, 256, 128)

    # Reshape for multi-head attention
    q = q.view(B, 16384, 2, 64).transpose(1, 2)  # (B, 2, 16384, 64)
    k = k.view(B, 256, 2, 64).transpose(1, 2)    # (B, 2, 256, 64)
    v = v.view(B, 256, 2, 64).transpose(1, 2)    # (B, 2, 256, 64)

    # Attention scores
    sim = q @ k.transpose(-1, -2) / √64  # (B, 2, 16384, 256)

    # Add spatial bias
    bias = einops.repeat(bias, 'b l n -> b h l n', h=2)  # (B, 2, 256, 16384)
    bias = bias.transpose(-2, -1)  # (B, 2, 16384, 256)
    sim = sim + bias  # SPATIAL BIAS INJECTION!

    # Softmax attention
    attn = sim.softmax(dim=-1)  # (B, 2, 16384, 256)

    # Weighted sum of values
    out = attn @ v  # (B, 2, 16384, 64)
    out = out.transpose(1, 2).reshape(B, 16384, 128)  # (B, 16384, 128)
    out = self.to_out(out)  # (B, 16384, 256)

    return out
```

### What Happens for One Pixel:

For pixel at `(64, 64)` (patch 136):

**1. Query vector**: `q = [q₁, q₂, ..., q₆₄]` (after splitting to 2 heads)

**2. Keys from LP tokens**:
```
LP[0]:   k₀ = [k₀₁, k₀₂, ..., k₀₆₄]
LP[1]:   k₁ = [k₁₁, k₁₂, ..., k₁₆₄]
...
LP[136]: k₁₃₆ = [k₁₃₆,₁, ..., k₁₃₆,₆₄]  ← Spatially closest
...
LP[255]: k₂₅₅ = [k₂₅₅,₁, ..., k₂₅₅,₆₄]
```

**3. Attention scores (before bias)**:
```
sim₀   = (q · k₀) / √64   = 0.23
sim₁   = (q · k₁) / √64   = -0.15
...
sim₁₃₆ = (q · k₁₃₆) / √64 = 0.87
...
```

**4. Attention scores (after spatial bias)**:
```
sim₀   = 0.23 + (-2.80) = -2.57
sim₁₃₆ = 0.87 + (0.0)   = 0.87   ← BOOSTED!
sim₂₅₅ = 0.11 + (-2.18) = -2.07
```

**5. Softmax attention weights**:
```
α₀   = exp(-2.57) / Z = 0.001
α₁₃₆ = exp(0.87) / Z  = 0.823   ← DOMINANT!
α₂₅₅ = exp(-2.07) / Z = 0.004
```

**6. Modulation vector (weighted sum of values)**:
```
modulation = Σ αᵢ · vᵢ
           ≈ 0.823 · v₁₃₆ + 0.001·v₀ + ... + 0.004·v₂₅₅
           ≈ v₁₃₆  (dominated by nearby LP token!)
```

**Result**: Each pixel extracts features from spatially nearby LP tokens

---

## Step 6: Multi-Scale Bandwidth Encoding

```python
# Line 184-192
for k in range(self.layer_num):  # 2 scales: σ=[128, 32]
    # Encode coordinates at THIS frequency scale
    x_l = einops.repeat(
        self.calc_gamma(x[0], self.omegas_l[k]), 'l d -> b l d', b=B
    )
    # x_l shape: (B, 16384, 64)

    h_l = self.act(self.bandwidth_lins[k](x_l))
    # h_l shape: (B, 16384, 256)

    m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
    # m_l shape: (B, 16384, 256)
```

### Scale 1 (Coarse, σ=128):

```python
omegas_l[0] = logspace(10, 128, 16)
# [10.0, 10.54, ..., 121.4, 128.0]

# For pixel at (0.5, 0.5):
gamma_coarse = [sin(π·0.5·10), cos(π·0.5·10), ..., sin(π·0.5·128), cos(π·0.5·128)]
# = [0.0, -1.0, ..., 0.0, 1.0]

h_coarse = ReLU(W_coarse @ gamma_coarse)  # (256,)
modulation_proj = W_mod @ modulation  # (256,)

output_coarse = ReLU(h_coarse + modulation_proj)
```

**What this encodes**:
- `h_coarse`: Position encoding at COARSE frequency (low-freq structure)
- `modulation_proj`: Image-specific features from LP tokens
- Sum: "At THIS position, with THESE image features, at COARSE scale"

### Scale 2 (Fine, σ=32):

```python
omegas_l[1] = logspace(10, 32, 16)
# [10.0, 10.27, ..., 30.2, 32.0]

# For pixel at (0.5, 0.5):
gamma_fine = [sin(π·0.5·10), ..., sin(π·0.5·32), cos(π·0.5·32)]

h_fine = ReLU(W_fine @ gamma_fine)
output_fine = ReLU(h_fine + modulation_proj)
```

**What this encodes**:
- `h_fine`: Position encoding at FINE frequency (high-freq details)
- Same modulation as coarse scale
- Sum: "At THIS position, with THESE image features, at FINE scale"

---

## Step 7: Residual Connections

```python
# Line 195-198
h_v = [modulations_l[0]]  # Start with coarse

for i in range(self.layer_num - 1):
    h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
    h_v.append(h_vl)

# h_v[0] = coarse features
# h_v[1] = fine features + residual from coarse
```

**Effect**: Fine-scale features build on coarse-scale features (hierarchical)

---

## Step 8: Multi-Scale Output

```python
# Line 201-202
outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]
# outs[0]: (B, 16384, 3) - coarse RGB
# outs[1]: (B, 16384, 3) - fine RGB

out = sum(outs)  # (B, 16384, 3)
# Final RGB = coarse + fine
```

---

## The Problem: Information Bottleneck

### What LP Tokens Contain:

LP tokens were created from **32×32 patches**:
```
32×32 image
  → 16×16 patch grid (256 patches of 2×2 pixels each)
  → Patch embedding: 256 tokens × 256 dims
  → Mamba encoding: 256 LP tokens × 256 dims
```

Each LP token encodes information from a **2×2 pixel patch**.

**Maximum frequency content**: Limited by 2×2 sampling
- Nyquist limit: 1 cycle per pixel
- In normalized coords: ~50 cycles per unit

### At Super-Resolution (128×128):

You're asking each of 16,384 pixels to extract features from 256 LP tokens.

**Problem**: LP tokens DON'T contain information beyond σ≈50!

Even though you encode queries at σ=128:
```python
gamma_fine = [sin(π·x·128), cos(π·x·128)]
```

The **modulation vector** (which carries image content) is LIMITED to σ≈50.

### Mathematical View:

```
h_fine(x,y) = ReLU(W_fine @ γ_fine(x,y))       ← Can represent σ=128
modulation(x,y) = CrossAttention(query, LP)    ← Limited to σ≈50

output(x,y) = h_fine + W_mod @ modulation
            = [position at σ=128] + [content at σ=50]
            = BAND-LIMITED to σ=50!
```

---

## Summary: Why It's Resampling

1. **Queries encode high-frequency positions** (σ=128) ✓
2. **But modulation vectors are band-limited** (σ≈50) ✗
3. **Final output = position × content**
   - Position: can be high-freq
   - Content: limited by LP tokens
   - **Result: Smooth interpolation of band-limited signal**

This is equivalent to:
1. Bicubic interpolation (σ≈50)
2. + Learned filtering (from network)
3. = **High-quality resampling**, not super-resolution

---

## To Achieve True Super-Resolution

You need LP tokens to contain higher-frequency information:

**Option 1**: Train on higher resolution images
**Option 2**: Multi-scale encoder (encode 32×32 AND 64×64)
**Option 3**: High-frequency hallucination network (learn texture priors)
**Option 4**: Adversarial loss (discriminator forces sharp details)

The architecture CANNOT generate frequencies beyond what LP tokens saw during encoding!
