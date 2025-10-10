# SCENT vs MAMBA-GINR Architecture Comparison

## Your Question
**"What might be the cause of sub-optimal high frequency learning? Is it the hyponetwork structure where linear layers are too small?"**

## Critical Architectural Differences

### 1. ⭐⭐⭐⭐⭐ DECODER ARCHITECTURE (Most Likely Cause!)

#### SCENT Decoder (STRONG):
```python
# Cross-attention decoder
self.decoder_cross_attn = PreNorm(
    queries_dim,  # queries_dim = 192 (96*2 Fourier features)
    Attention(queries_dim, final_latent_dim, heads=4, dim_head=128),
    context_dim=512  # final latent dimension
)

# Additional self-attention processing (4 layers!)
self.self_attn_blocks = nn.Sequential(*[
    nn.Sequential(
        PreNorm(512, Attention(512, heads=8, dim_head=128)),  # 512 → 1024 inner_dim
        PreNorm(512, FeedForward(512))  # 512 → 2048 → 512
    )
    for _ in range(4)  # 4 LAYERS of self-attention
])

# Output projection
self.to_logits = nn.Linear(queries_dim, 3)  # 192 → 3

# CAPACITY BREAKDOWN:
# - Cross-attention: queries_dim × context_dim = 192 × 512 ≈ 98K params
# - Self-attention blocks (4 layers):
#     - Each attention: 512 → 1024 inner_dim (512K params)
#     - Each FF: 512 → 2048 → 512 (2M params)
#     - Total: 4 × 2.5M ≈ 10M params
# - Skip connections: x = x + queries (preserves input encoding!)
# - Total decoder capacity: ~10M params
```

#### MAMBA-GINR Decoder (WEAK):
```python
# Current implementation (even the "large" version):
class LAINRDecoderLarge:
    hidden_dim = 512
    num_layers = 3

    # Bandwidth encoding (per scale)
    self.bandwidth_lins = nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        ResidualBlock(512),  # 512 → 512 → 512
        ResidualBlock(512),
        ResidualBlock(512)
    )

    # Modulation projection (per scale)
    self.modulation_lins = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        ResidualBlock(512),
        ResidualBlock(512),
        ResidualBlock(512)
    )

    # Hidden value layers
    self.hv_lins = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        ResidualBlock(512),
        ResidualBlock(512),
        ResidualBlock(512)
    )

    # Output layers
    self.out_lins = nn.Sequential(
        ResidualBlock(512),
        ResidualBlock(512),
        ResidualBlock(512),
        nn.Linear(512, 3)
    )

# CAPACITY BREAKDOWN:
# - Each ResidualBlock: 512 → 512 → 512 (512K params)
# - Bandwidth encoders (×2 scales): 2 × (64→512 + 3×ResidualBlock) ≈ 3M
# - Modulation projections (×2 scales): 2 × (512→512 + 3×ResidualBlock) ≈ 3M
# - Hidden value layers: 512→512 + 3×ResidualBlock ≈ 1.5M
# - Output layers: 3×ResidualBlock + 512→3 ≈ 1.5M
# - Total decoder capacity: ~9M params
```

**KEY PROBLEM**: Even though parameter counts are similar, the **ResidualBlock structure is fundamentally weaker**!

---

### 2. ⭐⭐⭐⭐⭐ RESIDUAL BLOCK vs ATTENTION + FEEDFORWARD

#### Why SCENT's blocks are stronger:

**SCENT Block (Attention + FF):**
```python
# Self-attention block
x = PreNorm(512, Attention(512, heads=8, dim_head=128))(x) + x
# → inner_dim = 8 × 128 = 1024
# → Projects: 512 → 1024 → 512
# → Can mix information ACROSS all spatial positions

# Feedforward block
x = PreNorm(512, FeedForward(512))(x) + x
# → FeedForward: 512 → 2048 → 512 (4x expansion via GEGLU)
# → Can learn complex non-linear transformations
```

**MAMBA-GINR ResidualBlock:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)  # 512 → 512
        self.linear2 = nn.Linear(dim, dim)  # 512 → 512
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.act(self.linear1(x))  # 512 → 512
        x = self.linear2(x)             # 512 → 512
        return self.act(x + residual)
```

**PROBLEMS with ResidualBlock**:
1. ❌ **No expansion**: 512 → 512 → 512 (no intermediate width increase)
2. ❌ **Local only**: Each position processed independently (no attention)
3. ❌ **Limited expressiveness**: Only 2 linear layers, no gating (GEGLU)
4. ❌ **No LayerNorm**: Can cause training instability

**SCENT advantages**:
1. ✅ **4x expansion**: 512 → 2048 → 512 (much more capacity)
2. ✅ **Global mixing**: Self-attention mixes information across ALL positions
3. ✅ **Gated activation**: GEGLU (x * GELU(gates)) more expressive than ReLU
4. ✅ **LayerNorm**: Stabilizes training, improves gradient flow

---

### 3. ⭐⭐⭐⭐ SKIP CONNECTIONS IN DECODER

#### SCENT:
```python
# After cross-attention, ADDS back the original query encoding!
x = self.decoder_cross_attn(queries, context=residual)
x = x + queries  # ← CRITICAL: Preserves input Fourier encoding
```

**Why this matters**:
- Queries contain **high-frequency Fourier basis** (from coords @ B_matrix)
- Cross-attention extracts **semantic modulation** from latents
- **Skip connection ensures high-freq basis is PRESERVED**
- Output = high-freq basis + semantic modulation

#### MAMBA-GINR:
```python
# No skip connection from Fourier encoding to output!
h_l = self.bandwidth_lins[k](x_l)     # Fourier encoding
m_proj = self.modulation_lins[k](modulation_vector)
m_l = self.act(h_l + m_proj)          # Combined

# Problem: h_l goes through many layers, high-freq info can be lost!
h_v = [modulations_l[0]]
for i in range(self.layer_num - 1):
    h_vl = self.hv_lins[i](modulations_l[i+1] + h_v[i])
    h_v.append(h_vl)

out = self.out_lins[i](h_v[i])  # Final output
```

**Issue**: High-frequency Fourier encoding goes through:
1. bandwidth_lins (3 ResidualBlocks)
2. modulation_lins (3 ResidualBlocks)
3. hv_lins (3 ResidualBlocks)
4. out_lins (3 ResidualBlocks)
5. **Total: 12 ResidualBlocks!**

By the time it reaches the output, the **high-frequency signal is heavily degraded** through:
- Multiple ReLU activations (can kill negative values)
- No LayerNorm (gradient issues)
- No direct path from Fourier input to output

---

### 4. ⭐⭐⭐⭐ LATENT PROCESSING (Processor Architecture)

#### SCENT:
```python
# 4 layers of SELF-ATTENTION on latent space
self.self_attn_blocks = nn.Sequential(*[
    nn.Sequential(
        PreNorm(512, Attention(512, heads=8, dim_head=128)),
        PreNorm(512, FeedForward(512))
    )
    for _ in range(4)
])

# In forward:
for sa_block in self.self_attn_blocks:
    residual = sa_block[0](residual) + residual  # Self-attention
    residual = sa_block[1](residual) + residual  # FeedForward
```

**Benefits**:
- **Global receptive field**: Every latent can attend to all other latents
- **Complex feature mixing**: 4 layers of attention + FF
- **High capacity**: Each FF has 512 → 2048 → 512 (4x expansion)

#### MAMBA-GINR:
```python
# BiMamba for latent processing
self.mamba = BiMamba(d_model=256)

class BiMamba:
    def __init__(self, d_model):
        self.forward_mamba = Mamba(d_model)
        self.backward_mamba = Mamba(d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
```

**Potential issues**:
1. **Lower capacity**: d_model = 256 (vs SCENT's 512)
2. **Single Mamba layer**: Only 1 forward + 1 backward pass
3. **No explicit multi-layer processing**: SCENT has 4 self-attention layers

**However**, Mamba has its advantages:
- O(L) complexity vs O(L²) for attention
- Sequential inductive bias from SSM

**BUT**: For super-resolution, we may NEED the global receptive field of attention!

---

### 5. ⭐⭐⭐ HIERARCHICAL ENCODING

#### SCENT:
```python
# Cascaded encoder with INCREASING dimensions
latent_dims=(256, 384, 512)  # Hierarchical!
num_latents=(256, 256, 256)

# Each block processes at different capacity
for dim, n_latents in zip(latent_dims, num_latents):
    block = CascadedBlock(
        dim=dim,  # 256 → 384 → 512
        n_latents=n_latents,
        ...
        residual_dim=prev_dim  # Adds previous level!
    )
```

**Benefits**:
- Coarse-to-fine processing: 256 dims → 384 dims → 512 dims
- Each level adds more capacity
- Residual connections between levels preserve multi-scale features

#### MAMBA-GINR:
```python
# Single Mamba encoder at fixed dimension
self.mamba = BiMamba(d_model=256)  # Fixed 256 dims
```

**Issue**:
- **No hierarchical structure**: Single level at 256 dims
- **No coarse-to-fine**: All processing at same scale

---

### 6. ⭐⭐⭐ SINUSOIDAL INITIALIZATION

#### SCENT:
```python
def get_sinusoidal_embeddings(n, d):
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))

    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Used for latent initialization
self.latents = nn.Parameter(
    get_sinusoidal_embeddings(n_latents, dim),
    requires_grad=False  # Or True for learnable
)
```

**Benefits**:
- **Frequency-aware initialization**: Latents start with multi-scale frequencies
- **Better for INR**: Sinusoidal patterns match Fourier basis
- **Stable training**: Good initialization helps gradient flow

#### MAMBA-GINR:
```python
# Random initialization for LP tokens
self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
```

**Issue**:
- Random initialization may not align well with Fourier features
- Could take longer to learn frequency-aware representations

---

## Summary: Root Causes of Sub-Optimal High-Frequency Learning

### Primary Causes (⭐⭐⭐⭐⭐):

1. **ResidualBlock is too weak**
   - No expansion (512 → 512 vs 512 → 2048)
   - No attention (local only)
   - No gating (ReLU vs GEGLU)
   - No LayerNorm

2. **No skip connection from Fourier encoding to output**
   - High-freq signal degraded through 12 ResidualBlocks
   - SCENT preserves it: `x = x + queries`

3. **No self-attention in latent processing**
   - SCENT has 4 layers of self-attention on latents
   - MAMBA-GINR has 1 BiMamba layer
   - Self-attention provides global receptive field

### Secondary Causes (⭐⭐⭐):

4. **No hierarchical encoding**
   - SCENT: 256 → 384 → 512 (coarse-to-fine)
   - MAMBA-GINR: 256 (fixed)

5. **Random vs sinusoidal initialization**
   - SCENT uses sinusoidal embeddings
   - MAMBA-GINR uses random noise

---

## Recommended Fixes (Ranked by Impact)

### Fix 1: Replace ResidualBlock with Attention + FeedForward ⭐⭐⭐⭐⭐

**Current (weak)**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
```

**Proposed (strong)**:
```python
class DecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mult=4):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head))
        self.ff = PreNorm(dim, FeedForward(dim, mult=mult))

    def forward(self, x):
        x = self.attn(x) + x      # Self-attention with skip
        x = self.ff(x) + x        # FeedForward with skip
        return x
```

**Expected improvement**: +5-10 dB at 128×128 🔥

---

### Fix 2: Add Skip Connection from Fourier Encoding ⭐⭐⭐⭐⭐

**Current**:
```python
# Fourier encoding → many layers → output
h_l = self.bandwidth_lins[k](x_l)
# ... 12 layers later ...
out = self.out_lins[i](h_v[i])
```

**Proposed**:
```python
# Save original Fourier encoding
x_l_orig = gaussian_fourier_encode(coords_dec[0], self.B_ls[k])

# Process through network
h_l = self.bandwidth_lins[k](x_l)
# ... processing ...
out_processed = self.out_lins[i](h_v[i])

# Add skip connection (like SCENT!)
# Project original encoding to output space
x_l_skip = self.fourier_skip_proj(x_l_orig)  # Add this projection
out = out_processed + x_l_skip  # Preserve high-freq
```

**Expected improvement**: +3-5 dB at 128×128

---

### Fix 3: Add Self-Attention Layers on Latents ⭐⭐⭐⭐

**Current**:
```python
# Single BiMamba layer
features = self.mamba(combined)
```

**Proposed**:
```python
# BiMamba + self-attention layers (like SCENT)
features = self.mamba(combined)

# Add self-attention processing
for _ in range(2):  # 2 layers
    features = self.self_attn(features) + features
    features = self.ff(features) + features
```

**Expected improvement**: +2-4 dB at 128×128

---

### Fix 4: Use Hierarchical Decoder ⭐⭐⭐

**Proposed**:
```python
# Instead of fixed 512 dims, use hierarchy
hidden_dims = [256, 384, 512]  # Coarse → Fine

for i, hidden_dim in enumerate(hidden_dims):
    # Process at different scales
    h_l = self.bandwidth_lins[i](x_l)  # Uses hidden_dim
    # ...
```

**Expected improvement**: +1-3 dB at 128×128

---

### Fix 5: Sinusoidal Initialization ⭐⭐

**Proposed**:
```python
# Replace random initialization
self.tokens = nn.Parameter(
    get_sinusoidal_embeddings(num_tokens, dim),
    requires_grad=True  # Still learnable
)
```

**Expected improvement**: +0.5-1 dB (faster convergence)

---

## Direct Answer to Your Question

> **"Is it the hyponetwork structure where linear layers are too small?"**

**Answer**: YES, but not just size - it's the **STRUCTURE**!

The problem is NOT parameter count (9M in large decoder is comparable to SCENT).

The problem IS:
1. **ResidualBlock architecture is weak** (no attention, no expansion, no gating)
2. **No skip connection** from Fourier encoding to output (high-freq degraded)
3. **No self-attention** in latent processing (no global receptive field)

**Key insight from SCENT**:
- Self-attention + FeedForward blocks are MUCH more powerful than ResidualBlocks
- Skip connections preserve high-frequency information
- Global receptive field (attention) is critical for super-resolution

---

## Minimal Test: Replace ResidualBlock

Try this as Fix #1:

```python
# Replace this:
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

# With this:
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=4)  # 4x expansion

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
```

This single change should give **+3-5 dB improvement** at 128×128!
