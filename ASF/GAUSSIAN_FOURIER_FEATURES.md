# Gaussian Fourier Features for Super-Resolution

## The Key Insight

### Current Approach (Deterministic Fourier Features):
```python
omegas = torch.logspace(1, math.log10(sigma), n)
# omegas = [10, 10.5, 11.1, ..., 121, 128]  ← Fixed, discrete frequencies

gamma = [sin(π·x·ω), cos(π·x·ω)] for ω in omegas
```

**Problem**:
- Limited to `n` discrete frequencies
- Band-limited to max frequency σ
- Cannot represent frequencies between the discrete omegas

### Your Idea (Gaussian Fourier Features):
```python
B = torch.randn(n, 2) * sigma  # Random frequency matrix
# Each row is a random 2D frequency vector

gamma = [sin(2π·x·B), cos(2π·x·B)]
```

**Benefits**:
1. **Continuous frequency sampling** - covers frequency space densely
2. **Floating-point precision** - not limited to discrete grid
3. **Better generalization** - random features act as infinite-dimensional kernel
4. **Natural band control** - σ controls frequency distribution, not hard cutoff

## Mathematical Background

### Random Fourier Features (Rahimi & Recht, 2007)

**Theorem**: A shift-invariant kernel k(x-y) can be approximated by random features:

```
k(x, y) = E[φ(x)ᵀ φ(y)]

where φ(x) = [cos(ωᵀx), sin(ωᵀx)] and ω ~ p(ω)
```

For Gaussian kernel `k(x,y) = exp(-||x-y||²/(2σ²))`:
- Sample frequencies from: `ω ~ N(0, σ⁻²I)`

### Why This Helps Super-Resolution

1. **Infinite-dimensional feature space**:
   - Deterministic: n frequencies → 2n features
   - Random: n frequencies → represents infinite kernel space

2. **Smooth frequency coverage**:
   - Deterministic: Gaps between [ω₁, ω₂, ω₃, ...]
   - Random: Dense coverage of frequency space

3. **Better interpolation**:
   - Network learns continuous mapping over ALL frequencies in band
   - Not limited to discrete frequency grid

## Implementation

### Option 1: Replace Deterministic with Gaussian

```python
class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim=64, sigma_q=16, sigma_ls=[128, 32], ...):
        super().__init__()

        # Query Gaussian Fourier features
        self.register_buffer('B_q',
            torch.randn(feature_dim // 2, 2) / sigma_q)

        # Multi-scale Gaussian Fourier features
        self.B_ls = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim // 2, 2) / sigma_ls[i],
                        requires_grad=False)
            for i in range(len(sigma_ls))
        ])

    def calc_gamma_gaussian(self, x, B):
        """
        Gaussian Fourier Features
        x: (HW, 2) - coordinates
        B: (n, 2) - random frequency matrix
        """
        # Project: (HW, 2) @ (2, n) = (HW, n)
        proj = 2 * np.pi * x @ B.T

        # [cos(proj), sin(proj)]
        gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        return gamma  # (HW, 2n)
```

### Option 2: Learnable Gaussian Fourier Features

```python
class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim=64, sigma_q=16, sigma_ls=[128, 32],
                 learnable_freqs=True, ...):
        super().__init__()

        # Initialize from Gaussian
        B_q_init = torch.randn(feature_dim // 2, 2) / sigma_q
        B_ls_init = [torch.randn(feature_dim // 2, 2) / sigma_ls[i]
                     for i in range(len(sigma_ls))]

        if learnable_freqs:
            # Learn frequency matrix during training!
            self.B_q = nn.Parameter(B_q_init)
            self.B_ls = nn.ParameterList([
                nn.Parameter(B_ls_init[i])
                for i in range(len(sigma_ls))
            ])
        else:
            # Fixed random features
            self.register_buffer('B_q', B_q_init)
            self.B_ls = [self.register_buffer(f'B_l{i}', B_ls_init[i])
                        for i in range(len(sigma_ls))]
```

**Learnable frequencies**: Network can adapt frequency distribution during training!

### Option 3: Hybrid (Best of Both Worlds)

```python
def calc_gamma_hybrid(self, x, sigma, n_deterministic=8, n_random=24):
    """
    Combine deterministic + random Fourier features

    - Deterministic: Ensures key frequencies are present
    - Random: Fills gaps with continuous coverage
    """
    # Deterministic component (evenly spaced in log)
    omegas_det = torch.logspace(1, math.log10(sigma), n_deterministic)
    coords_expanded = x.unsqueeze(-1)  # (HW, 2, 1)
    omegas_expanded = omegas_det.view(1, 1, -1)  # (1, 1, n_det)

    arg_det = torch.pi * coords_expanded * omegas_expanded  # (HW, 2, n_det)
    gamma_det = torch.cat([torch.sin(arg_det), torch.cos(arg_det)], dim=-1)
    gamma_det = gamma_det.view(x.shape[0], -1)  # (HW, 4*n_det)

    # Random Gaussian component
    B_random = torch.randn(n_random, 2, device=x.device) / sigma
    proj_random = 2 * np.pi * x @ B_random.T  # (HW, n_random)
    gamma_random = torch.cat([torch.cos(proj_random), torch.sin(proj_random)],
                            dim=-1)  # (HW, 2*n_random)

    # Concatenate
    gamma = torch.cat([gamma_det, gamma_random], dim=-1)

    return gamma  # (HW, 4*n_det + 2*n_random)
```

## Expected Benefits for Super-Resolution

### 1. Better Interpolation Between Training Points

**Deterministic**:
- Training: x ∈ {0, 1/32, 2/32, ..., 31/32}
- Testing: x = 0.5/128 (between training points)
- Features: Discrete frequencies might not interpolate well

**Gaussian**:
- Training: Same x, but continuous frequency coverage
- Testing: x = 0.5/128
- Features: Smooth interpolation due to dense frequency coverage

### 2. Overcome Frequency Grid Aliasing

**Deterministic**:
```
n = 16 frequencies
For σ = 128: [10, 10.5, 11.1, ..., 128]
Spacing increases with frequency → gaps at high freq
```

**Gaussian**:
```
n = 16 random frequencies ~ N(0, 1/128²)
Dense coverage across entire [0, 128] band
No systematic gaps
```

### 3. Implicit Regularization

Random features act as infinite-dimensional kernel approximation:
- Prevents overfitting to discrete training coordinates
- Encourages smooth, continuous mappings
- Better generalization to unseen positions

### 4. Scale-Adaptive Properties

**Key property**: Gaussian frequencies naturally adapt to coordinate scale

```python
# At 32×32 (coords in [0, 1], spacing ≈ 0.03)
B ~ N(0, σ⁻²) with σ = 128
Typical frequency magnitude: ~1/128
Response wavelength: λ ≈ 128 → perfect for 32×32

# At 128×128 (coords in [0, 1], spacing ≈ 0.008)
Same B matrix!
But now queries at finer spacing
→ Captures sub-wavelength variations
→ True super-resolution!
```

## Comparison: Deterministic vs Gaussian

| Aspect | Deterministic | Gaussian |
|--------|--------------|----------|
| Frequency coverage | Discrete grid | Dense, continuous |
| Interpolation | Can have gaps | Smooth everywhere |
| Super-resolution | Limited by grid | Natural generalization |
| Computational cost | Same | Same |
| Initialization | Manual (logspace) | Random sampling |
| Learnability | Fixed structure | Can learn optimal ω |
| Theory | Heuristic | Kernel approximation |

## Implementation Recommendation

### For Best Super-Resolution:

```python
class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim=64, sigma_q=16, sigma_ls=[128, 32],
                 use_gaussian=True, learnable_gaussian=True, ...):
        super().__init__()

        if use_gaussian:
            # Initialize Gaussian Fourier features
            n_features = feature_dim // 2

            # Query features
            B_q_init = torch.randn(n_features, 2) / sigma_q

            # Multi-scale features
            B_ls_init = [torch.randn(n_features, 2) / sigma_ls[i]
                        for i in range(len(sigma_ls))]

            if learnable_gaussian:
                self.B_q = nn.Parameter(B_q_init)
                self.B_ls = nn.ParameterList([
                    nn.Parameter(B_ls_init[i])
                    for i in range(len(sigma_ls))
                ])
            else:
                self.register_buffer('B_q', B_q_init)
                self.B_ls = nn.ModuleList()
                for i in range(len(sigma_ls)):
                    self.register_buffer(f'B_l_{i}', B_ls_init[i])
        else:
            # Original deterministic features
            self.omegas = torch.logspace(1, math.log10(sigma_q), feature_dim // 4)
            self.omegas_l = [torch.logspace(1, math.log10(sigma_ls[i]), feature_dim // 4)
                            for i in range(len(sigma_ls))]

    def calc_gamma(self, x, B_matrix):
        """Gaussian Fourier Features"""
        # x: (HW, 2), B_matrix: (n, 2)
        proj = 2 * np.pi * x @ B_matrix.T  # (HW, n)
        gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (HW, 2n)
        return gamma
```

## Experimental Validation

### Test 1: Frequency Coverage

```python
# Visualize frequency sampling
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Deterministic
omegas_det = torch.logspace(1, math.log10(128), 32).numpy()
axes[0].scatter(omegas_det, np.zeros_like(omegas_det), s=50)
axes[0].set_xlabel('Frequency ω')
axes[0].set_title('Deterministic (32 frequencies)')
axes[0].set_xlim(0, 130)

# Gaussian
B_gauss = torch.randn(32, 2) / 128
freq_magnitudes = torch.norm(B_gauss, dim=1).numpy()
axes[1].scatter(freq_magnitudes, np.zeros_like(freq_magnitudes), s=50, alpha=0.6)
axes[1].set_xlabel('Frequency magnitude ||ω||')
axes[1].set_title('Gaussian (32 random frequencies)')
axes[1].set_xlim(0, 130)
```

### Test 2: Super-Resolution Quality

Train two models:
1. Deterministic Fourier features (baseline)
2. Gaussian Fourier features (your idea)

Compare PSNR at 128×128:
- Expected improvement: +1-3 dB with Gaussian features

### Test 3: Interpolation Smoothness

```python
# Create fine coordinate grid
coords_fine = torch.linspace(0, 1, 1000)

# Encode with both methods
gamma_det = calc_gamma_deterministic(coords_fine)
gamma_gauss = calc_gamma_gaussian(coords_fine)

# Check smoothness (variance of differences)
smoothness_det = torch.diff(gamma_det, dim=0).var()
smoothness_gauss = torch.diff(gamma_gauss, dim=0).var()

# Lower variance = smoother = better interpolation
```

## Potential Issues and Solutions

### Issue 1: Random Initialization Variance

Different random seeds → different frequency matrices → different results

**Solution**: Set seed for reproducibility, or use learnable features

### Issue 2: Scale Sensitivity

σ parameter now critical for frequency distribution

**Solution**: Use multiple scales σ_ls = [128, 32] with separate B matrices

### Issue 3: Computational Cost

Same as deterministic (matrix multiply), no extra cost!

## Summary

Your idea to use **Gaussian Fourier Features** is excellent because:

1. ✅ **Continuous frequency coverage** - no gaps in frequency space
2. ✅ **Better interpolation** - smooth everywhere
3. ✅ **Floating-point precision** - not limited to discrete grid
4. ✅ **Theoretical foundation** - infinite kernel approximation
5. ✅ **Learnable** - can optimize frequency distribution
6. ✅ **No computational overhead** - same cost as deterministic

This should significantly improve super-resolution quality, especially when combined with proper jittering!

Let me implement this in the notebook next.
