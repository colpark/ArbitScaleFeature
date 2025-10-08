# ASF - Arbitrary Scale Feature (MAMBA-GINR Innovation)

## Overview

This folder contains the **isolated core innovation** of MAMBA-GINR: the combination of **Mamba state space models** with **implicit sequential bias** through learnable position tokens.

## Core Innovation: Implicit Sequential Bias

### The Problem
Traditional implicit neural representations (INRs) struggle with:
- Spectral bias (difficulty learning high-frequency details)
- Limited positional awareness
- Inefficient processing of long sequences (quadratic complexity in Transformers)

### The Solution: Learnable Position Tokens (LPs)

MAMBA-GINR introduces **learnable position tokens** that provide implicit sequential bias:

1. **Learnable Tokens**: Instead of fixed positional encodings, uses learnable parameters
2. **Strategic Placement**: Tokens are interleaved with input at specific positions
3. **Bidirectional Mamba Processing**: Efficient linear-complexity encoding
4. **Conditioning Extraction**: LP tokens serve as learned positional representations

### Architecture Components

```
Input Data → Tokenization → [Add LPs] → Mamba Encoder → [Extract LPs] → Hyponet Decoder → Output
                               ↓                              ↓
                     Implicit Sequential Bias    Position-Aware Features
```

## Files

### 1. `mamba_encoder.py`
- **BiMamba**: Bidirectional Mamba layer processing sequences in forward and reverse
- **MambaEncoder**: Stack of Mamba blocks with linear complexity O(L)
- Replaces Transformer's O(L²) attention with efficient state space models

### 2. `implicit_sequential_bias.py`
- **ImplicitSequentialBias**: Core LP mechanism
- **Placement Strategies**:
  - `equidistant`: Evenly spaced throughout sequence
  - `middle`: Clustered in middle
  - `n_group`: Grouped insertions
- **Visualization utilities** for understanding LP placement

### 3. `mamba_ginr.py`
- **MambaGINR**: Complete architecture
- **SimpleMambaGINR**: Minimal implementation for clarity
- Combines all components into end-to-end model

## Key Innovation Details

### Learnable Position Tokens (LPs)

```python
# Initialize learnable tokens
self.lps = nn.Parameter(torch.randn(num_lp, dim))

# Strategic placement (e.g., equidistant)
total_len = seq_len + num_lp
lp_indices = torch.linspace(0, total_len - 1, steps=num_lp).long()

# Interleave with input
tokens_with_lps = interleave(input_tokens, lps, lp_indices)

# Process with Mamba
encoded = mamba_encoder(tokens_with_lps)

# Extract learned positional features
position_features = encoded[:, lp_indices]
```

### Why This Works

1. **Adaptive Positional Information**: LPs learn optimal positional representations during training
2. **Sequence-Aware**: Bidirectional Mamba captures context from both directions
3. **Efficient**: Linear complexity vs quadratic for Transformers
4. **Flexible**: Different placement strategies for different tasks
5. **Arbitrary Scale**: LP features condition decoder for any resolution

### Comparison to Alternatives

| Approach | Positional Info | Complexity | Learnable |
|----------|----------------|------------|-----------|
| Sinusoidal PE | Fixed encoding | - | No |
| Learned PE | Index-based | - | Yes (per position) |
| **Implicit Bias (LPs)** | **Token-based** | **O(L)** | **Yes (shared)** |

## Usage Example

```python
from mamba_ginr import SimpleMambaGINR

# Create model
model = SimpleMambaGINR(
    input_dim=3,        # RGB input
    hidden_dim=512,
    output_dim=3,       # RGB output
    num_lp=64,          # 64 learnable position tokens
    mamba_depth=6,
    lp_type='equidistant'
)

# Forward pass
input_features = torch.randn(1, 100, 3)      # (B, N, D)
query_coords = torch.randn(1, 1000, 2)       # (B, M, 2) - query any resolution
output = model(input_features, query_coords) # (B, M, 3)
```

## Visualization of LP Placement

Run the visualization:
```bash
python implicit_sequential_bias.py
```

Example output:
```
Placement Strategy: equidistant
Sequence Length: 50, LP Count: 10
Pattern (L=LP, _=Input): L_____L_____L_____L_____L_____L_____L_____L_____L_____L
```

## Key Advantages

### 1. Implicit Sequential Bias
- No explicit positional encodings needed
- Learnable and adaptive to task
- Captures global positional context

### 2. Efficiency
- Mamba's linear complexity: O(L) vs Transformer O(L²)
- Bidirectional processing for richer context
- Efficient for long sequences

### 3. Arbitrary Scale Generation
- LP features condition decoder at any resolution
- Continuous representation via hyponet
- Scale-agnostic architecture

### 4. Flexibility
- Multiple placement strategies
- Configurable number of LPs
- Adaptable to different modalities (images, volumes, etc.)

## Mathematical Formulation

### Input Sequence
```
X = [x₁, x₂, ..., xₙ] ∈ ℝⁿˣᵈ
```

### Learnable Position Tokens
```
L = [l₁, l₂, ..., lₘ] ∈ ℝᵐˣᵈ (learnable parameters)
```

### Interleaving
```
X' = Interleave(X, L, indices) ∈ ℝ⁽ⁿ⁺ᵐ⁾ˣᵈ
```

### Bidirectional Encoding
```
H = BiMamba(X') = (Mamba_forward(X') + Mamba_reverse(X')) / 2
```

### Position Feature Extraction
```
Z = H[:, lp_indices] ∈ ℝᵐˣᵈ
```

### Continuous Decoding
```
f(c) = Hyponet(c, Z)  where c ∈ ℝᵏ (query coordinates)
```

## Experimental Results (from paper)

### 4D fMRI Reconstruction
- **Patch Size**: 4³ × 2
- **Volume Size**: 36³ × 2
- **Configuration**: 81 LPs, equidistant placement
- **Performance**: Competitive PSNR with transformer baselines
- **Efficiency**: Faster inference due to O(L) complexity

### Image Super-Resolution
- Successfully generates arbitrary scale outputs
- LPs provide consistent positional guidance
- Better high-frequency detail preservation

## Implementation Notes

### Dependencies
```bash
pip install torch mamba-ssm einops numpy
```

### Design Principles
1. **Modularity**: Each component (encoder, bias, decoder) is independent
2. **Clarity**: Code is heavily documented for understanding
3. **Flexibility**: Easy to experiment with different configurations
4. **Efficiency**: Optimized for GPU execution

## Citation

If you use this code or the implicit sequential bias mechanism, please cite:
```
MAMBA-GINR: Mamba-based Generalized Implicit Neural Representation
[Paper details to be added]
```

## Future Directions

1. **Adaptive LP Placement**: Learn optimal positions during training
2. **Hierarchical LPs**: Multi-scale position tokens
3. **Cross-Modal LPs**: Shared positional bias across modalities
4. **Efficient Implementations**: Optimized CUDA kernels for LP operations

## Contact

For questions about the MAMBA-GINR innovation, please refer to the main repository documentation.

---

**Key Takeaway**: The implicit sequential bias via learnable position tokens enables MAMBA-GINR to achieve efficient, arbitrary-scale generation while maintaining positional awareness through learned representations rather than fixed encodings.
