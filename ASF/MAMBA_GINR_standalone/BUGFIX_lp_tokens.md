# Bug Fix: LearnablePositionTokens Index Out of Bounds

## Problem

The `LearnablePositionTokens.extract_lp()` method fails with **index out of bounds** error when processing variable-length sequences (masked patches).

### Root Cause

```python
# In __init__: Computes indices for FULL sequence (256 patches + 256 LP = 512 total)
total_len = input_len + num_tokens  # 256 + 256 = 512
self.lp_idxs = torch.linspace(0, total_len - 1, steps=num_tokens).long()  # indices up to 511

# In forward: But actual sequence is SHORTER (128 visible + 256 LP = 384 total)
encoded = self.encoder(tokens_with_lp)  # shape: (B, 384, dim)
lp_features = self.lp_module.extract_lp(encoded)  # tries to index up to 511 → OUT OF BOUNDS!
```

## Solution

**Option 1**: Simplify - Don't interleave LP tokens, just concatenate them at the end.

**Option 2**: Dynamically compute LP indices based on actual sequence length.

### Recommended Fix: Option 1 (Simplest)

Replace the `LearnablePositionTokens` class with a simpler version:

```python
class LearnablePositionTokens(nn.Module):
    """LP tokens - simplified for variable-length sequences"""
    def __init__(self, num_tokens=256, dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim

        # Initialize with sinusoidal embeddings
        init_tokens = get_sinusoidal_embeddings(num_tokens, dim)
        self.tokens = nn.Parameter(init_tokens, requires_grad=True)

    def add_lp(self, x):
        """Add LP tokens to END of sequence"""
        B = x.shape[0]
        lps = repeat(self.tokens, 'n d -> b n d', b=B)
        return torch.cat([x, lps], dim=1)  # Just concatenate, no interleaving

    def extract_lp(self, x):
        """Extract LP tokens from END of sequence"""
        return x[:, -self.num_tokens:]  # Last num_tokens positions
```

This works because:
- LP tokens are **always at the end** regardless of input length
- No complex indexing or permutation needed
- Works with variable-length visible patches

### Trade-off

- **Lost**: Equidistant interleaving (sequential bias from position)
- **Gained**: Simplicity, robustness to variable lengths, no index errors

For masked image modeling, this trade-off is acceptable because:
- The BiMamba encoder has **bidirectional processing** which captures sequential relationships
- Masking creates **irregular patterns** anyway, so strict equidistant placement matters less
- The model can learn positional relationships through the Fourier positional encodings on patches
