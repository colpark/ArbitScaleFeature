# LAINR Decoder Spatial Bias Fix

## Problem

The error occurs because:
- **Patch grid**: 16×16 = 256 patches (for 2×2 patch size on 32×32 image)
- **Pixel grid**: 32×32 = 1024 pixels
- **Spatial bias**: Computed for 256 LP tokens × 1024 pixels = (256, 1024)
- **Error**: Trying to add bias of shape (B, H, 1024, 256) to sim of shape (B, H, 1024, 256) ✓

Actually wait, the shapes should match! Let me check...

The error says:
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (1024) at non-singleton dimension 3
```

This means `sim` has shape (B, H, HW, L) where L=256 (num LP tokens)
But `bias` after transpose has shape (B, H, HW, ??) where ?? should be 256 but is 1024

## Root Cause

In `approximate_relative_distances`:
```python
rel_distances = -alpha * torch.stack(
    [torch.abs((t - s)**2) for s in token_positions], dim=0
)
```

This creates shape (m, ) where m = num_lp_tokens = 256

Then:
```python
bias = rel_distances.transpose(1, 0)  # (L, HW) = (256, HW)
```

Wait, `rel_distances` is (m,) not (m, HW)!

The issue is that `rel_distances` should have shape (m, len(target_index)) but the stack creates (m,).

## Solution

Fix the `approximate_relative_distances` to broadcast correctly:

```python
def approximate_relative_distances(self, target_index, H, W, m):
    """
    Compute spatial bias based on distance

    Args:
        target_index: (HW,) - patch indices for each pixel
        H, W: patch grid dimensions
        m: number of LP tokens

    Returns:
        rel_distances: (m, HW) - bias for each LP token to each pixel
    """
    alpha = self.alpha
    N = H * W  # Number of patches
    t = target_index.float() / N  # Normalize patch indices to [0, 1]

    # LP token positions (evenly distributed)
    token_positions = torch.tensor(
        [(i + 0.5) / m for i in range(m)],
        device=target_index.device
    )  # (m,)

    # Broadcast: t is (HW,), token_positions is (m,)
    # We want (m, HW)
    t_expanded = t.unsqueeze(0)  # (1, HW)
    tokens_expanded = token_positions.unsqueeze(1)  # (m, 1)

    # Distance-based bias: -alpha * distance^2
    rel_distances = -alpha * torch.abs(t_expanded - tokens_expanded)**2  # (m, HW)

    return rel_distances
```

The key change is to properly broadcast to create (m, HW) instead of (m,).
