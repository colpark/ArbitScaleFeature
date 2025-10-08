# Quick Fix for Spatial Bias Bug

## Error:
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (1024) at non-singleton dimension 3
```

## Location:
**Cell 4** in `cifar10_experiments_CORRECTED.ipynb` - in the `LAINRDecoder` class

## What to Replace:

### OLD CODE (Lines ~140-152 in Cell 4):
```python
def approximate_relative_distances(self, target_index, H, W, m):
    """Compute spatial bias based on distance"""
    alpha = self.alpha
    N = H * W
    t = target_index / N
    token_positions = torch.tensor([(i + 0.5) / m for i in range(m)],
                                  device=target_index.device)

    # Distance-based bias: -alpha * distance^2
    rel_distances = -alpha * torch.stack(
        [torch.abs((t - s)**2) for s in token_positions], dim=0
    )
    return rel_distances
```

### NEW CODE (REPLACE WITH THIS):
```python
def approximate_relative_distances(self, target_index, H, W, m):
    """
    Compute spatial bias based on distance

    Args:
        target_index: (HW,) - patch indices for each query pixel
        H, W: patch grid dimensions (e.g., 16×16 for 256 patches)
        m: number of LP tokens (e.g., 256)

    Returns:
        rel_distances: (m, HW) - bias matrix [LP_tokens × pixels]
    """
    alpha = self.alpha
    N = H * W  # Number of patches

    # Normalize patch indices to [0, 1]
    t = target_index.float() / N  # (HW,)

    # LP token positions (evenly distributed in [0, 1])
    token_positions = torch.tensor(
        [(i + 0.5) / m for i in range(m)],
        device=target_index.device
    )  # (m,)

    # Broadcast to create (m, HW) matrix
    t_expanded = t.unsqueeze(0)  # (1, HW)
    tokens_expanded = token_positions.unsqueeze(1)  # (m, 1)

    # Distance-based bias: -alpha * |distance|^2
    # Result shape: (m, HW) = (256, 1024) for full image
    rel_distances = -alpha * torch.abs(t_expanded - tokens_expanded)**2

    return rel_distances
```

## What Changed:

1. **OLD**: Used `torch.stack([...], dim=0)` which created shape `(m,)`
2. **NEW**: Uses broadcasting to create shape `(m, HW)`

### The Fix Explained:
- `t_expanded`: (1, HW) - one row, HW columns
- `tokens_expanded`: (m, 1) - m rows, one column
- Subtraction broadcasts to: **(m, HW)** ✓

This creates bias matrix of shape:
- **(256 LP tokens, 1024 pixels)**
- Which transposes to **(1024, 256)**
- Matching attention scores shape ✓

## Steps to Apply Fix:

1. Open `cifar10_experiments_CORRECTED.ipynb`
2. Go to **Cell 4**
3. Find the `approximate_relative_distances` function (around line 140)
4. Replace it with the NEW CODE above
5. Save and re-run the notebook

The error should be fixed! ✅
