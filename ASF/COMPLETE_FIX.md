# COMPLETE FIX - Two Changes Needed

## Fix #1: In `LAINRDecoder.approximate_relative_distances()`

Replace the entire function with:

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

## Fix #2: In `LAINRDecoder.forward()`

Find this code (around line 168-172):

```python
# OLD (WRONG):
rel_distances = self.approximate_relative_distances(
    indexes, self.patch_num, self.patch_num, tokens.shape[1]
)
bias = rel_distances.transpose(1, 0)  # (L, HW)  ← REMOVE THIS LINE!
bias = einops.repeat(bias, 'l n -> b l n', b=B)
```

Replace with:

```python
# NEW (CORRECT):
rel_distances = self.approximate_relative_distances(
    indexes, self.patch_num, self.patch_num, tokens.shape[1]
)
# rel_distances is already (L, HW) = (256, 1024), don't transpose!
bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)
```

## Why Both Fixes Are Needed:

### Shape Flow:
1. `approximate_relative_distances` returns **(m, HW) = (256, 1024)**
2. This is already in the form **(L, HW)**
3. **DON'T TRANSPOSE** - it's already correct!
4. Repeat to batch: **(B, L, HW) = (16, 256, 1024)**
5. In cross-attention, it becomes **(B, H, HW, L) = (16, 2, 1024, 256)**
6. Matches `sim` shape ✓

## Quick Replace Instructions:

### In Cell 4 of the notebook, make TWO changes:

**Change 1** (around line 148): Replace entire `approximate_relative_distances` function

**Change 2** (around line 170): Delete this line:
```python
bias = rel_distances.transpose(1, 0)  # DELETE THIS!
```

And change the next line from:
```python
bias = einops.repeat(bias, 'l n -> b l n', b=B)
```

To:
```python
bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)
```

That's it! Both fixes together will solve the error.
