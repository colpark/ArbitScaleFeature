# Debug Spatial Bias Shape Issue

## Expected Flow:

1. **Coordinates**: (B, H, W, 2) = (16, 32, 32, 2)
2. **Flattened**: (B, HW, 2) = (16, 1024, 2)
3. **First batch**: grid = x[0] = (1024, 2)
4. **Patch indices**: indexes = get_patch_index(grid, 16, 16) = (1024,)
   - Each of 1024 pixels maps to one of 256 patches

5. **Spatial bias**: rel_distances = approximate_relative_distances(indexes, 16, 16, 256)
   - Should return: (m, HW) = (256, 1024)
   - After transpose: bias = (HW, 256) = (1024, 256) ❌ WRONG!

## The Real Issue:

Looking at the code:
```python
bias = rel_distances.transpose(1, 0)  # (L, HW)
bias = einops.repeat(bias, 'l n -> b l n', b=B)
```

This assumes `rel_distances` has shape (HW, L) but we're creating (L, HW)!

Then in SharedTokenCrossAttention:
```python
if bias is not None:
    bias = einops.repeat(bias, 'b l n -> b h l n', h=H)  # (B, H, L, HW)
    bias = bias.transpose(-2, -1)  # (B, H, HW, L)
    sim = sim + bias
```

Where `sim` has shape (B, H, HW, L) = (16, 2, 1024, 256)

So bias should be (B, L, HW) = (16, 256, 1024) before the repeat in cross-attention.

## Fix Required:

The issue is in the LAINRDecoder.forward():

### CURRENT (WRONG):
```python
rel_distances = self.approximate_relative_distances(...)  # (m, HW) = (256, 1024)
bias = rel_distances.transpose(1, 0)  # (HW, m) = (1024, 256) ❌
bias = einops.repeat(bias, 'l n -> b l n', b=B)  # (B, HW, m) = (16, 1024, 256) ❌
```

Then in cross-attention it expects (B, L, HW) but gets (B, HW, L)!

### CORRECT:
```python
rel_distances = self.approximate_relative_distances(...)  # (m, HW) = (256, 1024)
# DON'T TRANSPOSE! It's already (L, HW) = (256, 1024)
bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)  # (B, L, HW) = (16, 256, 1024) ✓
```

## The Complete Fix:

In `LAINRDecoder.forward()`, change:

```python
# OLD:
bias = rel_distances.transpose(1, 0)  # (L, HW)
bias = einops.repeat(bias, 'l n -> b l n', b=B)

# NEW:
bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)  # rel_distances is already (L, HW)
```

AND keep the approximate_relative_distances fix to return (m, HW)!
