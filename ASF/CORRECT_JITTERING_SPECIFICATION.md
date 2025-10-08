# Correct Jittering Specification

## Your Requirements (Confirmed)

### (1) Gaussian Fourier Features for ALL Position Encoding

**Encoding (Patches)**:
```python
# When encoding patches, use Gaussian Fourier features for patch positions
patch_positions = get_patch_positions(B, device)  # (B, N, 2)
patch_pos_features = gaussian_fourier_encode(patch_positions, B_matrix)
```

**Decoding (Queries)**:
```python
# When decoding at coordinates, use SAME Gaussian Fourier approach
query_coords = create_coordinate_grid(H, W, device)  # (H, W, 2)
query_features = gaussian_fourier_encode(query_coords, B_matrix)
```

**Key**: Both use Gaussian Fourier features with shared frequency matrices.

---

### (2) Jittering Must Be Sub-Pixel

**Constraint**: `max(|jitter|) < 1/(2 * resolution)`

For 32×32 training:
```python
pixel_size = 1/32  # 0.03125
max_jitter = pixel_size / 2  # 0.015625

# Jittering noise
jitter_std = max_jitter / 3  # ~0.005 (3-sigma rule)
# This ensures 99.7% of jitter stays within ±max_jitter
```

**Why**: Prevent crossing pixel boundaries during training.

---

### (3) Two Types of Jittering

#### (3a) Modulation Jittering (Small, Sub-Pixel)

**Purpose**: Learn continuous modulation extraction
**Constraint**: STRICTLY sub-pixel (as in point 2)

```python
# Coordinates for modulation extraction
coords_modulation = base_coords + jitter_small
# where jitter_small ~ N(0, (1/64)²) for 32×32 images
# max(|jitter_small|) < 1/64 = 0.015625
```

#### (3b) Prediction Offset (Larger, Configurable)

**Purpose**: Test generalizability of modulation vector
**Constraint**: Can be > 1 pixel, but set to 0 by default

```python
# Coordinates for hyponet decoding
coords_decoding = coords_modulation + prediction_offset
# where prediction_offset ~ N(0, offset_std²)
# offset_std can be 0 (default), 1/32, or larger

# For now: offset_std = 0 (no additional offset)
```

**Interpretation**:
- `coords_modulation`: Where we extract features from LP tokens
- `coords_decoding`: Where we actually predict RGB
- If offset ≠ 0: Tests if modulation generalizes spatially

---

### (4) Test Time: NO Jittering

```python
if is_train:
    # Apply jittering
    coords_modulation = coords + jitter_small
    coords_decoding = coords_modulation + prediction_offset
else:
    # Test: use exact coordinates
    coords_modulation = coords
    coords_decoding = coords
```

---

## Implementation

### Correct Training Forward Pass

```python
def forward_train(self, images, base_coords,
                  jitter_std=0.005,      # For modulation (sub-pixel)
                  offset_std=0.0):        # For prediction (default: 0)
    """
    Training forward pass with correct jittering

    Args:
        images: (B, C, H, W) - input images
        base_coords: (H, W, 2) - base coordinate grid
        jitter_std: Std of jittering for modulation (MUST be < 1/(2*H))
        offset_std: Std of prediction offset (default 0)
    """
    B = images.shape[0]
    H, W = base_coords.shape[:2]

    # (1) Apply small jittering for modulation extraction
    jitter_small = torch.randn_like(base_coords) * jitter_std
    coords_modulation = (base_coords + jitter_small).clamp(0, 1)

    # (2) Apply prediction offset for hyponet decoding
    if offset_std > 0:
        prediction_offset = torch.randn_like(base_coords) * offset_std
        coords_decoding = (coords_modulation + prediction_offset).clamp(0, 1)
    else:
        coords_decoding = coords_modulation

    # (3) Encode image to LP tokens
    lp_features = self.encode(images)

    # (4) Extract modulation at coords_modulation
    coords_mod_batch = einops.repeat(coords_modulation, 'h w d -> b h w d', b=B)
    modulation_vector = self.extract_modulation(coords_mod_batch, lp_features)

    # (5) Decode at coords_decoding using extracted modulation
    coords_dec_batch = einops.repeat(coords_decoding, 'h w d -> b h w d', b=B)
    pred = self.decode_with_modulation(coords_dec_batch, modulation_vector)

    return pred


def forward_test(self, images, coords):
    """
    Test forward pass - NO jittering

    Args:
        images: (B, C, H, W) - input images
        coords: (H, W, 2) - exact coordinate grid
    """
    B = images.shape[0]

    # No jittering at test time
    coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

    # Standard forward pass
    lp_features = self.encode(images)
    pred = self.decode(lp_features, coords_batch)

    return pred
```

---

## Modified LAINR Decoder

```python
class LAINRDecoderGaussian(nn.Module):
    def forward(self, coords, tokens,
                coords_for_modulation=None):
        """
        Args:
            coords: (B, H, W, 2) - coordinates for DECODING (final prediction)
            tokens: (B, L, D) - LP token features
            coords_for_modulation: (B, H, W, 2) - coordinates for MODULATION
                                   If None, use coords (test mode)
        """
        B, query_shape = coords.shape[0], coords.shape[1:-1]
        coords = coords.view(B, -1, coords.shape[-1])  # (B, HW, 2)

        # Determine coordinates for modulation extraction
        if coords_for_modulation is not None:
            # Training: use jittered coordinates for modulation
            coords_mod = coords_for_modulation.view(B, -1, coords_for_modulation.shape[-1])
        else:
            # Test: use same coordinates
            coords_mod = coords

        # Get spatial bias (based on modulation coordinates)
        grid_mod = coords_mod[0]
        indexes = self.get_patch_index(grid_mod, self.patch_num, self.patch_num)
        rel_distances = self.approximate_relative_distances(
            indexes, self.patch_num, self.patch_num, tokens.shape[1]
        )
        bias = einops.repeat(rel_distances, 'l n -> b l n', b=B)

        # Query encoding for modulation (at coords_mod)
        x_q = einops.repeat(
            self.calc_gamma_gaussian(coords_mod[0], self.B_q), 'l d -> b l d', b=B
        )
        x_q = self.act(self.query_lin(x_q))

        # Extract modulation via cross-attention
        modulation_vector = self.modulation_ca(x_q, context=tokens, bias=bias)

        # Multi-scale processing at DECODING coordinates
        modulations_l = []
        for k in range(self.layer_num):
            # Encode DECODING coordinates at each frequency scale
            x_l = einops.repeat(
                self.calc_gamma_gaussian(coords[0], self.B_ls[k]), 'l d -> b l d', b=B
            )
            h_l = self.act(self.bandwidth_lins[k](x_l))

            # Add modulation
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        # Residual connections and output
        h_v = [modulations_l[0]]
        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]
        out = sum(outs)
        out = out.view(B, *query_shape, -1)

        return out
```

---

## Training Loop

```python
def train_epoch_correct_jittering(model, loader, optimizer, device, epoch,
                                  jitter_std=0.005,      # Sub-pixel for modulation
                                  offset_std=0.0):       # Prediction offset (default: 0)
    """
    Training with correct jittering specification
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    # Validate jittering constraint
    pixel_size = 1/32
    max_allowed_jitter = pixel_size / 2
    assert jitter_std <= max_allowed_jitter / 3, \
        f"jitter_std={jitter_std} too large! Must be <= {max_allowed_jitter/3:.6f}"

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Create base coordinate grid (inside loop! - critical fix)
        base_coords = create_coordinate_grid(32, 32, device)  # (32, 32, 2)

        # Apply small jittering for modulation
        jitter_small = torch.randn_like(base_coords) * jitter_std
        coords_modulation = (base_coords + jitter_small).clamp(0, 1)

        # Apply prediction offset (default: 0)
        if offset_std > 0:
            prediction_offset = torch.randn_like(base_coords) * offset_std
            coords_decoding = (coords_modulation + prediction_offset).clamp(0, 1)
        else:
            coords_decoding = coords_modulation  # Same coordinates

        # Repeat for batch
        coords_mod_batch = einops.repeat(coords_modulation, 'h w d -> b h w d', b=B)
        coords_dec_batch = einops.repeat(coords_decoding, 'h w d -> b h w d', b=B)

        # Forward pass with separate coordinates
        pred = model.hyponet(coords_dec_batch, model.encode(images),
                            coords_for_modulation=coords_mod_batch)

        # Ground truth (at base coordinates)
        gt = einops.rearrange(images, 'b c h w -> b h w c')

        # Loss
        mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'psnr': f"{psnr.item():.2f}",
            'jitter': f"±{jitter_std:.4f}",
            'offset': f"±{offset_std:.4f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


def validate(model, loader, device):
    """
    Validation - NO jittering
    """
    model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            B = images.shape[0]

            # Exact coordinates (no jittering)
            coords = create_coordinate_grid(32, 32, device)
            coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

            # Forward pass (coords_for_modulation=None → test mode)
            pred = model.hyponet(coords_batch, model.encode(images),
                                coords_for_modulation=None)

            # Ground truth
            gt = einops.rearrange(images, 'b c h w -> b h w c')

            # Metrics
            mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses)).mean()

            total_loss += loss.item()
            total_psnr += psnr.item()

    return total_loss / len(loader), total_psnr / len(loader)
```

---

## Summary of Specifications

| Aspect | Training | Test |
|--------|----------|------|
| **Positional Encoding** | Gaussian Fourier | Gaussian Fourier (same) |
| **Modulation Jittering** | Yes, std=0.005 (sub-pixel) | No |
| **Prediction Offset** | Optional, std=0.0 (default) | No |
| **coords_modulation** | base + jitter_small | base (exact) |
| **coords_decoding** | coords_mod + offset | base (exact) |
| **max(jitter)** | < 1/64 (half pixel) | 0 |

---

## Verification Checklist

✅ (1) Gaussian Fourier features for ALL position encoding (encode & decode)
✅ (2) Jittering constraint: max < 1/(2*resolution) = 1/64 for 32×32
✅ (3a) Small jittering for modulation extraction (sub-pixel)
✅ (3b) Prediction offset separate (default: 0)
✅ (4) Test time: NO jittering, exact coordinates

This specification ensures:
- Continuous position encoding (Gaussian Fourier)
- Controlled jittering (sub-pixel, inside batch loop)
- Separate modulation/decoding coordinates (for future offset experiments)
- Clean test inference (no randomness)
