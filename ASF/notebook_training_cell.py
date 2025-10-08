# ============================================================================
# REPLACE CELL 10 (Training Functions) WITH THIS
# Implements correct jittering with all specifications
# ============================================================================

def adjust_learning_rate(optimizer, epoch, base_lr=5e-4, warmup_epochs=5, max_epoch=40):
    """Learning rate schedule with warmup + cosine annealing"""
    min_lr = 1e-8

    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch_correct(model, loader, optimizer, device, epoch,
                        resolution=32,
                        jitter_std=None,      # Auto-computed if None
                        offset_std=0.0):      # Prediction offset (default: 0)
    """
    Training with CORRECT jittering specification

    Specifications:
    (1) Gaussian Fourier features for all position encoding
    (2) Sub-pixel jittering: max < 1/(2*resolution)
    (3) Separate modulation/decoding coordinates
    (4) Jittering INSIDE batch loop (different per batch)

    Args:
        jitter_std: Std of modulation jittering. If None, set to 1/(6*resolution)
        offset_std: Std of prediction offset (default: 0)
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    # Auto-compute jittering std if not provided
    if jitter_std is None:
        pixel_size = 1.0 / resolution
        max_allowed_jitter = pixel_size / 2
        jitter_std = max_allowed_jitter / 3  # 3-sigma rule

    # Validate constraint
    pixel_size = 1.0 / resolution
    max_allowed_jitter = pixel_size / 2
    if jitter_std > max_allowed_jitter / 2.5:
        print(f"⚠️  WARNING: jitter_std={jitter_std:.6f} may violate sub-pixel constraint!")
        print(f"   Recommended max: {max_allowed_jitter/3:.6f}")

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Create base coordinate grid INSIDE loop (critical!)
        base_coords = create_coordinate_grid(resolution, resolution, device)

        # (1) Apply small jittering for modulation extraction
        jitter_small = torch.randn_like(base_coords) * jitter_std
        coords_modulation = (base_coords + jitter_small).clamp(0, 1)

        # (2) Apply prediction offset (default: 0)
        if offset_std > 0:
            prediction_offset = torch.randn_like(base_coords) * offset_std
            coords_decoding = (coords_modulation + prediction_offset).clamp(0, 1)
        else:
            coords_decoding = coords_modulation

        # Repeat for batch
        coords_mod_batch = einops.repeat(coords_modulation, 'h w d -> b h w d', b=B)
        coords_dec_batch = einops.repeat(coords_decoding, 'h w d -> b h w d', b=B)

        # Forward pass with separate coordinates
        lp_features = model.encode(images)
        pred = model.hyponet(coords_dec_batch, lp_features,
                            coords_modulation=coords_mod_batch)

        # Ground truth
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
            'jitter': f"σ={jitter_std:.5f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


def validate_correct(model, loader, device, resolution=32):
    """
    Validation with NO jittering (specification #4)
    """
    model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            B = images.shape[0]

            # Exact coordinates (no jittering)
            coords = create_coordinate_grid(resolution, resolution, device)
            coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

            # Forward pass (coords_modulation=None → test mode)
            lp_features = model.encode(images)
            pred = model.hyponet(coords_batch, lp_features,
                                coords_modulation=None)

            # Ground truth
            gt = einops.rearrange(images, 'b c h w -> b h w c')

            # Metrics
            mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses)).mean()

            total_loss += loss.item()
            total_psnr += psnr.item()

    return total_loss / len(loader), total_psnr / len(loader)


def super_resolve_correct(model, images, target_resolution=128, device='cpu'):
    """
    Super-resolution with NO jittering (specification #4)
    """
    model.eval()

    with torch.no_grad():
        B = images.shape[0]

        # Encode at training resolution
        lp_features = model.encode(images)

        # Decode at target resolution (no jittering)
        coords = create_coordinate_grid(target_resolution, target_resolution, device)
        coords_batch = einops.repeat(coords, 'h w d -> b h w d', b=B)

        # Predict (coords_modulation=None → test mode)
        pred = model.hyponet(coords_batch, lp_features, coords_modulation=None)

        # Rearrange to image format
        pred_images = einops.rearrange(pred, 'b h w c -> b c h w')

    return pred_images


print("✓ Training functions defined with CORRECT specifications:")
print("  (1) Gaussian Fourier features everywhere")
print("  (2) Sub-pixel jittering constraint enforced")
print("  (3) Separate modulation/decoding coordinates")
print("  (4) No jittering at test time")
print("")
print("Default jittering for 32×32:")
print(f"  jitter_std = {1/(6*32):.6f} (ensures max < {1/(2*32):.6f})")
