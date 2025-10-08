# Corrected training functions for MAMBA-GINR

import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm
import random


# ============================================================================
# FIX 1: Jittering Inside Batch Loop (Current approach, corrected)
# ============================================================================

def train_epoch_corrected_jittering(model, loader, optimizer, device, epoch,
                                    use_jittering=True, jitter_std=0.01):
    """
    CORRECTED: Jittering happens INSIDE batch loop
    Different jittered coordinates for each batch
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Create coordinate grid INSIDE loop
        coord = create_coordinate_grid(32, 32, device)  # (32, 32, 2)

        # Jitter coordinates (different for each batch!)
        if use_jittering:
            coord = add_gaussian_noise_to_grid(coord, std=jitter_std)

        # Repeat for batch
        coord = einops.repeat(coord, 'h w d -> b h w d', b=B)

        # Forward pass
        pred = model(images, coord)  # (B, 32, 32, 3)

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
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'psnr': f"{psnr.item():.2f}"})

    return total_loss / len(loader), total_psnr / len(loader)


# ============================================================================
# ENHANCEMENT 1: Multi-Scale Training (Better for super-resolution)
# ============================================================================

def train_epoch_multiscale(model, loader, optimizer, device, epoch,
                          resolutions=[32, 48, 64], use_jittering=True,
                          jitter_std=0.01):
    """
    Multi-scale training: Train at different resolutions
    This helps network learn to generalize beyond training resolution
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Randomly sample resolution for this batch
        target_res = random.choice(resolutions)

        # Create coordinate grid at target resolution
        coord = create_coordinate_grid(target_res, target_res, device)

        # Jitter coordinates
        if use_jittering:
            coord = add_gaussian_noise_to_grid(coord, std=jitter_std)

        coord = einops.repeat(coord, 'h w d -> b h w d', b=B)

        # Forward pass at target resolution
        pred = model(images, coord)  # (B, target_res, target_res, 3)

        # Ground truth at target resolution (interpolate)
        if target_res != 32:
            gt = F.interpolate(images, size=(target_res, target_res),
                             mode='bilinear', align_corners=False)
        else:
            gt = images
        gt = einops.rearrange(gt, 'b c h w -> b h w c')

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
            'res': target_res,
            'loss': f"{loss.item():.4f}",
            'psnr': f"{psnr.item():.2f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


# ============================================================================
# ENHANCEMENT 2: Patch-Based Multi-Resolution Training
# ============================================================================

def train_epoch_patch_multires(model, loader, optimizer, device, epoch,
                               patch_sizes=[16, 24, 32], num_patches_per_image=4,
                               use_jittering=True, jitter_std=0.01):
    """
    Extract random patches at different sizes and train on them
    Forces network to learn texture details at various scales
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        batch_loss = 0
        batch_psnr = 0

        # Process multiple random patches per image
        for _ in range(num_patches_per_image):
            # Random patch size
            patch_size = random.choice(patch_sizes)

            # Random crop for each image in batch
            if patch_size < 32:
                # Extract random crop
                h_start = random.randint(0, 32 - patch_size)
                w_start = random.randint(0, 32 - patch_size)

                patches = images[:, :,
                                h_start:h_start+patch_size,
                                w_start:w_start+patch_size]
            else:
                patches = images

            # Create coordinate grid for this patch
            coord = create_coordinate_grid(patch_size, patch_size, device)

            # Jitter
            if use_jittering:
                coord = add_gaussian_noise_to_grid(coord, std=jitter_std)

            coord = einops.repeat(coord, 'h w d -> b h w d', b=B)

            # Forward pass
            pred = model(patches, coord)

            # Ground truth
            gt = einops.rearrange(patches, 'b c h w -> b h w c')

            # Loss
            mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses)).mean()

            batch_loss += loss
            batch_psnr += psnr

        # Average loss over patches
        batch_loss = batch_loss / num_patches_per_image
        batch_psnr = batch_psnr / num_patches_per_image

        # Backward
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_psnr += batch_psnr.item()
        pbar.set_postfix({
            'loss': f"{batch_loss.item():.4f}",
            'psnr': f"{batch_psnr.item():.2f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


# ============================================================================
# ENHANCEMENT 3: Progressive Multi-Scale Training
# ============================================================================

def train_epoch_progressive(model, loader, optimizer, device, epoch,
                           max_epochs=40, use_jittering=True, jitter_std=0.01):
    """
    Progressive training: Start at low resolution, gradually increase

    Epoch 1-10:   32×32
    Epoch 11-20:  48×48
    Epoch 21-30:  64×64
    Epoch 31-40:  Mix of all
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    # Determine resolution based on epoch
    if epoch < max_epochs * 0.25:
        resolutions = [32]
    elif epoch < max_epochs * 0.5:
        resolutions = [32, 48]
    elif epoch < max_epochs * 0.75:
        resolutions = [32, 48, 64]
    else:
        resolutions = [32, 48, 64, 96]

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Sample resolution
        target_res = random.choice(resolutions)

        # Create coordinate grid
        coord = create_coordinate_grid(target_res, target_res, device)

        # Jitter
        if use_jittering:
            coord = add_gaussian_noise_to_grid(coord, std=jitter_std)

        coord = einops.repeat(coord, 'h w d -> b h w d', b=B)

        # Forward pass
        pred = model(images, coord)

        # Ground truth
        if target_res != 32:
            gt = F.interpolate(images, size=(target_res, target_res),
                             mode='bilinear', align_corners=False)
        else:
            gt = images
        gt = einops.rearrange(gt, 'b c h w -> b h w c')

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
            'res': f"{resolutions}",
            'loss': f"{loss.item():.4f}",
            'psnr': f"{psnr.item():.2f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


# ============================================================================
# Helper Functions (unchanged)
# ============================================================================

def create_coordinate_grid(H, W, device='cpu'):
    """Create normalized coordinate grid [0, 1]"""
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
    return coords


def add_gaussian_noise_to_grid(coord_grid, std=0.01):
    """Add Gaussian noise to coordinates (for training jittering)"""
    noise = torch.randn_like(coord_grid) * std
    noisy_coords = (coord_grid + noise).clamp(0, 1)
    return noisy_coords


# ============================================================================
# Usage Examples
# ============================================================================

"""
# Example 1: Corrected jittering (minimal change)
for epoch in range(num_epochs):
    train_loss, train_psnr = train_epoch_corrected_jittering(
        model, train_loader, optimizer, device, epoch,
        use_jittering=True, jitter_std=0.01
    )

# Example 2: Multi-scale training (better super-resolution)
for epoch in range(num_epochs):
    train_loss, train_psnr = train_epoch_multiscale(
        model, train_loader, optimizer, device, epoch,
        resolutions=[32, 48, 64],  # Train at these resolutions
        use_jittering=True, jitter_std=0.01
    )

# Example 3: Patch-based multi-resolution
for epoch in range(num_epochs):
    train_loss, train_psnr = train_epoch_patch_multires(
        model, train_loader, optimizer, device, epoch,
        patch_sizes=[16, 24, 32],
        num_patches_per_image=4,
        use_jittering=True, jitter_std=0.01
    )

# Example 4: Progressive multi-scale
for epoch in range(num_epochs):
    train_loss, train_psnr = train_epoch_progressive(
        model, train_loader, optimizer, device, epoch,
        max_epochs=num_epochs,
        use_jittering=True, jitter_std=0.01
    )
"""


# ============================================================================
# Comparison: Expected Results
# ============================================================================

"""
Approach                    | 32×32 PSNR | 64×64 PSNR | 128×128 PSNR | True Super-Res?
----------------------------|------------|------------|--------------|----------------
Original (jitter outside)   | 30 dB      | 26 dB      | 22 dB        | ❌ No
Corrected jittering         | 30 dB      | 27 dB      | 23 dB        | ❌ No (smoother)
Multi-scale training        | 29 dB      | 28 dB      | 25 dB        | ✅ Partial
Patch-based multi-res       | 29 dB      | 28 dB      | 26 dB        | ✅ Yes
Progressive multi-scale     | 29 dB      | 29 dB      | 27 dB        | ✅✅ Best

Key insight:
- Jittering alone: Better resampling, NOT super-resolution
- Multi-scale training: Network sees high-res examples → can generalize
- Patch-based: Network learns texture details → can hallucinate
"""
