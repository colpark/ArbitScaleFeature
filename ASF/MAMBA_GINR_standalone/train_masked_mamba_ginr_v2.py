"""
Training script for Masked MAMBA-GINR with proper architecture

Two-stage training:
1. Stage 1: Masked reconstruction pretraining
2. Stage 2: Classification with CNN on pixel-level modulation features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from masked_mamba_ginr_v2 import MaskedMAMBAGINR, CNNClassifier, device


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Model
    'img_size': 32,
    'patch_size': 2,
    'dim': 256,
    'num_lp': 256,
    'mamba_depth': 6,
    'ff_dim': 1024,
    'hidden_dim': 512,
    'n_features': 32,
    'mask_ratio': 0.5,  # 50% masking

    # Training - Stage 1 (Reconstruction)
    'stage1_epochs': 100,
    'stage1_lr': 5e-4,
    'stage1_batch_size': 64,
    'stage1_weight_decay': 1e-4,

    # Training - Stage 2 (Classification)
    'stage2_epochs': 100,
    'stage2_lr': 1e-3,
    'stage2_batch_size': 256,
    'stage2_weight_decay': 1e-4,

    # Other
    'num_workers': 4,
    'save_dir': './checkpoints',
}


# ============================================================================
# Data Loading
# ============================================================================

def get_dataloaders(batch_size, num_workers=4):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset


# ============================================================================
# Stage 1: Masked Reconstruction Pretraining
# ============================================================================

def train_reconstruction_epoch(model, loader, optimizer, device):
    """Train one epoch of masked reconstruction"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    for images, _ in pbar:
        images = images.to(device)
        B = images.shape[0]

        # Forward (with masking)
        loss, pred_rgb, mask = model(images, return_modulation=False)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_samples


def validate_reconstruction(model, loader, device):
    """Validate reconstruction performance"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            B = images.shape[0]

            loss, pred_rgb, mask = model(images, return_modulation=False)

            total_loss += loss.item() * B
            total_samples += B

    return total_loss / total_samples


def stage1_pretrain_reconstruction(model, train_loader, test_loader, config):
    """Stage 1: Pretrain with masked reconstruction"""
    print("\n" + "="*70)
    print("STAGE 1: MASKED RECONSTRUCTION PRETRAINING")
    print("="*70)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['stage1_lr'],
        weight_decay=config['stage1_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['stage1_epochs'], eta_min=1e-6
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config['stage1_epochs']):
        print(f"\nEpoch {epoch+1}/{config['stage1_epochs']}")

        train_loss = train_reconstruction_epoch(model, train_loader, optimizer, device)
        val_loss = validate_reconstruction(model, test_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config['save_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config['save_dir'], 'masked_mamba_ginr_pretrain_best.pth'))
            print(f"  → Best model saved (val_loss: {best_val_loss:.6f})")

        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_reconstruction(model, test_loader, device, epoch+1, config)

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Stage 1: Masked Reconstruction Pretraining')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config['save_dir'], 'stage1_training_curves.png'), dpi=150)
    plt.close()

    print(f"\n✓ Stage 1 complete! Best val loss: {best_val_loss:.6f}")

    return best_val_loss


def visualize_reconstruction(model, loader, device, epoch, config):
    """Visualize reconstruction results"""
    model.eval()

    # Get sample batch
    images, _ = next(iter(loader))
    images = images[:16].to(device)

    with torch.no_grad():
        loss, pred_rgb, mask = model(images, return_modulation=False)

    # Convert mask to pixel mask for visualization
    B = images.shape[0]
    p = model.patch_size
    patch_mask_spatial = mask.reshape(B, model.patch_num, model.patch_num)
    from einops import repeat
    pixel_mask = repeat(patch_mask_spatial, 'b h w -> b c (h p1) (w p2)', c=3, p1=p, p2=p)

    # Create masked images
    masked_images = images * pixel_mask

    # Visualize
    fig, axes = plt.subplots(3, 16, figsize=(20, 4))

    for i in range(16):
        # Original
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10, fontweight='bold', rotation=0, labelpad=30)

        # Masked (50% visible)
        axes[1, i].imshow(masked_images[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Masked (50%)', fontsize=10, fontweight='bold', rotation=0, labelpad=30)

        # Reconstructed
        axes[2, i].imshow(pred_rgb[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Reconstructed', fontsize=10, fontweight='bold', rotation=0, labelpad=30)

    plt.suptitle(f'Masked Reconstruction - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], f'reconstruction_epoch_{epoch}.png'), dpi=150)
    plt.close()


# ============================================================================
# Stage 2: Classification with Spatial Modulation Features
# ============================================================================

def extract_modulation_features(model, loader, device):
    """Extract pixel-level modulation features for all images"""
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting modulation features"):
            images = images.to(device)

            # Extract modulation features: (B, H, W, hidden_dim)
            modulation = model(images, return_modulation=True)

            all_features.append(modulation.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def train_classifier_epoch(classifier, loader, optimizer, device):
    """Train CNN classifier on modulation features"""
    classifier.train()
    correct = 0
    total = 0
    total_loss = 0.0

    for features, labels in tqdm(loader, desc="Training classifier"):
        features = features.to(device)
        labels = labels.to(device)

        # Forward
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


def validate_classifier(classifier, loader, device):
    """Validate CNN classifier"""
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Validating classifier"):
            features = features.to(device)
            labels = labels.to(device)

            logits = classifier(features)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def stage2_train_classifier(model, train_loader, test_loader, config):
    """Stage 2: Train CNN classifier on frozen modulation features"""
    print("\n" + "="*70)
    print("STAGE 2: CNN CLASSIFICATION ON MODULATION FEATURES")
    print("="*70)

    # Load best pretrained model
    checkpoint = torch.load(os.path.join(config['save_dir'], 'masked_mamba_ginr_pretrain_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded pretrained model from epoch {checkpoint['epoch']+1}")

    # Extract modulation features (freeze encoder)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("\nExtracting modulation features...")
    train_features, train_labels = extract_modulation_features(model, train_loader, device)
    test_features, test_labels = extract_modulation_features(model, test_loader, device)

    print(f"Train features: {train_features.shape}")  # (50000, 32, 32, 512)
    print(f"Test features: {test_features.shape}")    # (10000, 32, 32, 512)

    # Create dataloaders for features
    train_dataset_cls = TensorDataset(train_features, train_labels)
    test_dataset_cls = TensorDataset(test_features, test_labels)

    train_loader_cls = DataLoader(
        train_dataset_cls, batch_size=config['stage2_batch_size'],
        shuffle=True, num_workers=0  # Features already in memory
    )
    test_loader_cls = DataLoader(
        test_dataset_cls, batch_size=config['stage2_batch_size'],
        shuffle=False, num_workers=0
    )

    # Initialize CNN classifier
    classifier = CNNClassifier(
        hidden_dim=config['hidden_dim'],
        num_classes=10
    ).to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config['stage2_lr'],
        weight_decay=config['stage2_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['stage2_epochs'], eta_min=1e-6
    )

    best_acc = 0.0
    train_accs = []
    test_accs = []

    print(f"\nTraining CNN classifier for {config['stage2_epochs']} epochs...")

    for epoch in range(config['stage2_epochs']):
        train_acc, train_loss = train_classifier_epoch(classifier, train_loader_cls, optimizer, device)
        test_acc = validate_classifier(classifier, test_loader_cls, device)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc:.2f}%, Test={test_acc:.2f}% (Best={best_acc:.2f}%)")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': classifier.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(config['save_dir'], 'cnn_classifier_best.pth'))

    # Plot classification results
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.axhline(y=best_acc, color='r', linestyle='--', alpha=0.3, label=f'Best: {best_acc:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Stage 2: CNN Classification on Modulation Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    plt.savefig(os.path.join(config['save_dir'], 'stage2_classification_results.png'), dpi=150)
    plt.close()

    print(f"\n✓ Stage 2 complete! Best test accuracy: {best_acc:.2f}%")

    return best_acc


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("="*70)
    print("MASKED MAMBA-GINR TRAINING (Proper Architecture)")
    print("="*70)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        batch_size=CONFIG['stage1_batch_size'],
        num_workers=CONFIG['num_workers']
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    print("\nInitializing Masked MAMBA-GINR model...")
    model = MaskedMAMBAGINR(
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        in_channels=3,
        dim=CONFIG['dim'],
        num_lp=CONFIG['num_lp'],
        mamba_depth=CONFIG['mamba_depth'],
        ff_dim=CONFIG['ff_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        n_features=CONFIG['n_features'],
        mask_ratio=CONFIG['mask_ratio']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Stage 1: Masked reconstruction pretraining
    best_recon_loss = stage1_pretrain_reconstruction(
        model, train_loader, test_loader, CONFIG
    )

    # Reload data with larger batch size for classification
    train_loader_cls, test_loader_cls, _, _ = get_dataloaders(
        batch_size=CONFIG['stage2_batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # Stage 2: Classification with spatial modulation features
    best_clf_acc = stage2_train_classifier(
        model, train_loader_cls, test_loader_cls, CONFIG
    )

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Stage 1 (Reconstruction): Best val loss = {best_recon_loss:.6f}")
    print(f"Stage 2 (Classification): Best test accuracy = {best_clf_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
