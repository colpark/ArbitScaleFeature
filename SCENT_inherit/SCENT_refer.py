from torch.optim import AdamW
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange, repeat
import ssl
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from math import pi, log
from functools import wraps
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
# Fix for torchvision dataset download issue
ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================================
# --- 1. The One True Perceiver IO Model Architecture ---
# ===============================================================
D = 256
# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.latest_attn = None

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim = -1)
        self.latest_attn = attn.detach()
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

from math import log

# This helper function creates the sinusoidal embeddings
def get_sinusoidal_embeddings(n, d):
    """
    Generates sinusoidal positional embeddings.
    
    Args:
        n (int): The number of positions (num_latents).
        d (int): The embedding dimension (latent_dim).

    Returns:
        torch.Tensor: A tensor of shape (n, d) with sinusoidal embeddings.
    """
    # Ensure latent_dim is even for sin/cos pairs
    assert d % 2 == 0, "latent_dim must be an even number for sinusoidal embeddings"
    
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))
    
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def add_white_noise(coords, scale=0.01):
    return coords + torch.randn_like(coords) * scale




class CascadedBlock(nn.Module):
    def __init__(self, dim, n_latents, input_dim, cross_heads, cross_dim_head, self_heads, self_dim_head, residual_dim=None):
        super().__init__()
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim), requires_grad=False)
        # self.latents = nn.Parameter(torch.randn(n_latents, dim))
        self.cross_attn = PreNorm(dim, Attention(dim, input_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=input_dim)
        self.self_attn = PreNorm(dim, Attention(dim, heads=self_heads, dim_head=self_dim_head))
        self.residual_proj = nn.Linear(residual_dim, dim) if residual_dim and residual_dim != dim else None
        self.ff = PreNorm(dim, FeedForward(dim))

    def forward(self, x, context, mask=None, residual=None):
        b = context.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        latents = self.cross_attn(latents, context=context, mask=mask) + latents
        if residual is not None:
            if self.residual_proj:
                residual = self.residual_proj(residual)
            latents = latents + residual
        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


class CascadedPerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        queries_dim,
        logits_dim = None,
        latent_dims=(512, 512, 512),
        num_latents=(256, 256, 256),
        cross_heads = 4,
        cross_dim_head = 128,
        self_heads = 8,
        self_dim_head = 128,
        decoder_ff = False,
        
    ):
        super().__init__()
        
        assert len(latent_dims) == len(num_latents), "latent_dims and num_latents must have same length"
        
    
        # self.input_proj = nn.Linear(4, 128)
        self.input_proj = nn.Sequential(
                nn.Linear(4, 128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        self.projection_matrix = nn.Parameter(torch.randn(4, 128) / np.sqrt(4)).to(DEVICE)
        # proj = torch.randn(4, 128) / np.sqrt(4)
        # self.projection_matrix = nn.Parameter(proj.detach())  # make it a leaf tenso

        # Cascaded encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_dim = None
        for dim, n_latents in zip(latent_dims, num_latents):
            block = CascadedBlock(
                dim=dim,
                n_latents=n_latents,
                input_dim=input_dim,
                cross_heads=cross_heads,
                cross_dim_head=cross_dim_head,
                self_heads=self_heads,
                self_dim_head=self_dim_head,
                residual_dim=prev_dim
            )
            self.encoder_blocks.append(block)
            prev_dim = dim

        # Decoder
        final_latent_dim = latent_dims[-1]
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=final_latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        

        # self.decoder_swin = SwinTransformerLayer(
        #     dim=queries_dim,
        #     depth=2,                  # or 4 if you want deeper decoding
        #     num_heads=4,
        #     window_size=16,           # assuming 64x64 → 4096 tokens → 256 windows of size 16
        #     mlp_ratio=4.0,
        #     drop_path=0.1,
        #     use_checkpoint=False
        # )
        self.self_attn_blocks = nn.Sequential(*[
        nn.Sequential(
            PreNorm(latent_dims[-1], Attention(latent_dims[-1], heads=self_heads, dim_head=self_dim_head)),
            PreNorm(latent_dims[-1], FeedForward(latent_dims[-1]))
        )
        for _ in range(4)  # or 3
    ])

    def forward(self, data, mask=None, queries=None):
        b = data.size(0)
        residual = None

        
        for block in self.encoder_blocks:
            residual = block(x=residual, context=data, mask=mask, residual=residual)

            
            
            
        for sa_block in self.self_attn_blocks:
            residual = sa_block[0](residual) + residual
            residual = sa_block[1](residual) + residual
        
        if  b == 1:  # Optional: only log for one sample
            latent_std = residual.std(dim=1).mean().item()
            print(f"[Latent std]: {latent_std:.4f}")
        
        if queries is None:
            return latents

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        x = self.decoder_cross_attn(queries, context=residual)

        # Optional: skip connection to preserve input query encoding
        x = x + queries

        # Local refinement (like SCENT)
        # x = self.decoder_swin(x)

        # Final FF
        if self.decoder_ff:
            x = x + self.decoder_ff(x)

        return self.to_logits(x)


    
    
# class GaussianFourierFeatures(nn.Module):
#     def __init__(self, in_features, mapping_size, scale=50.0):
#         super().__init__()
#         self.in_features = in_features
#         self.mapping_size = mapping_size
#         self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

#     def forward(self, coords):
#         projections = coords @ self.B
#         fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
#         return fourier_feats
    
    
    
# # For (x, t) -> used in queries (2D input)
# pos_encoder_2d = GaussianFourierFeatures(in_features=2, mapping_size=128).to(DEVICE)

# # For (x, t_in, t_out) -> used in model input (3D input)
# pos_encoder_3d = GaussianFourierFeatures(in_features=3, mapping_size=128).to(DEVICE)

# # For (x, y, t_in, t_out)
# pos_enc_4d = GaussianFourierFeatures(in_features=4, mapping_size=128).to(DEVICE)

# # For (x, y, t_out)
# pos_enc_query = GaussianFourierFeatures(in_features=3, mapping_size=128).to(DEVICE)




# pos_encoder_1d_x = GaussianFourierFeatures(1, 32).to(DEVICE)
# pos_encoder_1d_y = GaussianFourierFeatures(1, 32).to(DEVICE)
# pos_encoder_1d_t0 = GaussianFourierFeatures(1, 32).to(DEVICE)
# pos_encoder_1d_t1 = GaussianFourierFeatures(1, 32).to(DEVICE)

# ===============================================================
# --- Training Script Starts Here ---
# ===============================================================


# --- Configuration and Setup ---
BATCH_SIZE = 64
EPOCHS = 50
# Adjusted learning rate to a more standard value for this kind of task
LEARNING_RATE = 2e-4
# Using an available GPU, change if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE_TRAIN = 32
IMAGE_SIZE_HI_RES = 128
CHANNELS = 3

POS_EMBED_DIM = 64
INPUT_DIM = CHANNELS + POS_EMBED_DIM
QUERIES_DIM = POS_EMBED_DIM
LOGITS_DIM = CHANNELS

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

    def forward(self, coords):
        projections = coords @ self.B
        fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return fourier_feats

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, n_bands=16):
        super().__init__()
        self.in_features = in_features
        fourier_dim = in_features * 2 * n_bands
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )
        self.register_buffer('freqs', 2**torch.arange(n_bands) * torch.pi)

    def forward(self, coords):
        b, n, d = coords.shape
        projections = coords.unsqueeze(-1) * self.freqs
        fourier_feats_list = [torch.sin(projections), torch.cos(projections)]
        fourier_feats = torch.cat(fourier_feats_list, dim=-1)
        fourier_feats = rearrange(fourier_feats, 'b n d bands -> b n (d bands)')
        return self.mlp(fourier_feats)
    
# --- In your main script ---
FOURIER_MAPPING_SIZE = 96
POS_EMBED_DIM = FOURIER_MAPPING_SIZE * 2
INPUT_DIM = CHANNELS + POS_EMBED_DIM
QUERIES_DIM = POS_EMBED_DIM

fourier_encoder = GaussianFourierFeatures(
    in_features=2,
    mapping_size=FOURIER_MAPPING_SIZE,
    scale=15.0
).to(DEVICE)

# fourier_encoder = FourierFeatures(in_features=2, out_features=POS_EMBED_DIM).to(DEVICE)


model = CascadedPerceiverIO(
    input_dim=INPUT_DIM,
    queries_dim=QUERIES_DIM,
    logits_dim=LOGITS_DIM,
    latent_dims=(256, 384, 512),
    num_latents=(256, 256, 256),
    decoder_ff=True
).to(DEVICE)


optimizer = AdamW(list(model.parameters()) + list(fourier_encoder.parameters()), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

print(f"Training on {DEVICE}")
total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in fourier_encoder.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")


# --- Helper Functions and Coordinate Grids ---
def create_coordinate_grid(h, w, device):
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing='ij'
    ), dim=-1)
    return rearrange(grid, 'h w c -> (h w) c')





coords_32x32 = create_coordinate_grid(IMAGE_SIZE_TRAIN, IMAGE_SIZE_TRAIN, DEVICE)
coords_128x128 = create_coordinate_grid(IMAGE_SIZE_HI_RES, IMAGE_SIZE_HI_RES, DEVICE)


# coords_base_noisy = add_white_noise(coords_32x32, scale=0.01)
# coords_highres_noisy = add_white_noise(coords_128x128, scale=0.01)

# fixed_query_32 = fourier_encoder(coords_base_noisy).detach()
# fixed_query_128 = fourier_encoder(coords_highres_noisy).detach()

def prepare_model_input(images, coords, fourier_encoder_fn):
    b, c, h, w = images.shape
    pixels = rearrange(images, 'b c h w -> b (h w) c')
    batch_coords = repeat(coords, 'n d -> b n d', b=b)
    pos_embeddings = fourier_encoder_fn(batch_coords)
    input_with_pos = torch.cat((pixels, pos_embeddings), dim=-1)
    return input_with_pos, pixels, pos_embeddings

def imshow(img, title):
    img = img.cpu() / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

# ===============================================================
# --- NEW: FFT Validation Function ---
# ===============================================================
def calculate_and_visualize_fft_power_delta(original_imgs, recon_imgs, epoch_num):
    """
    Calculates and visualizes the FFT power spectrum difference between
    the original and reconstructed images.
    """
    # Use the first image in the batch for visualization
    original_img = original_imgs[0]
    recon_img = recon_imgs[0]

    # Convert to grayscale for 2D FFT
    # New versions of torchvision use functional transforms
    original_gray = transforms.functional.rgb_to_grayscale(original_img)
    recon_gray = transforms.functional.rgb_to_grayscale(recon_img)

    # --- FFT Calculation ---
    def get_log_power_spectrum(img_tensor):
        # Squeeze the channel dimension
        img_tensor = img_tensor.squeeze(0)
        # Apply 2D FFT
        fft = torch.fft.fft2(img_tensor)
        # Shift the zero frequency component to the center
        fft_shifted = torch.fft.fftshift(fft)
        # Calculate the power spectrum (magnitude squared)
        power_spectrum = torch.abs(fft_shifted)**2
        # Use log scale for better visualization
        log_power_spectrum = torch.log1p(power_spectrum)
        return log_power_spectrum.cpu().numpy()

    original_fft_power = get_log_power_spectrum(original_gray)
    recon_fft_power = get_log_power_spectrum(recon_gray)
    delta_power = np.abs(original_fft_power - recon_fft_power)

    # --- Visualization ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Epoch {epoch_num}: FFT Power Spectrum Comparison', fontsize=16)

    im1 = axs[0].imshow(original_fft_power, cmap='viridis')
    axs[0].set_title('Original Image FFT Power')
    axs[0].axis('off')
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(recon_fft_power, cmap='viridis')
    axs[1].set_title('Reconstructed Image FFT Power')
    axs[1].axis('off')
    fig.colorbar(im2, ax=axs[1])
    
    im3 = axs[2].imshow(delta_power, cmap='magma')
    axs[2].set_title('Power Difference (Delta)')
    axs[2].axis('off')
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout()
    plt.show()

def compare_fft_upsampled_vs_generated(recon_32_imgs, gen_128_imgs, epoch_num):
    """
    Compare the FFT power spectrum between upsampled 32x32 reconstructions
    and generated 128x128 outputs to detect high-frequency learning.

    Args:
        recon_32_imgs (Tensor): (B, C, 32, 32) reconstructed images from the model.
        gen_128_imgs  (Tensor): (B, C, 128, 128) generated hi-res images from the model.
        epoch_num     (int):    Epoch number for the plot title.
    """
    # Use first sample for visualization
    recon_32 = recon_32_imgs[0]    # (C, 32, 32)
    gen_128  = gen_128_imgs[0]     # (C, 128, 128)

    # Convert to grayscale (-> 1 x H x W)
    recon_32_gray = transforms.functional.rgb_to_grayscale(recon_32)
    gen_128_gray  = transforms.functional.rgb_to_grayscale(gen_128)

    # Upsample 32x32 -> 128x128 to align frequency resolution
    upsampled_recon_gray = F.interpolate(
        recon_32_gray.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
    ).squeeze(0)  # (1, 128, 128)

    def get_log_power_spectrum(img_tensor):
        # img_tensor: (1, H, W); grayscale; can be on CUDA
        img_tensor = img_tensor.squeeze(0)  # -> (H, W)
        fft = torch.fft.fft2(img_tensor)
        fft_shifted = torch.fft.fftshift(fft)
        power = torch.abs(fft_shifted) ** 2
        log_power = torch.log1p(power)
        return log_power.detach().cpu().numpy()

    upsampled_fft = get_log_power_spectrum(upsampled_recon_gray)
    generated_fft = get_log_power_spectrum(gen_128_gray)
    delta_power   = np.abs(generated_fft - upsampled_fft)

    # --- Visualization ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Epoch {epoch_num}: FFT (Upsampled 32x32 vs Generated 128x128)', fontsize=16)

    im0 = axs[0].imshow(upsampled_fft, cmap='viridis')
    axs[0].set_title('Upsampled 32x32 FFT Power')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(generated_fft, cmap='viridis')
    axs[1].set_title('Generated 128x128 FFT Power')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(delta_power, cmap='magma')
    axs[2].set_title('Delta Power (Gen - Upsampled)')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# --- Main Training and Validation Loop ---
for epoch in range(EPOCHS):
    model.train()
    fourier_encoder.train()
    total_train_loss = 0.0

    for i, (images, _) in enumerate(train_loader):
        images = images.to(DEVICE)
        input_data, target_pixels, _ = prepare_model_input(images, coords_32x32, fourier_encoder)
        # At each training batch
        coords_noisy = add_white_noise(coords_32x32, scale=0.01)
        # queries = repeat(fourier_encoder(coords_noisy), 'n d -> b n d', b=images.size(0))

        batch_coords_noisy = repeat(coords_noisy, 'n d -> b n d', b=images.size(0))  # (b, n, d)
        queries = fourier_encoder(batch_coords_noisy)
        
        optimizer.zero_grad(set_to_none=True)
        reconstructed_pixels = model(input_data, queries=queries)
        loss = loss_fn(reconstructed_pixels, target_pixels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        if (i + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], LR: {scheduler.get_last_lr()[0]:.6f}, Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    
    model.eval()
    fourier_encoder.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            input_data, target_pixels, _ = prepare_model_input(images, coords_32x32, fourier_encoder)
            # queries = repeat(fourier_encoder(coords_32x32), 'n d -> b n d', b=images.size(0))

            batch_coords = repeat(coords_32x32, 'n d -> b n d', b=images.size(0))
            queries = fourier_encoder(batch_coords)

            reconstructed_pixels = model(input_data, queries=queries)
            total_val_loss += loss_fn(reconstructed_pixels, target_pixels).item()

    avg_val_loss = total_val_loss / len(test_loader)
    print(f"--- Epoch [{epoch+1}/{EPOCHS}] Summary ---")
    print(f"  Avg Training Loss: {avg_train_loss:.4f}")
    print(f"  Avg Validation Loss: {avg_val_loss:.4f}\n")

    # --- Visualization at the end of each epoch ---
    with torch.no_grad():
        context_images, _ = next(iter(test_loader))
        context_images = context_images.to(DEVICE)[:8]
        b, c, h, w = context_images.shape

        # 1. Low-Resolution Reconstruction Test
        input_context, _, _ = prepare_model_input(context_images, coords_32x32, fourier_encoder)
        # queries_context = repeat(fourier_encoder(coords_32x32), 'n d -> b n d', b=context_images.size(0))

        queries_context = fourier_encoder(repeat(coords_32x32, 'n d -> b n d', b=context_images.size(0)))


        reconstructed_pixels = model(input_context, queries=queries_context)
        reconstructed_images = rearrange(reconstructed_pixels, 'b (h w) c -> b c h w', h=h, w=w)
        
        comparison_grid = torch.cat((context_images, reconstructed_images), dim=0)
        final_grid = torchvision.utils.make_grid(comparison_grid, nrow=8, padding=2)
        imshow(final_grid, f"Epoch {epoch+1}: Top: Original | Bottom: Reconstructed (32x32)")

        # ===============================================================
        # --- NEW: Call the FFT validation function ---
        # ===============================================================
        calculate_and_visualize_fft_power_delta(context_images, reconstructed_images, epoch + 1)
        # ===============================================================

        # 2. High-Resolution Generation Test
        # high_res_batch_coords = repeat(coords_128x128, 'n d -> b n d', b=b)
        # high_res_queries = fourier_encoder(high_res_batch_coords)
        
        # high_res_queries = repeat(fourier_encoder(coords_128x128), 'n d -> b n d', b=b)
        high_res_queries = fourier_encoder(repeat(coords_128x128, 'n d -> b n d', b=b))

        
        generated_pixels = model(input_context, queries=high_res_queries)
        generated_images = rearrange(generated_pixels, 'b (h w) c -> b c h w', h=IMAGE_SIZE_HI_RES, w=IMAGE_SIZE_HI_RES)

        generated_grid = torchvision.utils.make_grid(generated_images, nrow=4, padding=2)
        imshow(generated_grid, f"Epoch {epoch+1}: Generated High-Resolution Images (128x128)")
        compare_fft_upsampled_vs_generated(reconstructed_images, generated_images, epoch + 1)


print("--- Training finished. ---")
