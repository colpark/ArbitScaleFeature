import json
import math
import numpy as np
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(Embedder, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']
        self.embed = CoordinateEmbedder(method = pe_method, 
                                        n_continuous_dim = 4, 
                                        target_dim = embed_dim, 
                                        learnable_projection = learnable_projection)

    def forward(self, pos):
        pos_embed = self.embed(pos)  
        return pos_embed
        
class CoordinateEmbedder(nn.Module):
    """
    Three different continuous coordinate embedding methods are merged.
    1. Fourier features
    2. Nerf
    3. Continuous PE
    """
    
    def __init__(self, method = 'cpe', n_continuous_dim = 3, target_dim = 256, learnable_projection = False):
        super(CoordinateEmbedder, self).__init__()
        
        pseudo_input = torch.randn(1, 2, n_continuous_dim)
        
        if method == 'ff':
            self.get_ff()
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            # print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
            
        elif method == 'nerf':
            multires = 10
            self.get_nerf(multires, n_continuous_dim)
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            # print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
                        
        elif method == 'cpe':
            self.get_cpe(n_continuous_dim, target_dim)
            self.projection = None     
            
        elif method == 'none':
            self.pec = nn.Identity()
            self.projection = nn.Parameter(torch.randn(3, target_dim), requires_grad = learnable_projection)
            
    def apply_projection(self, tensor):
        return torch.matmul(tensor, self.projection)
    
    def get_ff(self,):
        pos2fourier_position_encoding_kwargs = dict(
        num_bands = [12, 12, 12],
        max_resolution = [20, 20, 20],
        )
        self.pec = FourierPositionEncoding(**pos2fourier_position_encoding_kwargs)

    def get_cpe(self, n_continuous_dim, target_dim):
        self.pec = PositionEmbeddingCoordsSine(n_dim = n_continuous_dim, d_model = target_dim)
    
    def get_nerf(self, multires, n_continuous_dim):
        embed_kwargs = {
                'include_input': True,
                'n_continuous_dim': n_continuous_dim,
                'max_freq_log2': multires-1,
                'num_freqs': multires,
                'log_sampling': True,
                'periodic_fns': [torch.sin, torch.cos],
            }
        self.pec = NerfEmbedder(**embed_kwargs)

    def forward(self, tensor):
        """
        tensor: b x N_seq x self.n_continuous_dim
        out: b x N_seq x self.target_dim
        """
        out = self.pec.forward(tensor)
        if self.projection is not None:
            out = self.apply_projection(out)
        return out

    
class FourierPositionEncoding():
    """ Fourier (Sinusoidal) position encoding. """

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    def output_size(self):
        """ Returns size of positional encodings last dimension. """
        encoding_size = sum(self.num_bands)
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += len(self.max_resolution)
        return encoding_size

    def forward(self, pos=None):
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(2 ** 10), concat_pos=True, sine_only=False):
    """
    Generate a Fourier feature position encoding with linear spacing.

    Args:
        pos: The Tensor containing the position of n points in d dimensional space.
        num_bands: The number of frequency bands (K) to use.
        max_resolution: The maximum resolution (i.e., the number of pixels per dim). A tuple representing resoltuion for each dimension.
        concat_pos: Whether to concatenate the input position encoding to the Fourier features.
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    """
    batch_size = pos.shape[0]
    min_freq = 1.0 
    stacked = []
    
    for i, (res, num_band) in enumerate(zip(max_resolution, num_bands)):       
        stacked.append(pos[:, :, i, None] * torch.linspace(start=min_freq, end=res / 2, steps=num_band)[None, :].to(device = pos.device))

    per_pos_features = torch.cat(stacked, dim=-1)  
    per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


class NerfEmbedder:
    def __init__(self, n_continuous_dim, include_input, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        
        self.n_continuous_dim = n_continuous_dim
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.n_continuous_dim 
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    
class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
       arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super(PositionEmbeddingCoordsSine, self).__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


def torchgengrid(steps=(32, 32, 32, 32), bot=(0, 0, 0, 0), top=(1, 1, 1, 1)):
    arrs = []
    for bot_, top_, step_ in zip(bot, top, steps):
        arrs.append(torch.linspace(bot_, top_, steps=step_))
    meshlist = torch.meshgrid(*arrs, indexing='ij')
    mesh = torch.stack(meshlist, dim=len(steps))
    return mesh


class PatchEmbed(nn.Module):
    """ 4D Image to Patch Embedding
    """
 
    def __init__(
        self,
        img_size=(96, 96, 96, 20),
        patch_size=(6, 6, 6, 2),
        in_chans=1,
        embed_dim=24,
        norm_layer=None,
        flatten=False,
        spatial_dims=3,
        learnable_pos_projection=False,
        learnable_x_projection=True,
        pe_method='nerf',
    ):
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
        #assert patch_size[3] == 1, "temporal axis merging is not implemented yet"
 
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] * self.grid_size[3]
        self.flatten = flatten
        self.orig_size = in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3]
        self.projection = nn.Parameter(torch.randn(self.orig_size, self.embed_dim), requires_grad = learnable_x_projection)
        self.mesh = torchgengrid(steps=self.grid_size, bot=(0, 0, 0, 0), top=(1, 1, 1, 1)) # (A,B,C,D,4)
        self.pe_method = pe_method
        if pe_method == 'learned':
            pD, pH, pW, pT = self.grid_size
            self. pos = nn.Parameter(torch.randn(pD, pH, pW, pT, self.embed_dim), requires_grad = True)
        else:
            pos = Embedder(pe_method=pe_method, embed_dim=embed_dim, learnable_projection = learnable_pos_projection)(self.mesh) # B, ..., embed_dim
            self.register_buffer('pos', pos.unsqueeze(0)) # (1,C,*grid_size)
            print(self.pos.shape)
 
    def forward(self, x):
        # print(x.shape)
        B, C, D, H, W, T = x.shape
        # assert D == self.img_size[0], f"Input image height ({D}) doesn't match model ({self.img_size[0]})."
        # assert H == self.img_size[1], f"Input image width ({H}) doesn't match model ({self.img_size[1]})."
        # assert W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]})."
        x = self.patchify(x)
        return x
 
    def patchify(self, x):
        B, C, D, H, W, T = x.shape
        pD, pH, pW, pT = self.grid_size
        sD, sH, sW, sT = self.patch_size
 
        x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
        patch_x = x.view(B, pD, pH, pW, pT, -1).contiguous()

        # Linear Projection
        x = torch.matmul(patch_x, self.projection)
        if self.pe_method == 'learned':
            x = x + einops.repeat(self.pos, 'z h w t d -> b z h w t d', b=B)
        else:
            x = x + self.pos
        return x

def fourier_encode(xy: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    xy = xy.unsqueeze(1)
    freq1 = freq[0:len(freq)//2 ]
    freq2 = freq[len(freq)//2:]
    
    
    freq1 = torch.tensor(freq1, dtype = torch.float32).view(1, -1, 1)
    freq2 = torch.tensor(freq2, dtype = torch.float32).view(1, -1, 1)

    scaled1 = 2 * torch.pi * (1/freq1) * xy  
    scaled2 = 2 * torch.pi * (1/freq2) * xy  

    sin_feat = torch.sin(scaled1)  
    cos_feat = torch.cos(scaled2)  


    features = torch.cat([sin_feat, cos_feat], dim=-1)  
    return features.view(xy.shape[0], -1) 

class MambaPatchTokenizer4D(nn.Module):
    """4D Patch Tokenizer with PatchEmbed patchify + MambaPatchTokenizer-style pos_emb"""
    def __init__(
        self,
        img_size=(96, 96, 96, 20),
        patch_size=(6, 6, 6, 2),
        in_chans=1,
        embed_dim=24,
        norm_layer=None,
        flatten=False,
        spatial_dims=3,
        learnable_pos_projection=False,  # unused, kept for signature compatibility
        learnable_x_projection=True,     # unused, kept for signature compatibility
        pe_method='nerf',                # unused, replaced with learned/fourier logic
        pos_emb='learned',
        pos_emb_dim=64,
        fourier_size=20
    ):
        super().__init__()
        assert len(patch_size) == 4, "Must give four numbers for patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.embed_dim = embed_dim
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.pos_emb_dim = pos_emb_dim
        self.fourier_size = fourier_size
        self.learned_posemb = (pos_emb == 'learned')

        orig_size = in_chans * np.prod(patch_size)
        # If using fixed pos_emb, patch feature dim = embed_dim - pos_emb_dim
        proj_dim = embed_dim if self.learned_posemb else embed_dim - pos_emb_dim
        self.prefc = nn.Linear(orig_size, proj_dim)

        if self.learned_posemb:
            self.posemb = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        else:
            self.posemb = self.generate_posemb(*self.grid_size)  # (num_patches, pos_emb_dim)

    def generate_posemb(self, D, H, W, T):
        # Normalized coords in [0, 1]
        z = np.linspace(0, 1, D)
        y = np.linspace(0, 1, H)
        x = np.linspace(0, 1, W)
        t = np.linspace(0, 1, T)
        zz, yy, xx, tt = np.meshgrid(z, y, x, t, indexing='ij')
        coords = np.stack([xx, yy, zz, tt], axis=-1).reshape(-1, 4)
        coords = torch.tensor(coords, dtype=torch.float32)

        freq = [1 / num for num in range(1, self.pos_emb_dim // 2 + 1)]
        X = fourier_encode(coords, freq)  # same helper as in your 2D tokenizer
        return X  # (num_patches, pos_emb_dim)

    def patchify(self, x):
        # Exactly like PatchEmbed.patchify
        B, C, D, H, W, T = x.shape
        pD, pH, pW, pT = self.grid_size
        sD, sH, sW, sT = self.patch_size
        x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous()
        x = x.view(B, self.num_patches, -1)
        return x

    def forward(self, x):
        x = self.patchify(x)        # (B, num_patches, orig_size)
        x = self.prefc(x)           # (B, num_patches, proj_dim)

        if self.learned_posemb:
            x = x + self.posemb.unsqueeze(0)
        else:
            B = x.size(0)
            pos = self.posemb.unsqueeze(0).repeat(B, 1, 1).to(x.device)
            x = torch.cat((pos, x), dim=2)  # concat pos_emb
        return x

# example usage

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device= None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * torch.sigmoid(x)
    
class PseudoModel(nn.Module):
    def __init__(self, embed_dim=128, img_size=(96, 96, 96, 20), patch_size=(6, 6, 6, 2), 
                 learnable_pos_projection=False, learnable_x_projection=True, pe_method='nerf'):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                   learnable_pos_projection=learnable_pos_projection,
                                   learnable_x_projection=learnable_x_projection)
        self.output_layer = nn.Linear(embed_dim, self.embedder.orig_size)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        x = self.embedder(x)  # Add slight noise
        x = x.contiguous().reshape(B, -1, self.embed_dim)
        x = self.norm(x)
        return self.output_layer(x)
