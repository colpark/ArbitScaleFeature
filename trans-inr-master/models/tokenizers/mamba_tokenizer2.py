import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import register
from einops import rearrange, repeat
import math


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

def nerf_posenc(xy: torch.Tensor, L: int) -> torch.Tensor:
    """
    xy: (B, 2) tensor in [-1, 1]
    Returns: (B, 4L) encoded tensor
    """
    B = xy.shape[0]
    freqs = 2 ** torch.arange(L, dtype=torch.float32, device=xy.device) * math.pi  # (L,)
    freqs = freqs.view(1, L, 1)                      # (1, L, 1)
    xy = xy.view(B, 1, 2)                            # (B, 1, 2)

    scaled = xy * freqs                              # (B, L, 2)
    sin_feat = torch.sin(scaled)
    cos_feat = torch.cos(scaled)

    return torch.cat([sin_feat, cos_feat], dim=-1).view(B, -1)  # (B, 4L)

def patch_to_vector(patch, patch_pos, L=10, d=128):
    """
    patch: (m, m, 3) tensor of RGB values in [0, 1]
    patch_pos: (2,) tensor with normalized top-left corner coords in [-1, 1]
    returns: (d,) tensor
    """
    m = patch.shape[0]
    rgb_flat = patch.reshape(-1)            # shape: (3 * m * m,)
    posenc = nerf_posenc(patch_pos, L)      # shape: (4L,)
    feat = torch.cat([rgb_flat, posenc])    # shape: (3*m*m + 4L,)
    
    # Linear projection to dimension d
    proj = torch.nn.Linear(feat.shape[0], d).to(feat.device)
    return proj(feat)  # (d
    
@register('mamba_patch_tokenizer2')
class MambaPatchTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3, pos_emb = 'learned', pos_emb_dim = 32):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.fourier_size = 20
        self.padding = padding
        self.pos_emb_dim = pos_emb_dim
        self.dim = dim
        self.small_dim = 128
        self.postfc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, self.dim)
        
        self.n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * ((input_size[1] + padding[1] * 2)  // patch_size[1])        
        self.learned_posemb = (pos_emb == 'learned')

        if (self.learned_posemb):
            self.posemb_x = nn.Parameter(torch.randn(self.n_patches, dim))
            self.posemb_y = nn.Parameter(torch.randn(self.n_patches, dim))
        else:
            self.posemb_x = self.generate_posemb((input_size[0]+2*self.padding[0])//patch_size[0], (input_size[1] + padding[1] * 2)  // patch_size[1])
            self.posemb_y = self.generate_posemb((input_size[0]+2*self.padding[0])//patch_size[0], (input_size[1] + padding[1] * 2)  // patch_size[1], vertical = True)
            #print(self.posemb.shape)
            self.dim -= self.pos_emb_dim
        '''else:
            self.posemb = self.generate_posemb((input_size[0]+2*self.padding[0])//patch_size[0], (input_size[1] + padding[1] * 2)  // patch_size[1])
            print(self.posemb.shape)
            self.dim -= 4*self.fourier_size
        '''
 
        self.prefc_x = nn.Linear(patch_size[0] * patch_size[1] * img_channels, self.dim)
        self.prefc_y = nn.Linear(patch_size[0] * patch_size[1] * img_channels, self.dim)
        
        #self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, self.small_dim)
        #self.postfc = nn.Linear(self.small_dim + 4*self.fourier_size, self.dim)
        #self.postfc = nn.Linear(patch_size[0] * patch_size[1] * img_channels + 4*self.fourier_size, self.dim)
   

    def generate_posemb(self, H, W, vertical=False):
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        L = self.fourier_size
    
        if vertical:
            yy, xx = np.meshgrid(y, x)  # Column-major (vertical scan)
        else:
            xx, yy = np.meshgrid(x, y)  # Row-major (horizontal scan)
    
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        coords = torch.tensor(coords, dtype=torch.float32)
    
        freq = [1 / num for num in range(1, self.pos_emb_dim // 2 + 1)]
        X = fourier_encode(coords, freq)
        #X = nerf_posenc(X, L)
        
        return X


    def forward(self, data, mode = 'duo'):
        x = data['inp']
        B, C, H, W = x.shape
        x_vert = torch.transpose(x, 2, 3)
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C * p * p, L)
        x_vert = F.unfold(x_vert, p, stride=p, padding=self.padding)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc_x(x) #B, L, D

        x_vert = x_vert.permute(0, 2, 1).contiguous()
        x_vert = self.prefc_y(x_vert) #B, L, D
        #before prefc, 
        if (self.learned_posemb):
            x = x + self.posemb_x.unsqueeze(0)
            x_vert = x_vert + self.posemb_y.unsqueeze(0)
        else:
            x = torch.cat((self.posemb_x.repeat(B, 1, 1).to(x.device), x), dim = 2) #B, L, D+4L
            x_vert =torch.cat((self.posemb_y.repeat(B, 1, 1).to(x.device), x_vert), dim = 2)
            #x = self.postfc(x)

        if mode == 'quad':
            return (x, x_vert)
        else:
            return x
            
        
@register('mamba_tokenizer')
class MambaTokenizer(nn.Module):

    def __init__(self, input_size, dim, pos_emb = 'learned', padding=0, img_channels=3):
        super().__init__() 
        
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.learned_posemb = (pos_emb == 'learned')

        if (self.learned_posemb):
            self.posemb = nn.Parameter(torch.randn(input_size[0]*input_size[1], dim))
        else:
            self.posemb = self.generate_posemb(input_size[0], input_size[1])
            dim -= 32

        self.n_patches = ((input_size[0] + padding * 2)) * ((input_size[1] + padding * 2))

        self.prefc = nn.Linear(img_channels, dim)
            

    def scan(self, x):
        return rearrange(x.reshape(x.shape[0], x.shape[1], -1), 'b c l -> b l c')

    def generate_posemb(self, H, W):
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)  # shape: (H, W)
        
        X = torch.tensor(np.stack([xx, yy], axis=-1).reshape(-1, 2), dtype = torch.float32)  # shape: (H*W, 2)
        
        num_freq = 10
        gauss_scale = 10
        freq = [1/num for num in range(1, 17)]
        X = fourier_encode(X, freq)
        
        return X

    def forward(self, data):
        x = data['inp'] #B C H W
        #x = data #B C H W
        B, C, H, W = x.shape
        x = self.scan(x)
        x = self.prefc(x)
        if (self.learned_posemb):
            x = x + self.posemb.unsqueeze(0)
        else:
            x = torch.cat((self.posemb.repeat(B, 1, 1).to(x.device), x), dim = 2)
        return x