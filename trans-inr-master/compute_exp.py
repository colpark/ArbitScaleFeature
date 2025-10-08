import torch
import torch.nn as nn
import yaml
import einops
import torchvision
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import umap.umap_ as umap
from utils import make_coord_grid
import yaml
from datasets import make as make_dataset
from models import make as make_model
import pickle
from sklearn.manifold import TSNE
from glob import glob
import ipywidgets as widgets
from ipywidgets import interact_manual
import matplotlib.pyplot as plt
import numpy as np
from datasets.fmri_dataloader import DataModule
from patchembed import PatchEmbed
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from models.hyponets.layers import batched_linear_mm
import math
import time

class LAINRDecoder(nn.Module):
    def __init__(self, feature_dim, input_dim, output_dim, sigma_q, sigma_ls, n_patches, hidden_dim = 256, context_dim = None):
        super().__init__()
        self.layer_num = len(sigma_ls)
        self.n = feature_dim//(2*input_dim)
        self.omegas = torch.logspace(1, math.log10(sigma_q), self.n)
        self.patch_num = int(math.sqrt(n_patches))
        self.alpha = 10.0
        self.omegas_l = [torch.logspace(1, math.log10(sigma_ls[i]), self.n) for i in range(self.layer_num)]
        self.query_lin = nn.Linear(feature_dim, hidden_dim)
        self.modulation_ca = SharedTokenCrossAttention(query_dim = hidden_dim, heads=2)
        self.bandwidth_lins = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for i in range(self.layer_num)
                                       ])
        self.modulation_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for i in range(self.layer_num)
        ])
        self.hv_lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(len(sigma_ls) - 1)
        ])
        self.out_lins = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(len(sigma_ls))
        ])
        self.act = nn.ReLU()
        
    def calc_gamma(self, x, omegas):
        #x is passed as H*W, D
        L = x.shape[0]
        coords = x.unsqueeze(-1)  # (H*W, 2, 1)
        omegas = omegas.view(1, 1, -1).to(x.device)  # (1, 1, F)
        arg = torch.pi * coords * omegas  # shape: (B, 2, F)
        sin_part = torch.sin(arg)
        cos_part = torch.cos(arg)
        
        gamma = torch.cat([sin_part, cos_part], dim=-1).view(L, -1)  
        
        return gamma

    def get_patch_index(self, grid, D, H, W, T):
        z = grid[:, 0]  # D dimension
        y = grid[:, 1]  # H dimension
        x = grid[:, 2]  # W dimension
        t = grid[:, 3]  # T dimension
    
        # Convert normalized coords to integer indices
        z_idx = (z * D).to(torch.int32)
        y_idx = (y * H).to(torch.int32)
        x_idx = (x * W).to(torch.int32)
        t_idx = (t * T).to(torch.int32)
    
        # Flatten: W → H → D → T
        return (((t_idx * D + z_idx) * H + y_idx) * W + x_idx)
    
    
    def approximate_relative_distances(self, target_index, D, H, W, T, m):
        alpha = self.alpha
        N = D * H * W * T
        t = target_index / N
        token_positions = torch.tensor([(i + 0.5) / m for i in range(m)],
                                        device=target_index.device)
    
        rel_distances = -1 * alpha * torch.stack([
            torch.abs((t - s) ** 2) for s in token_positions
        ], dim=0)
        return rel_distances

    def forward(self, x, tokens, shape):
        D, H, W, T = shape
        B, query_shape = x.shape[0], x.shape[1]
        x = x.view(B, -1, x.shape[-1]) #B, HW, 2
        #print(f'query shape is {x.shape[1:]}')
        grid = x[0]
        indexes = self.get_patch_index(grid, D, H, W, T)
        rel_distances = self.approximate_relative_distances(indexes, D, H, W, T, len(tokens[0]))
        bias = rel_distances.transpose(1, 0)
        bias = einops.repeat(bias, 'l n -> b l n', b=B) #B, L, HW
        x_q = einops.repeat(self.calc_gamma(x[0], self.omegas), 'l d -> b l d', b=B) #B, HW, input_dim
        #print(f'x_q shape is {x_q.shape}')
        x_q = self.act(self.query_lin(x_q))

    
        modulation_vector = self.modulation_ca(x_q, context = tokens, bias = bias)

        modulations_l = []
        h_f = []


        for k in range(self.layer_num):
            x_l = einops.repeat(self.calc_gamma(x[0], self.omegas_l[k]), 'l d -> b l d', b=B)
            h_l = self.act(self.bandwidth_lins[k](x_l))
            h_f.append(h_l)
            m_l = self.act(h_l + self.modulation_lins[k](modulation_vector))
            modulations_l.append(m_l)

        h_v = [modulations_l[0]]

        for i in range(self.layer_num - 1):
            h_vl = self.act(self.hv_lins[i](modulations_l[i+1] + h_v[i]))
            h_v.append(h_vl)

        outs = [self.out_lins[i](h_v[i]) for i in range(self.layer_num)]

        out = sum(outs)
        out = out.view(B, query_shape, -1)  # (B, H, W, output_dim)

        return out

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

'''class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)'''

class SharedTokenCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=2, dim_head=64):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context, bias=None):
        # x: (B, HW, D)         ← 1 query per pixel (you can squeeze that 1)
        # context: (B, L, D)       ← shared tokens

        B, HW, D = x.shape

        H = self.heads
        Dh = self.dim_head
        D_inner = H * Dh

        q = self.to_q(x)              # (B, HW, H*Dh)
        kv = self.to_kv(context)      # (B, L, 2*H*Dh)
        k, v = kv.chunk(2, dim=-1)    # (B, L, H*Dh)

        # Reshape
        q = q.view(B, HW, H, Dh).transpose(1, 2)   # (B, H, HW, Dh)
        k = k.view(B, -1, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        v = v.view(B, -1, H, Dh).transpose(1, 2)   # (B, H, L, Dh)

        # Attention
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, HW, L)
        if bias != None:
            bias = einops.repeat(bias, 'b l n -> b h l n', h=H) #B, L, HW
            sim += bias
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)                              # (B, H, HW, Dh)

        out = out.transpose(1, 2).contiguous().view(B, HW, D_inner)  # (B, HW, H*Dh)
        out = self.to_out(out)                                        # (B, HW, D)
        return out                                   # (B, HW, 1, D)


class BiMamba(torch.nn.Module):
    def __init__(self, dim = 512):
        super(BiMamba, self).__init__()
        
        self.f_mamba = Mamba(d_model = dim)
        self.r_mamba = Mamba(d_model = dim)
        
    def forward(self, x, **kwargs):
        x_f = self.f_mamba(x, **kwargs)
        x_r = torch.flip(self.r_mamba(torch.flip(x, dims=[1]), **kwargs), dims=[1])
        out = (x_f + x_r)/2
        
        return out

 
class MambaEncoder(torch.nn.Module):
    def __init__(self, depth = 6, dim = 768, ff_dim = None, dropout=0.):
        super(MambaEncoder, self).__init__()
        if not ff_dim:
            self.ff_dim = 4*dim
        else: 
            self.ff_dim = ff_dim
        token_dim = dim
        self.blocks = nn.ModuleList([
            Block(
                dim=token_dim,
                mixer_cls= lambda dim: BiMamba(dim),
                mlp_cls= lambda dim: torch.nn.Sequential(
                    nn.Linear(dim, self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.ff_dim, dim),
                    nn.Dropout(dropout),
                ),
                norm_cls= nn.LayerNorm,  # or RMSNorm, 
                fused_add_norm=False
            )
            for _ in range(depth)
        ])
    
    def forward(self, x):
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual=residual)
        return x

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

class MambaInr(torch.nn.Module):
    
    def __init__(self, input_size, patch_size, hyponet, mamba_encoder, num_lp = 128, type = 'equidistant', n_group = 1, latent_token_len = 64):
        super().__init__()

        self.dim = 256
        self.latent_token_len = latent_token_len
        #self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet =  hyponet
        self.shape = [input_size[i]//patch_size[i] for i in range(len(input_size))]
        self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim)
        self.input_len = self.patchifier.num_patches
        self.mamba_encoder = mamba_encoder
        self.type = type
        self.num_lp = num_lp
        self.n_group = n_group
                
            
        self.lps = nn.Parameter(torch.randn(self.num_lp, self.dim))
        self.lp_idxs = None
        self.set_lp_idxs(self.input_len, type = self.type, n = self.n_group)
        self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    def set_lp_idxs(self, seq_len, type = 'equidistant', n = 1):
        total_len = seq_len + self.num_lp
        if type == 'equidistant':
            insert_idxs = torch.linspace(0, total_len - 1, steps=self.num_lp).long()
            self.lp_idxs = insert_idxs
        elif type == 'middle':
            insert_idxs = (np.array(range(self.num_lp))+(seq_len//2)).tolist()
            self.lp_idxs = insert_idxs
        elif type == 'n_group':
            if self.num_lp%n != 0:
                raise Exception("n must divide number of lps evenly")
            insert_idxs = []
            pre_idxs = torch.linspace(0, total_len - n, steps=self.num_lp//n).long()
            for idx in pre_idxs:
                insert_idxs.extend([idx+i for i in range(n)])
            self.lp_idxs = insert_idxs
            

    def add_lp(self, x):
        B, L, D = x.shape
        w = einops.repeat(self.lps, 'n d -> b n d', b=B)  # (B, N, D)
        x_full = torch.cat([x, w], dim=1)  # (B, L + N, D)
        x_perm = x_full[:, self.perm]  # (B, L + N, D) — interleaved

        return x_perm

    def set_patchifier(self, patchifier):
        self.patchifier = patchifier
        self.input_len = self.patchifier.num_patches
        self.set_lp_idxs(self.input_len, type = self.type, n = self.n_group)
        self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)
    
    def extract_lp_tokens(self, x):
        return x[:, self.lp_idxs]

    def compute_interleave_permutation(self, seq_len, n_insert):
        total_len = seq_len + n_insert
        insert_idxs = torch.linspace(0, total_len - 1, steps=n_insert).long()  
        token_ids = torch.arange(seq_len + n_insert)
        perm = torch.full((total_len,), -1, dtype=torch.long)
        

        perm[insert_idxs] = torch.arange(seq_len, seq_len + n_insert)
        input_token_ids = torch.arange(seq_len)
        perm[perm == -1] = input_token_ids
        
        return perm

    def scan(self, x):
        
        B, pD, pH, pW, pT, D = x.shape
        x = x.permute(0, 4, 1, 2, 3, 5)  # (B, pT, pD, pH, pW, D)
        #x = x.permute(0, 2, 3, 4, 1, 5)
    
        # Flatten spatial+time dims into one dimension
        x = x.reshape(B, pD*pH*pW*pT, D)
        return x

    def forward(self, data, coord):
        start = time.time()
        all_data = data.float()# (B, C, Z, H, W, T)
        B = all_data.shape[0]
        dtokens = self.patchifier(all_data) #(B, pD, pH, pW, pT, D)
        dtokens = self.scan(dtokens) # (B, pD*pH*pW*pT, D)
        #dtokens = self.tokenizer(data)
        #Here, delete patches and keep track of which ones are deleted. Can create the pixel mask here too and return it
        #lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = self.add_lp(dtokens)
        mamba_out = self.mamba_encoder(all_tokens)
        mamba_out = self.extract_lp_tokens(mamba_out)
        encoder_time = time.time() - start
        start = time.time()
        pred = self.hyponet(coord, mamba_out, self.shape)
        decoder_time = time.time() - start

        return pred, decoder_time, encoder_time
class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TPreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TPreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                TPreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return x
        
class LAINR(torch.nn.Module):
    
    def __init__(self, input_size, patch_size, hyponet, transformer_encoder, num_lp = 256):
        super().__init__()

        self.dim = 256 #replace with mamba_encoder
        #self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim, pe_method = 'learned')
        self.hyponet = hyponet
        self.shape = [input_size[i]//patch_size[i] for i in range(len(input_size))]
    
        self.transformer_encoder = transformer_encoder
        self.input_len = self.patchifier.num_patches
        self.lps = nn.Parameter(torch.randn(num_lp, self.dim))

    def scan(self, x):
        B, pD, pH, pW, pT, D = x.shape
        x = x.permute(0, 4, 1, 2, 3, 5)  # (B, pT, pD, pH, pW, D)
        #x = x.permute(0, 2, 3, 4, 1, 5)
    
        # Flatten spatial+time dims into one dimension
        x = x.reshape(B, pD*pH*pW*pT, D)
        return x
        
    def set_patchifier(self, patchifier):
        self.patchifier = patchifier
        self.input_len = self.patchifier.num_patches
        #self.set_lp_idxs(self.input_len, type = self.type, n = self.n_group)
        #self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    def forward(self, data, coord):
        start = time.time()
        all_data = data.float()
        B = all_data.shape[0]
        dtokens = self.patchifier(all_data)
        dtokens = self.scan(dtokens)
        lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = torch.cat([dtokens, lps], dim=1)
        trans_out = self.transformer_encoder(all_tokens)
        trans_out = trans_out[:, -len(self.lps):, :]
        encoder_time = time.time() - start
        
        start = time.time()
        pred = self.hyponet(coord, trans_out, self.shape)
        decoder_time = time.time() - start
        
        return pred, decoder_time, encoder_time

class HypoMlp(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, query_shape, -1)
        return x
        
class TransInr(nn.Module):

    def __init__(self, input_size, patch_size, hyponet, n_groups, transformer_encoder):
        super().__init__()
        dim = 256
        self.dim = dim
        #self.tokenizer = models.make(tokenizer, args={'dim': dim})
        self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim, pe_method = 'learned')
        self.hyponet = hyponet
        self.transformer_encoder = transformer_encoder

        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        log_result(f'n_wtokens is {n_wtokens}')
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, self.dim))

    def scan(self, x):
        B, pD, pH, pW, pT, D = x.shape
        x = x.permute(0, 4, 1, 2, 3, 5)  # (B, pT, pD, pH, pW, D)
        #x = x.permute(0, 2, 3, 4, 1, 5)
    
        # Flatten spatial+time dims into one dimension
        x = x.reshape(B, pD*pH*pW*pT, D)
        return x

    def set_patchifier(self, patchifier):
        self.patchifier = patchifier

    def forward(self, data, coord):
        start = time.time()
        all_data = data.float()
        B = all_data.shape[0]
        #dtokens = self.tokenizer(all_data)
        dtokens = self.patchifier(all_data)
        dtokens = self.scan(dtokens)
        
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        
        self.hyponet.set_params(params)
        encoder_time = time.time() - start
        
        start = time.time()
        pred = self.hyponet(coord)
        decoder_time = time.time() - start
        return pred, decoder_time, encoder_time
# =====================
# HELPER: Append to log
# =====================
def log_result(text):
    print(text)
    LOG_FILE = './compute_log.txt'
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

# =====================
# HELPER: Measure throughput + memory
# =====================
def benchmark_model(model, dataloader, token_len):
    # Warm-up
    print('warming up...')
    for i, batch in zip(range(NUM_WARMUP_STEPS), dataloader):
        #batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        gt_full = batch.pop('fmri_sequence')
        #gt = data
        B = gt_full.shape[0]

        # Flatten spatial dims for sampling
        coord = make_coord_grid(gt_full.shape[-4:], (0, 1), device=gt_full.device)
        coord = einops.repeat(coord, 'z h w t d -> b z h w t d', b=B)
        coord_flat = coord.view(B, -1, coord.shape[-1])  # (B, N, dim), N=Z*H*W*T
        
        gt = einops.rearrange(gt_full, 'b c z h w t -> b z h w t c')  # (B, Z, H, W, T, C)
        gt_flat = gt.view(B, -1, gt.shape[-1])  # (B, N, C)
    
        N = coord_flat.shape[1]
        sample_size = max(1, int(p * N))  # at least sample 1 coord
    
        # Random indices to sample, same number for each batch element
        indices = torch.randperm(N, device=gt.device)[:sample_size]  # (sample_size,)
    
        # Index coords and gt
        coord_sampled = coord_flat[:, indices, :]  # (B, sample_size, dim)
        gt_sampled = gt_flat[:, indices, :]       # (B, sample_size, C)
        
        pred, decoder_time, encoder_time = model(gt_full, coord_sampled)

        #pred = hyponet(coord, tokens) # b h w 3
        #gt = einops.rearrange(gt, 'b c z h w t -> b z h w t c')
        mses = ((pred - gt_sampled)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        #loss = model(batch['fmri_sequence']).loss
        loss.backward()
        torch.cuda.synchronize()

    # Measurement
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    decoder_total_time = 0.0
    encoder_total_time = 0.0

    total_tokens = BATCH_SIZE*NUM_MEASURE_STEPS*token_len
    #total_samples = 0
    
    print('measuring data...')
    for i, batch in zip(range(NUM_MEASURE_STEPS), dataloader):
        #batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        gt_full = batch.pop('fmri_sequence')
        #gt = data
        B = gt_full.shape[0]

        # Flatten spatial dims for sampling
        coord = make_coord_grid(gt_full.shape[-4:], (0, 1), device=gt_full.device)
        coord = einops.repeat(coord, 'z h w t d -> b z h w t d', b=B)
        coord_flat = coord.view(B, -1, coord.shape[-1])  # (B, N, dim), N=Z*H*W*T
        
        gt = einops.rearrange(gt_full, 'b c z h w t -> b z h w t c')  # (B, Z, H, W, T, C)
        gt_flat = gt.view(B, -1, gt.shape[-1])  # (B, N, C)
    
        N = coord_flat.shape[1]
        sample_size = max(1, int(p * N))  # at least sample 1 coord
    
        # Random indices to sample, same number for each batch element
        indices = torch.randperm(N, device=gt.device)[:sample_size]  # (sample_size,)
    
        # Index coords and gt
        coord_sampled = coord_flat[:, indices, :]  # (B, sample_size, dim)
        gt_sampled = gt_flat[:, indices, :]       # (B, sample_size, C)
        
        pred, decoder_time, encoder_time = model(gt_full, coord_sampled)
        decoder_total_time += decoder_time
        encoder_total_time += encoder_time

        #pred = hyponet(coord, tokens) # b h w 3
        #gt = einops.rearrange(gt, 'b c z h w t -> b z h w t c')
        mses = ((pred - gt_sampled)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        
        #tokens = batch["input_ids"].numel()
        #samples = batch["input_ids"].shape[0]
        #total_tokens += tokens
        #total_samples += samples

        #loss = model(batch['fmri_sequence']).loss
        loss.backward()
        torch.cuda.synchronize()

    elapsed = time.time() - start
    total_forward_time = decoder_total_time + encoder_total_time
    max_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    tokens_per_sec = total_tokens / total_forward_time
    tokens_per_encoder_sec = total_tokens/(1000*encoder_total_time)
    decoder_prop = decoder_total_time / total_forward_time
    encoder_prop = encoder_total_time / total_forward_time
    #samples_per_sec = total_samples / elapsed

    return max_mem, tokens_per_sec, tokens_per_encoder_sec, decoder_prop, encoder_prop
    #return max_mem, tokens_per_sec, samples_per_sec

DEVICE = "cuda"
TOKEN_LENGTHS = [128, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
BATCH_SIZE = 2
NUM_WARMUP_STEPS = 5
NUM_MEASURE_STEPS = 20
LOG_FILE = "compute_log.txt"
p = 0.1

def main():
    log_result('Beginning compute experiments')

    mamba_encoder = MambaEncoder(depth = 6, dim = 256, ff_dim = 1024)
    hyponet = LAINRDecoder(feature_dim = 256, input_dim = 4, output_dim = 1, sigma_q = 16, sigma_ls = [128, 32], n_patches = 128)
    model_a = MambaInr(input_size = [96, 96, 96, 2], patch_size = [8, 8, 8, 2], hyponet=hyponet, mamba_encoder=mamba_encoder, num_lp = 256, type = 'equidistant', n_group = 1, latent_token_len = 64)
    transformer_encoder = TransformerEncoder(dim=256, depth=6, n_head=12, head_dim=64, ff_dim=1024)
    model_b = LAINR(input_size = [96, 96, 96, 2], patch_size = [8, 8, 8, 2], hyponet=hyponet, transformer_encoder=transformer_encoder, num_lp = 256)

    trans_hypo = HypoMlp(depth = 5, in_dim = 4, out_dim = 1, hidden_dim = 256, use_pe = True, pe_dim = 256, out_bias=0, pe_sigma=1024)

    model_c = TransInr(input_size = [96, 96, 96, 2], patch_size = [8, 8, 8, 2], hyponet=trans_hypo, transformer_encoder=transformer_encoder, n_groups = 64)


    log_result('models loaded')
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # =====================
    # CONFIG
    # =====================
    
    # Example: three models
    '''models = {
        "ModelA": model_a.to(DEVICE),
        "ModelB": model_b.to(DEVICE),
        "ModelC": model_c.to(DEVICE)
    }'''
    
    models = {
        "Mamba-GINR": model_a.to(DEVICE),
        "LAGINR": model_b.to(DEVICE),
        "TransINR": model_c.to(DEVICE)
    }
    
    # Example: different tokenizers for different lengths
    tokenizers = {
        128: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (24, 24, 24, 1), embed_dim = 256),
        512: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (12, 12, 24, 1), embed_dim = 256),
        1024: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (12, 12, 12, 1), embed_dim = 256),
        2048: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (6, 12, 12, 1), embed_dim = 256),
        3072: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (4, 6, 12, 2), embed_dim = 256),
        4096: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (6, 6, 12, 1), embed_dim = 256),
        6144: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (4, 6, 12, 1), embed_dim = 256),
        8192: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (6, 6, 6, 1), embed_dim = 256),
        12288: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (4, 6, 6, 1), embed_dim = 256),
        16384: PatchEmbed(img_size = (96, 96, 96, 2), patch_size = (3, 6, 6, 1), embed_dim = 256)
    }
   


    results = {name: [] for name in models.keys()}

    with open(LOG_FILE, "w") as f:
        f.write("Benchmark Results Log\n")
        f.write("="*50 + "\n")
    
    for token_len in TOKEN_LENGTHS:
        log_result(f"\n=== Token Length: {token_len} ===")
        tokenizer = tokenizers[token_len].to(DEVICE)
    
    
        data_module = DataModule('./cfgs/hcp_data_all_config.yaml')
        data_module.setup()
        dataloader = data_module.train_dataloader()
    
        for model_name, model in models.items():
            log_result(f"Benchmarking {model_name}...")
            model.set_patchifier(tokenizer)
            model.eval()
            torch.cuda.empty_cache()
    
            try:
                mem, tok_sec, tok_enc_sec, prop_decoder, prop_encoder = benchmark_model(model, dataloader, token_len)
                log_result(f"Memory: {mem:.2f} MB | Throughput: {tok_sec:.2f} tok/s | Throughput Enc: {tok_enc_sec:.2f} 1000 tok/s | Prop of Time in Decoder: {prop_decoder:.2f} | Prop of Time in Encoder: {prop_encoder:.2f}")
    
                results[model_name].append({
                    "token_length": token_len,
                    "gpu_memory_mb": mem,
                    "tokens_per_sec": tok_sec,
                    "tokens_per_encoder_sec": tok_enc_sec,
                    "prop_decoder": prop_decoder,
                    "prop_encoder": prop_encoder
                })
    
            except torch.cuda.OutOfMemoryError:
                log_result(f"⚠️  OOM at token length {token_len} for {model_name}")
                torch.cuda.empty_cache()
                results[model_name].append({
                    "token_length": token_len,
                    "gpu_memory_mb": None,
                    "tokens_per_sec": None,
                    "tokens_per_encoder_sec": None,
                    "prop_decoder": None,
                    "prop_encoder": None
                })
                '''mem, tok_sec, samp_sec = benchmark_model(model, dataloader)
                log_result(f"Memory: {mem:.2f} MB | Throughput: {tok_sec:.2f} tok/s, {samp_sec:.2f} samples/s")
    
                results[model_name].append({
                    "token_length": token_len,
                    "gpu_memory_mb": mem,
                    "tokens_per_sec": tok_sec
                })
    
            except torch.cuda.OutOfMemoryError:
                log_result(f"⚠️  OOM at token length {token_len} for {model_name}")
                torch.cuda.empty_cache()
                results[model_name].append({
                    "token_length": token_len,
                    "gpu_memory_mb": None,
                    "tokens_per_sec": None
                })'''
    
    
    # =====================
    # PLOT RESULTS
    # =====================
    for metric, ylabel in [("gpu_memory_mb", "GPU Memory (MB)"), ("tokens_per_sec", "Throughput (tokens/sec)"), ("tokens_per_encoder_sec", "Encoder Throughput (1000 tokens/sec)"), ("prop_decoder", "Proportion of Time in Decoder"), ("prop_encoder", "Proportion of Time in Encoder")]:
        plt.figure()
        for model_name in models.keys():
            x_vals = [r["token_length"] for r in results[model_name]]
            y_vals = [r[metric] if r[metric] is not None else float("nan") for r in results[model_name]]
            xy_pair = (x_vals, y_vals)
            
            with open(f'compute_exp_arrays_{model_name}_{metric}','wb') as f: 
                pickle.dump(xy_pair, f)
                
            plt.plot(
                x_vals,
                y_vals,
                marker='o',
                label=model_name
            )
        plt.xlabel("Token Length")
        plt.ylabel(ylabel)
        plt.title(f"Token Length vs {ylabel}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric}.png")
        plt.close()
        
    '''for metric, ylabel in [("gpu_memory_mb", "GPU Memory (MB)")]:
        plt.figure()
        for model_name in models.keys():
            x_vals = [r["token_length"] for r in results[model_name]]
            y_vals = [r[metric] if r[metric] is not None else float("nan") for r in results[model_name]]
            plt.plot(
                x_vals,
                y_vals,
                marker='o',
                label=model_name
            )
        plt.xlabel("Token Length")
        plt.ylabel(ylabel)
        plt.title(f"Token Length vs {ylabel}")
        plt.legend()
        plt.savefig("GPU_memory.png")
        plt.close()'''


     

if __name__ == "__main__":
    main()