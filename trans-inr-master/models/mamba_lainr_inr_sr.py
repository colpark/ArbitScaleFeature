import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import models
from models import register
from .cross_attention import CrossAttentionWithLearnedQueries


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

@register('mamba_lainr_inr_sr')
class MambaInr(torch.nn.Module):
    
    def __init__(self, tokenizer, hyponet, mamba_encoder, num_lp = 128, type = 'equidistant', n_group = 1, latent_token_len = 64, sin_init = True):
        super().__init__()

        self.dim = mamba_encoder['args']['dim'] #replace with mamba_encoder
        self.latent_token_len = latent_token_len
        self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet, args={'hidden_dim': self.dim})
        '''if max(gen_layers) >= self.hyponet.depth:
            raise Exception("all elements of gen_layers must be less that hyponet depth")
        elif min(gen_layers) < 0:
            raise Exception("all elements of gen_layers must be >= 0")'''
        self.mamba_encoder = models.make(mamba_encoder) #replace w mamba
        self.input_len = self.tokenizer.n_patches
        #self.gen_layers = gen_layers
        self.type = type
        self.num_lp = num_lp
        
        self.n_group = n_group

        '''self.U_values = nn.ParameterDict()
        
        self.V_projs = nn.ModuleDict()
        self.V_cross_att = nn.ModuleDict()

        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in gen_layers:
                mod_shape = (shape[0], self.latent_token_len)
                self.U_values[name] = nn.Parameter(init_wb(mod_shape))
                
                self.V_projs[name] = nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, shape[1])
                )
                self.V_cross_att[name] = CrossAttentionWithLearnedQueries(dim  = self.dim, num_queries = self.latent_token_len)
            else:
                self.U_values[name] = nn.Parameter(init_wb(shape))'''

        if sin_init:
            self.lps = nn.Parameter(self.get_sinusoidal_embeddings(self.num_lp, self.dim))
        else:
            self.lps = nn.Parameter(torch.randn(self.num_lp, self.dim))
       
        self.lp_idxs = None
        self.set_lp_idxs(self.input_len, type = self.type, n = self.n_group)
        self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    def get_sinusoidal_embeddings(self, n, d):
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
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def get_sigmas(self):
        if hasattr(self.hyponet, 'get_sigmas'):
            return self.hyponet.get_sigmas()
        else:
            return None

    def set_sigmas(self, sigmas):
        if hasattr(self.hyponet, 'set_sigmas'):
            return self.hyponet.set_sigmas(sigmas)
        else:
            return None

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

    def forward(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = self.add_lp(dtokens)
        mamba_out = self.mamba_encoder(all_tokens)
        mamba_out = self.extract_lp_tokens(mamba_out)

        '''params = dict()
        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in self.gen_layers:
                V_layer = self.V_cross_att[name](mamba_out)
                V_layer = self.V_projs[name](V_layer)
                U = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                W = torch.bmm(U, V_layer)
                W = F.normalize(W, dim=1)
                params[name] = W
                
            else:
                W = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                W = F.normalize(W, dim=1)
                params[name] = W'''           
            
        #self.hyponet.set_params(params)
        return self.hyponet, mamba_out
