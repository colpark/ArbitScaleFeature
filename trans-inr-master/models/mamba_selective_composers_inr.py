import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import models
from models import register


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

class Selective_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, shape):
        super().__init__()
        self.shape = shape
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shape[0]*shape[1])
        )

    def forward(self, x, U):  # x: (B, L, D)
        x = x.mean(dim=1) # B, D
        out = self.mlp(x)                  # (B, N*M)
        out = out.view(-1, self.shape[0], self.shape[1]) # (B, N, M)
        out = U.unsqueeze(0)*out
        return out

class Selective_U(nn.Module):
    def __init__(self, input_dim, hidden_dim, shape, r = 32):
        super().__init__()
        self.shape = shape 
        self.r = r
        self.to_A = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shape[0] * r)
        )
        self.to_B = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * shape[1])
        )

    def forward(self, x, U):  # x: (B, L, D)
        x = x.mean(dim=1) #B, D
        A = self.to_A(x).view(-1, self.shape[0], self.r)
        B = self.to_B(x).view(-1, self.r, self.shape[1])
        U_mod = torch.bmm(A, B)
        return  U*U_mod
        
        self.dim = mamba_encoder['args']['dim'] #replace with mamba_encoder
        self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet)
        if max(gen_layers) >= self.hyponet.depth:
            raise Exception("all elements of gen_layers must be less that hyponet depth")
        elif min(gen_layers) < 0:
            raise Exception("all elements of gen_layers must be >= 0")
        self.mamba_encoder = models.make(mamba_encoder) #replace w mamba
        self.input_len = self.tokenizer.n_patches
        self.gen_layers = gen_layers
        self.selective_U_dict = nn.ModuleDict()

        self.U_values = nn.ParameterDict()
        self.num_lp = num_lp
        self.V_projs = nn.ModuleDict()

@register('mamba_selective_composers_inr')
class MambaInr(torch.nn.Module):
    
    def __init__(self, tokenizer, mamba_encoder, hyponet, gen_layers = [0], num_lp = 128):
        super().__init__()

        self.dim = mamba_encoder['args']['dim'] #replace with mamba_encoder
        self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet)
        if max(gen_layers) >= self.hyponet.depth:
            raise Exception("all elements of gen_layers must be less that hyponet depth")
        elif min(gen_layers) < 0:
            raise Exception("all elements of gen_layers must be >= 0")
        self.mamba_encoder = models.make(mamba_encoder) #replace w mamba
        self.input_len = self.tokenizer.n_patches
        self.gen_layers = gen_layers
        self.selective_U_dict = nn.ModuleDict()

        self.U_values = nn.ParameterDict()
        self.num_lp = num_lp
        self.V_projs = nn.ModuleDict()

        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in gen_layers:
                mod_shape = (shape[0], num_lp)
                self.U_values[name] = nn.Parameter(init_wb(mod_shape))
                self.selective_U_dict[name] = Selective_U(input_dim = self.dim, hidden_dim = 2*self.dim, shape = mod_shape)
                self.V_projs[name] = nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, shape[1]),
                )
            else:
                self.U_values[name] = nn.Parameter(init_wb(shape))
                
            
        self.lps = nn.Parameter(torch.randn(self.num_lp, self.dim))
        self.lp_idxs = None
        self.set_lp_idxs(self.input_len)
        self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    def set_lp_idxs(self, seq_len):
        total_len = seq_len + self.num_lp
        insert_idxs = torch.linspace(0, total_len - 1, steps=self.num_lp).long()
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
        dtokens = self.tokenizer(data) #B, L, D
        B = dtokens.shape[0]
        lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = self.add_lp(dtokens)
        mamba_out = self.mamba_encoder(all_tokens)
        mamba_out = self.extract_lp_tokens(mamba_out)

        params = dict()
        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in self.gen_layers:
                V_layer = self.V_projs[name](mamba_out)
                U = self.U_values[name]
                U_shape = U.shape
                U = self.selective_U_dict[name](dtokens, U)
                #U = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                W = torch.bmm(U, V_layer)
                W = F.normalize(W, dim=1)
                params[name] = W
                
            else:
                W = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                W = F.normalize(W, dim=1)
                params[name] = W           
            
        self.hyponet.set_params(params)
        return self.hyponet
