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

@register('mamba_composers_inr_delta')
class MambaInr(torch.nn.Module):
    
    def __init__(self, tokenizer, hyponet, mamba_encoder, gen_layers = [0], num_lp = 128, type = 'equidistant', n_group = 1):
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
        self.type = type
        self.n_group = n_group

        self.U_values = nn.ParameterDict()
        self.A_values = nn.ParameterDict()
        self.num_lp = num_lp
        self.V_projs = nn.ModuleDict()

        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in gen_layers:
                if i == 0:
                    mod_shape = (shape[0], num_lp)
                    self.A_values[name] = nn.Parameter(init_wb(mod_shape))
                    self.V_projs[name] = nn.Sequential(
                        nn.LayerNorm(self.dim),
                        nn.Linear(self.dim, shape[1])
                    )
                else:
                    mod_shape = (shape[0], num_lp)
                    self.A_values[name] = nn.Parameter(init_wb(mod_shape))
                    self.V_projs[name] = nn.Sequential(
                        nn.LayerNorm(self.dim),
                        nn.Linear(self.dim, shape[1])
                    )
                    self.U_values[name] = nn.Parameter(init_wb(shape))
            else:
                self.U_values[name] = nn.Parameter(init_wb(shape))
                        
            
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

        params = dict()
        for i, (name, shape) in enumerate(self.hyponet.param_shapes.items()):
            if i in self.gen_layers:
                if i == 0:
                    V_layer = self.V_projs[name](mamba_out)
                    A = einops.repeat(self.A_values[name], 'n m -> b n m', b=B)
                    W = torch.bmm(A, V_layer)
                    W = F.normalize(W, dim=1)
                    params[name] = W
                else:
                    V_layer = self.V_projs[name](mamba_out)
                    A = einops.repeat(self.A_values[name], 'n m -> b n m', b=B)
                    U = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                    dW = torch.bmm(A, V_layer)
                    W = U + dW
                    W = F.normalize(W, dim=1)
                    params[name] = W
                
            else:
                W = einops.repeat(self.U_values[name], 'n m -> b n m', b=B)
                W = F.normalize(W, dim=1)
                params[name] = W           
            
        self.hyponet.set_params(params)
        return self.hyponet
