import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import pickle
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

@register('mamba_inr_print_dt')
class MambaInr(torch.nn.Module):
    
    def __init__(self, tokenizer, hyponet, n_groups, mamba_encoder):
        super().__init__()

        self.dim = mamba_encoder['args']['dim'] #replace with mamba_encoder
        self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet)
        self.mamba_encoder = models.make(mamba_encoder) #replace w mamba
        self.input_len = self.tokenizer.n_patches

        self.base_params = nn.ParameterDict()
        self.n_wtokens = 0
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
            self.wtoken_rng[name] = (self.n_wtokens, self.n_wtokens + g)
            self.n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(self.n_wtokens, self.dim))
        self.wtoken_idxs = None
        self.set_wtoken_idxs(self.input_len)
        self.perm = self.compute_interleave_permutation(self.input_len, self.n_wtokens)

    def set_wtoken_idxs(self, seq_len):
        total_len = seq_len + self.n_wtokens
        insert_idxs = torch.linspace(0, total_len - 1, steps=self.n_wtokens).long()
        self.wtoken_idxs = insert_idxs
    
    '''def add_lp(self, x):
        total_len = self.input_len + self.n_wtokens
        mask = torch.zeros(total_len, dtype=torch.bool)
        mask[self.wtoken_idxs] = True

        out = torch.empty((x.shape[0], total_len, self.dim), dtype=torch.float32).to(x.device)
        out[:, mask] = self.wtokens
        out[:, ~mask] = x
        return out'''

    def add_lp(self, x):
        B, L, D = x.shape
        w = einops.repeat(self.wtokens, 'n d -> b n d', b=B)  # (B, N, D)
        x_full = torch.cat([x, w], dim=1)  # (B, L + N, D)
        x_perm = x_full[:, self.perm]  # (B, L + N, D) — interleaved

        return x_perm
    
    def extract_lp_tokens(self, x):
        return x[:, self.wtoken_idxs]

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
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        all_tokens = self.add_lp(dtokens)
        mamba_out, features, dts = self.mamba_encoder(all_tokens)
        with open('./features_dts.pkl', 'wb') as file:
                pickle.dump((features, dts), file)
        mamba_out = self.extract_lp_tokens(mamba_out)

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](mamba_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        self.hyponet.set_params(params)
        return self.hyponet
