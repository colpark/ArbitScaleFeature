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

@register('lainr_inr_ddp')
class LAINR(torch.nn.Module):
    
    def __init__(self, tokenizer, hyponet, transformer_encoder, num_lp = 256):
        super().__init__()

        self.dim = transformer_encoder['args']['dim'] #replace with mamba_encoder
        self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet, args={'hidden_dim': self.dim})
    
        self.transformer_encoder = models.make(transformer_encoder) #replace w mamba
        self.input_len = self.tokenizer.n_patches
        #self.type = type
        #self.n_group = n_group
        self.lps = nn.Parameter(torch.randn(num_lp, self.dim))
        #self.lp_idxs = None
        #self.set_lp_idxs(self.input_len, type = self.type, n = self.n_group)
        #self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    '''def set_lp_idxs(self, seq_len, type = 'equidistant', n = 1):
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
        
        return perm'''

    def forward(self, data, coord):
        data = data.float()
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = torch.cat([dtokens, lps], dim=1)
        trans_out = self.transformer_encoder(all_tokens)
        trans_out = trans_out[:, -len(self.lps):, :]
        pred = self.hyponet(coord, trans_out, biased = False)

        #self.hyponet.set_params(params)
        return pred
