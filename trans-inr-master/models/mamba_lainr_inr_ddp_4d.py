import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import models
from models import register
from patchembed import PatchEmbed, MambaPatchTokenizer4D
from .cross_attention import CrossAttentionWithLearnedQueries


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

@register('mamba_lainr_inr_ddp_4d')
class MambaInr(torch.nn.Module):
    
    def __init__(self, input_size, patch_size, hyponet, mamba_encoder, num_lp = 128, type = 'equidistant', n_group = 1, latent_token_len = 64, patchembed = True, toptokens = False):
        super().__init__()

        self.dim = mamba_encoder['args']['dim'] #replace with mamba_encoder
        self.latent_token_len = latent_token_len
        #self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.hyponet = models.make(hyponet, args={'hidden_dim': self.dim})
        self.shape = [input_size[i]//patch_size[i] for i in range(len(input_size))]
        self.patchembed = patchembed
        if self.patchembed:
            self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim)
        else:
            print('using MambaPatchTokenizer4D')
            self.patchifier = MambaPatchTokenizer4D(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim)
        self.mamba_encoder = models.make(mamba_encoder) #replace w mamba
        self.input_len = self.patchifier.num_patches
        print(self.input_len)
        self.type = type
        if toptokens:
            self.num_lp = self.input_len//2
        else:
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
        all_data = data.float()# (B, C, Z, H, W, T)
        B = all_data.shape[0]
        dtokens = self.patchifier(all_data) #(B, pD, pH, pW, pT, D)
        if self.patchembed:
            dtokens = self.scan(dtokens) # (B, pD*pH*pW*pT, D)
        #dtokens = self.tokenizer(data)
        #Here, delete patches and keep track of which ones are deleted. Can create the pixel mask here too and return it
        #lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = self.add_lp(dtokens)
        #print(all_tokens.shape[1])
        mamba_out = self.mamba_encoder(all_tokens)
        mamba_out = self.extract_lp_tokens(mamba_out)
        pred = self.hyponet(coord, mamba_out, self.shape)

        return pred