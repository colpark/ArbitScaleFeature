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

@register('lainr_inr_ddp_4d')
class LAINR(torch.nn.Module):
    
    def __init__(self, input_size, patch_size, hyponet, transformer_encoder, num_lp = 256):
        super().__init__()

        self.dim = transformer_encoder['args']['dim'] #replace with mamba_encoder
        #self.tokenizer = models.make(tokenizer, args={'dim': self.dim}) #replace w correct tokenizer
        self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = self.dim, pe_method = 'learned')
        self.hyponet = models.make(hyponet, args={'hidden_dim': self.dim})
        self.shape = [input_size[i]//patch_size[i] for i in range(len(input_size))]
    
        self.transformer_encoder = models.make(transformer_encoder) #replace w mamba
        self.input_len = self.patchifier.num_patches
        self.lps = nn.Parameter(torch.randn(num_lp, self.dim))
        
    def scan(self, x):
        B, pD, pH, pW, pT, D = x.shape
        x = x.permute(0, 4, 1, 2, 3, 5)  # (B, pT, pD, pH, pW, D)
        #x = x.permute(0, 2, 3, 4, 1, 5)
    
        # Flatten spatial+time dims into one dimension
        x = x.reshape(B, pD*pH*pW*pT, D)
        return x

    def forward(self, data, coord):
        all_data = data.float()
        B = all_data.shape[0]
        dtokens = self.patchifier(all_data)
        dtokens = self.scan(dtokens)
        lps = einops.repeat(self.lps, 'n d -> b n d', b=B)
        all_tokens = torch.cat([dtokens, lps], dim=1)
        trans_out = self.transformer_encoder(all_tokens)
        trans_out = trans_out[:, -len(self.lps):, :]
        pred = self.hyponet(coord, trans_out, self.shape, biased = False)
        
        return pred
