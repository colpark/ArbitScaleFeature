import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from patchembed import PatchEmbed, MambaPatchTokenizer4D
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


@register('trans_inr_ddp_4d')
class TransInr(nn.Module):

    def __init__(self, input_size, patch_size, hyponet, n_groups, transformer_encoder):
        super().__init__()
        dim = transformer_encoder['args']['dim']
        self.patchifier = PatchEmbed(img_size = tuple(input_size), patch_size = tuple(patch_size), embed_dim = dim, pe_method = 'learned')
        self.hyponet = models.make(hyponet)
        self.transformer_encoder = models.make(transformer_encoder)

        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

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
        pred = self.hyponet(coord)
        return pred
