from models import register
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange, repeat
import ssl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from math import pi, log
from functools import wraps
import torch.nn.functional as F


# Fix for torchvision dataset download issue
ssl._create_default_https_context = ssl._create_unverified_context

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

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, context_dim, heads=heads, dim_head=dim_head), context_dim=context_dim)
        self.ff = PreNorm(dim, FeedForward(dim, mult=ff_mult))

    def forward(self, x, context):
        x = x + self.attn(x, context=context)  
        x = x + self.ff(x)                     
        return x

@register('cross_attention_lq')
class CrossAttentionWithLearnedQueries(nn.Module):
    def __init__(self, dim, num_queries, context_dim=None, heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.learned_queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.xavier_normal_(self.learned_queries)
        self.block = CrossAttentionBlock(dim, context_dim, heads, dim_head, ff_mult)

    def forward(self, context):
        B = context.shape[0]
        queries = repeat(self.learned_queries, 'n d -> b n d', b=B)
        return self.block(queries, context)
