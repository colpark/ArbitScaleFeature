# ============================================================================
# FIX FOR CELL 3: Attention class forward method
# The bias dimension order is backwards - needs transpose
# ============================================================================

class Attention(nn.Module):
    """Multi-head attention (from SCENT)"""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, bias=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(bias):
            # bias comes in as (b, num_tokens, num_queries) but sim is (b*h, num_queries, num_tokens)
            # Need to transpose!
            if bias.dim() == 3 and bias.shape[0] == x.shape[0]:  # (b, l, n)
                bias = repeat(bias, 'b l n -> (b h) l n', h=h)  # (b*h, l, n)
                bias = bias.transpose(-2, -1)  # TRANSPOSE to (b*h, n, l)
            sim = sim + bias

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# ============================================================================
# OR JUST ADD ONE LINE IN YOUR EXISTING Cell 3:
#
# After line:
#     bias = repeat(bias, 'b l n -> (b h) l n', h=h)
#
# Add this line:
#     bias = bias.transpose(-2, -1)  # FIX: transpose to match sim dimensions
#
# Before:
#     sim = sim + bias
# ============================================================================
