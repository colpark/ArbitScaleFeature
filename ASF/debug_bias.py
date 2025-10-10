# Debug script to understand the bias dimension issue

import torch
import math
from einops import repeat

# Simulate the scenario
B = 64  # batch size
resolution = 32
num_tokens = 256  # LP tokens
num_queries = resolution * resolution  # 1024

# Compute what the bias SHOULD be
H_mod = W_mod = int(math.sqrt(num_queries))  # 32
print(f"Grid size: {H_mod}×{W_mod} = {num_queries} queries")

# Create fake indices
indexes = torch.arange(num_queries)  # (1024,)

# Simulate approximate_relative_distances
N = H_mod * W_mod  # 1024
t = indexes.float() / N  # (1024,)
token_positions = torch.tensor([(i + 0.5) / num_tokens for i in range(num_tokens)])  # (256,)

t_expanded = t.unsqueeze(0)  # (1, 1024)
tokens_expanded = token_positions.unsqueeze(1)  # (256, 1)

rel_distances = -10.0 * torch.abs(t_expanded - tokens_expanded)**2  # Should be (256, 1024)

print(f"\nrel_distances shape: {rel_distances.shape}")
print(f"Expected: ({num_tokens}, {num_queries})")

# Now repeat for batch
bias = repeat(rel_distances, 'l n -> b l n', b=B)
print(f"\nbias shape after repeat: {bias.shape}")
print(f"Expected: ({B}, {num_tokens}, {num_queries})")

# In attention, this gets transposed and repeated for heads
# Let's check what attention expects
heads = 2
sim_shape = (B * heads, num_queries, num_tokens)  # (q, k) after einsum
print(f"\nAttention sim shape: {sim_shape}")
print(f"  (batch*heads, num_queries, num_tokens)")

# The bias needs to be transposed!
# bias is (b, l, n) where l=tokens=256, n=queries=1024
# But sim is (b*h, queries, tokens) = (b*h, 1024, 256)
# So we need to transpose bias from (b, 256, 1024) to (b, 1024, 256)

print(f"\nPROBLEM IDENTIFIED:")
print(f"  bias shape: (b, {num_tokens}, {num_queries}) = (b, 256, 1024)")
print(f"  sim shape: (b*h, {num_queries}, {num_tokens}) = (b*h, 1024, 256)")
print(f"  After repeat to (b*h): (b*h, 256, 1024)")
print(f"  This tries to add (b*h, 256, 1024) to (b*h, 1024, 256) → MISMATCH!")

print(f"\nSOLUTION: Transpose bias before adding to sim")
print(f"  bias.transpose(-2, -1) gives (b, 1024, 256) ✓")
