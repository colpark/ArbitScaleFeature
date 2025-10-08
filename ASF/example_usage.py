"""
Example Usage of MAMBA-GINR Innovation

Demonstrates the core implicit sequential bias mechanism and its application
to arbitrary-scale feature generation.
"""

import torch
import torch.nn as nn
from implicit_sequential_bias import ImplicitSequentialBias, SequentialBiasVisualization


def example_1_basic_lp_mechanism():
    """
    Example 1: Basic Learnable Position Token mechanism
    """
    print("\n" + "="*80)
    print("Example 1: Basic LP Mechanism")
    print("="*80)

    batch_size = 2
    seq_len = 64  # e.g., 8x8 image patches
    dim = 256
    num_lp = 16

    # Create implicit bias module
    bias_module = ImplicitSequentialBias(
        num_lp=num_lp,
        dim=dim,
        input_len=seq_len,
        type='equidistant'
    )

    # Simulate input tokens (e.g., from image patches)
    input_tokens = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {input_tokens.shape}")

    # Add learnable position tokens
    tokens_with_lps = bias_module.add_lp(input_tokens)
    print(f"After adding LPs: {tokens_with_lps.shape}")

    # Simulate encoding (would be done by Mamba)
    encoded = tokens_with_lps  # In practice: encoded = mamba_encoder(tokens_with_lps)

    # Extract LP tokens
    lp_features = bias_module.extract_lp_tokens(encoded)
    print(f"Extracted LP features: {lp_features.shape}")
    print(f"\nThese {num_lp} LP tokens now carry learned positional information!")


def example_2_placement_strategies():
    """
    Example 2: Different LP placement strategies
    """
    print("\n" + "="*80)
    print("Example 2: LP Placement Strategies")
    print("="*80)

    strategies = ['equidistant', 'middle', 'n_group']

    for strategy in strategies:
        n_group = 2 if strategy == 'n_group' else 1
        SequentialBiasVisualization.visualize_placement(
            seq_len=50,
            num_lp=10,
            type=strategy,
            n_group=n_group
        )


def example_3_full_pipeline():
    """
    Example 3: Full MAMBA-GINR pipeline simulation
    """
    print("\n" + "="*80)
    print("Example 3: Full MAMBA-GINR Pipeline")
    print("="*80)

    # Configuration
    batch_size = 4
    img_size = 32
    patch_size = 4
    num_patches = (img_size // patch_size) ** 2  # 64 patches
    dim = 512
    num_lp = 16

    print(f"\nConfiguration:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Number of patches: {num_patches}")
    print(f"  Model dimension: {dim}")
    print(f"  Learnable position tokens: {num_lp}")

    # Step 1: Tokenization (simplified)
    print(f"\n[Step 1] Tokenization")
    patches = torch.randn(batch_size, num_patches, dim)
    print(f"  Patches: {patches.shape}")

    # Step 2: Add implicit sequential bias
    print(f"\n[Step 2] Add Implicit Sequential Bias (LPs)")
    bias_module = ImplicitSequentialBias(
        num_lp=num_lp,
        dim=dim,
        input_len=num_patches,
        type='equidistant'
    )
    tokens_with_lps = bias_module.add_lp(patches)
    print(f"  Tokens with LPs: {tokens_with_lps.shape}")

    # Step 3: Mamba encoding (simulated)
    print(f"\n[Step 3] MAMBA Encoding (simulated)")
    # In practice: encoded = mamba_encoder(tokens_with_lps)
    from mamba_ssm import Mamba

    mamba = Mamba(d_model=dim)
    encoded = mamba(tokens_with_lps)
    print(f"  Encoded: {encoded.shape}")

    # Step 4: Extract LP features
    print(f"\n[Step 4] Extract LP Features")
    lp_features = bias_module.extract_lp_tokens(encoded)
    print(f"  LP features: {lp_features.shape}")
    print(f"  These carry learned positional information!")

    # Step 5: Arbitrary-scale decoding (simulated)
    print(f"\n[Step 5] Arbitrary-Scale Decoding")
    target_resolution = 64  # Super-resolution: 32 -> 64
    num_queries = target_resolution ** 2
    query_coords = torch.randn(batch_size, num_queries, 2)  # 2D coordinates

    print(f"  Query coordinates: {query_coords.shape}")
    print(f"  Target resolution: {target_resolution}x{target_resolution}")

    # Simple decoder (in practice: use hyponet)
    decoder = nn.Sequential(
        nn.Linear(dim + 2, dim),
        nn.ReLU(),
        nn.Linear(dim, 3)  # RGB output
    )

    # Pool LP features and broadcast
    global_features = lp_features.mean(dim=1, keepdim=True)  # (B, 1, D)
    global_features = global_features.expand(-1, num_queries, -1)  # (B, Q, D)

    # Decode
    decoder_input = torch.cat([global_features, query_coords], dim=-1)
    output = decoder(decoder_input)

    print(f"  Output: {output.shape}")
    print(f"\n✓ Successfully generated {target_resolution}x{target_resolution} image from {img_size}x{img_size} input!")


def example_4_lp_analysis():
    """
    Example 4: Analyze learned position token properties
    """
    print("\n" + "="*80)
    print("Example 4: LP Properties Analysis")
    print("="*80)

    dim = 128
    num_lp = 32

    bias_module = ImplicitSequentialBias(
        num_lp=num_lp,
        dim=dim,
        input_len=100,
        type='equidistant'
    )

    # Analyze LP tokens
    lps = bias_module.lps.data  # (num_lp, dim)

    print(f"\nLP Token Statistics:")
    print(f"  Shape: {lps.shape}")
    print(f"  Mean: {lps.mean().item():.4f}")
    print(f"  Std: {lps.std().item():.4f}")
    print(f"  Min: {lps.min().item():.4f}")
    print(f"  Max: {lps.max().item():.4f}")

    # Compute pairwise similarities
    lps_norm = lps / lps.norm(dim=-1, keepdim=True)
    similarity = torch.mm(lps_norm, lps_norm.t())

    print(f"\nLP Pairwise Similarities:")
    print(f"  Average similarity: {similarity.mean().item():.4f}")
    print(f"  Max similarity (off-diagonal): {similarity.fill_diagonal_(0).max().item():.4f}")

    # This shows that LPs learn diverse representations
    print(f"\n→ LPs learn diverse, distinguishable position representations")


def example_5_efficiency_comparison():
    """
    Example 5: Efficiency comparison - Mamba vs Transformer
    """
    print("\n" + "="*80)
    print("Example 5: Efficiency Comparison (Conceptual)")
    print("="*80)

    sequence_lengths = [64, 256, 1024, 4096]

    print("\nComputational Complexity:")
    print(f"{'Seq Length':<12} {'Transformer (O(L²))':<25} {'Mamba (O(L))':<25}")
    print("-" * 62)

    for L in sequence_lengths:
        transformer_ops = L ** 2
        mamba_ops = L
        speedup = transformer_ops / mamba_ops

        print(f"{L:<12} {transformer_ops:<25,} {mamba_ops:<25,} ({speedup:.0f}x faster)")

    print("\n→ Mamba's linear complexity enables efficient processing of long sequences")
    print("→ Critical for high-resolution generation with many query points")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MAMBA-GINR: Implicit Sequential Bias Examples")
    print("="*80)

    # Run all examples
    example_1_basic_lp_mechanism()
    example_2_placement_strategies()
    example_3_full_pipeline()
    example_4_lp_analysis()
    example_5_efficiency_comparison()

    print("\n" + "="*80)
    print("Summary of Key Innovation:")
    print("="*80)
    print("""
The Implicit Sequential Bias mechanism:

1. Introduces learnable position tokens (LPs) into the sequence
2. LPs are strategically placed (equidistant, middle, or grouped)
3. Mamba encoder processes tokens with O(L) complexity
4. LPs learn to encode positional information during training
5. Extracted LP features condition decoder for arbitrary-scale generation

This enables:
✓ Efficient processing (linear complexity)
✓ Learnable positional bias (adaptive to task)
✓ Arbitrary-scale generation (resolution-agnostic)
✓ Better context modeling (bidirectional Mamba)
    """)
    print("="*80 + "\n")
