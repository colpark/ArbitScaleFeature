"""
Implicit Sequential Bias - Learnable Position Tokens (LPs)
Core Innovation of MAMBA-GINR

This module implements the implicit sequential bias mechanism through learnable
position tokens that are strategically interleaved with input tokens to provide
positional information without explicit positional encodings.

Key Innovation:
- Learnable tokens inserted at specific positions in the sequence
- Multiple placement strategies: equidistant, middle, n_group
- Acts as implicit sequential bias for the Mamba encoder
- Enables arbitrary-scale generation through position-aware representations
"""

import torch
import torch.nn as nn
import numpy as np
import einops


class ImplicitSequentialBias(nn.Module):
    """
    Implicit Sequential Bias via Learnable Position Tokens

    This mechanism introduces learnable position tokens (LPs) that are interleaved
    with input tokens to provide implicit positional information to the model.
    Unlike traditional positional encodings, these tokens are:
    1. Learnable through backpropagation
    2. Strategically placed within the sequence
    3. Extracted after encoding to serve as conditioning information

    Args:
        num_lp (int): Number of learnable position tokens
        dim (int): Dimension of each token
        input_len (int): Length of input sequence (number of patches)
        type (str): Placement strategy ('equidistant', 'middle', 'n_group')
        n_group (int): Group size for 'n_group' placement strategy
    """

    def __init__(self, num_lp=128, dim=512, input_len=None, type='equidistant', n_group=1):
        super().__init__()

        self.num_lp = num_lp
        self.dim = dim
        self.input_len = input_len
        self.type = type
        self.n_group = n_group

        # Initialize learnable position tokens
        self.lps = nn.Parameter(torch.randn(self.num_lp, self.dim))

        # Compute placement indices and permutation
        self.lp_idxs = None
        self.perm = None
        if input_len is not None:
            self.set_lp_idxs(input_len, type=type, n=n_group)
            self.perm = self.compute_interleave_permutation(input_len, num_lp)

    def set_lp_idxs(self, seq_len, type='equidistant', n=1):
        """
        Compute indices where learnable position tokens will be inserted.

        Strategies:
        - 'equidistant': Evenly spaced throughout sequence
        - 'middle': Clustered in the middle of sequence
        - 'n_group': Grouped insertions at evenly spaced locations

        Args:
            seq_len (int): Length of input sequence
            type (str): Placement strategy
            n (int): Group size for 'n_group' strategy
        """
        total_len = seq_len + self.num_lp

        if type == 'equidistant':
            # Evenly distribute LPs across the sequence
            insert_idxs = torch.linspace(0, total_len - 1, steps=self.num_lp).long()
            self.lp_idxs = insert_idxs

        elif type == 'middle':
            # Place all LPs in the middle of sequence
            insert_idxs = (np.array(range(self.num_lp)) + (seq_len // 2)).tolist()
            self.lp_idxs = insert_idxs

        elif type == 'n_group':
            # Place LPs in groups at evenly spaced locations
            if self.num_lp % n != 0:
                raise ValueError("n must divide number of lps evenly")
            insert_idxs = []
            pre_idxs = torch.linspace(0, total_len - n, steps=self.num_lp // n).long()
            for idx in pre_idxs:
                insert_idxs.extend([idx + i for i in range(n)])
            self.lp_idxs = insert_idxs

    def compute_interleave_permutation(self, seq_len, n_insert):
        """
        Compute permutation for interleaving LPs with input tokens.

        Args:
            seq_len (int): Length of input sequence
            n_insert (int): Number of tokens to insert

        Returns:
            torch.Tensor: Permutation indices
        """
        total_len = seq_len + n_insert
        insert_idxs = torch.linspace(0, total_len - 1, steps=n_insert).long()
        perm = torch.full((total_len,), -1, dtype=torch.long)

        # Place LP indices
        perm[insert_idxs] = torch.arange(seq_len, seq_len + n_insert)

        # Place input token indices
        input_token_ids = torch.arange(seq_len)
        perm[perm == -1] = input_token_ids

        return perm

    def add_lp(self, x):
        """
        Add learnable position tokens to input sequence.

        Args:
            x: Input tensor of shape (B, L, D)

        Returns:
            Interleaved tensor of shape (B, L + num_lp, D)
        """
        B, L, D = x.shape

        # Repeat LPs for batch
        w = einops.repeat(self.lps, 'n d -> b n d', b=B)  # (B, num_lp, D)

        # Concatenate input and LPs
        x_full = torch.cat([x, w], dim=1)  # (B, L + num_lp, D)

        # Apply interleaving permutation
        x_perm = x_full[:, self.perm]  # (B, L + num_lp, D) — interleaved

        return x_perm

    def extract_lp_tokens(self, x):
        """
        Extract only the learnable position tokens from encoded sequence.

        Args:
            x: Encoded tensor of shape (B, L + num_lp, D)

        Returns:
            LP tokens of shape (B, num_lp, D)
        """
        return x[:, self.lp_idxs]

    def forward(self, x):
        """
        Full forward pass: add LPs, then prepare for extraction.

        Args:
            x: Input tensor (B, L, D)

        Returns:
            Interleaved tensor (B, L + num_lp, D)
        """
        return self.add_lp(x)


class SequentialBiasVisualization:
    """
    Utility class for visualizing LP placement strategies
    """

    @staticmethod
    def visualize_placement(seq_len=100, num_lp=10, type='equidistant', n_group=2):
        """
        Visualize how LPs are placed within a sequence.

        Args:
            seq_len (int): Input sequence length
            num_lp (int): Number of learnable position tokens
            type (str): Placement strategy
            n_group (int): Group size for 'n_group'
        """
        bias_module = ImplicitSequentialBias(
            num_lp=num_lp,
            dim=1,
            input_len=seq_len,
            type=type,
            n_group=n_group
        )

        # Create visualization array
        viz = ['_'] * (seq_len + num_lp)
        for idx in bias_module.lp_idxs:
            viz[idx] = 'L'

        # Apply permutation
        viz_perm = [viz[i] for i in bias_module.perm.tolist()]

        print(f"\nPlacement Strategy: {type}")
        print(f"Sequence Length: {seq_len}, LP Count: {num_lp}")
        print(f"Pattern (L=LP, _=Input): {''.join(viz_perm)}")
        print(f"LP Indices: {bias_module.lp_idxs.tolist()}")


if __name__ == "__main__":
    # Demonstrate different placement strategies
    print("=" * 80)
    print("Implicit Sequential Bias - Learnable Position Token Placement")
    print("=" * 80)

    SequentialBiasVisualization.visualize_placement(
        seq_len=50, num_lp=10, type='equidistant'
    )

    SequentialBiasVisualization.visualize_placement(
        seq_len=50, num_lp=10, type='middle'
    )

    SequentialBiasVisualization.visualize_placement(
        seq_len=50, num_lp=10, type='n_group', n_group=2
    )
