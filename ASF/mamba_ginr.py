"""
MAMBA-GINR: Mamba-based Generalized Implicit Neural Representation
Core Architecture Implementation

This module combines:
1. MAMBA encoder with bidirectional state space modeling
2. Implicit sequential bias via learnable position tokens
3. Continuous decoder (hyponet) for arbitrary-scale generation

The key innovation is the use of learnable position tokens (LPs) that provide
implicit sequential bias, enabling the model to generate continuous representations
at arbitrary resolutions.
"""

import torch
import torch.nn as nn
import einops
import numpy as np


class MambaGINR(nn.Module):
    """
    MAMBA-GINR: Complete architecture for implicit neural representations

    Architecture Flow:
    1. Input Tokenization: Convert input to patches/tokens
    2. LP Insertion: Add learnable position tokens (implicit sequential bias)
    3. MAMBA Encoding: Bidirectional state space model processing
    4. LP Extraction: Extract learned position representations
    5. Hyponet Decoding: Generate continuous output at arbitrary coordinates

    Args:
        tokenizer (nn.Module): Module to convert input to tokens
        hyponet (nn.Module): Decoder network for continuous representation
        mamba_encoder (nn.Module): MAMBA encoder module
        num_lp (int): Number of learnable position tokens
        type (str): LP placement strategy ('equidistant', 'middle', 'n_group')
        n_group (int): Group size for 'n_group' placement
        latent_token_len (int): Length of latent token sequence (for future use)
    """

    def __init__(
        self,
        tokenizer,
        hyponet,
        mamba_encoder,
        num_lp=128,
        type='equidistant',
        n_group=1,
        latent_token_len=64
    ):
        super().__init__()

        # Encoder dimension
        self.dim = mamba_encoder.blocks[0].mixer.f_mamba.d_model

        # Model components
        self.tokenizer = tokenizer
        self.hyponet = hyponet
        self.mamba_encoder = mamba_encoder

        # Implicit sequential bias parameters
        self.latent_token_len = latent_token_len
        self.input_len = self.tokenizer.n_patches if hasattr(tokenizer, 'n_patches') else None
        self.type = type
        self.num_lp = num_lp
        self.n_group = n_group

        # Initialize learnable position tokens
        self.lps = nn.Parameter(torch.randn(self.num_lp, self.dim))

        # Compute LP placement strategy
        self.lp_idxs = None
        if self.input_len is not None:
            self.set_lp_idxs(self.input_len, type=self.type, n=self.n_group)
            self.perm = self.compute_interleave_permutation(self.input_len, self.num_lp)

    def set_lp_idxs(self, seq_len, type='equidistant', n=1):
        """
        Set indices for learnable position token placement.

        Args:
            seq_len (int): Input sequence length
            type (str): Placement strategy
            n (int): Group size for n_group strategy
        """
        total_len = seq_len + self.num_lp

        if type == 'equidistant':
            # Evenly distribute across sequence
            insert_idxs = torch.linspace(0, total_len - 1, steps=self.num_lp).long()
            self.lp_idxs = insert_idxs

        elif type == 'middle':
            # Place in middle of sequence
            insert_idxs = (np.array(range(self.num_lp)) + (seq_len // 2)).tolist()
            self.lp_idxs = insert_idxs

        elif type == 'n_group':
            # Group placement
            if self.num_lp % n != 0:
                raise Exception("n must divide number of lps evenly")
            insert_idxs = []
            pre_idxs = torch.linspace(0, total_len - n, steps=self.num_lp // n).long()
            for idx in pre_idxs:
                insert_idxs.extend([idx + i for i in range(n)])
            self.lp_idxs = insert_idxs

    def compute_interleave_permutation(self, seq_len, n_insert):
        """
        Compute permutation for interleaving LPs with input tokens.

        Args:
            seq_len (int): Input sequence length
            n_insert (int): Number of LPs to insert

        Returns:
            torch.Tensor: Permutation indices
        """
        total_len = seq_len + n_insert
        insert_idxs = torch.linspace(0, total_len - 1, steps=n_insert).long()
        token_ids = torch.arange(seq_len + n_insert)
        perm = torch.full((total_len,), -1, dtype=torch.long)

        # Assign LP positions
        perm[insert_idxs] = torch.arange(seq_len, seq_len + n_insert)

        # Assign input token positions
        input_token_ids = torch.arange(seq_len)
        perm[perm == -1] = input_token_ids

        return perm

    def add_lp(self, x):
        """
        Add learnable position tokens to input sequence.

        Args:
            x: Input tokens (B, L, D)

        Returns:
            Interleaved sequence (B, L + num_lp, D)
        """
        B, L, D = x.shape
        w = einops.repeat(self.lps, 'n d -> b n d', b=B)  # (B, num_lp, D)
        x_full = torch.cat([x, w], dim=1)  # (B, L + num_lp, D)
        x_perm = x_full[:, self.perm]  # (B, L + num_lp, D) — interleaved

        return x_perm

    def extract_lp_tokens(self, x):
        """
        Extract learnable position tokens from encoded sequence.

        Args:
            x: Encoded sequence (B, L + num_lp, D)

        Returns:
            LP tokens (B, num_lp, D)
        """
        return x[:, self.lp_idxs]

    def forward(self, data, coord):
        """
        Forward pass of MAMBA-GINR.

        Args:
            data: Input data (e.g., image, volume)
            coord: Query coordinates for continuous representation

        Returns:
            Predicted values at query coordinates
        """
        # Step 1: Tokenize input
        dtokens = self.tokenizer(data)  # (B, L, D)
        B = dtokens.shape[0]

        # Step 2: Add learnable position tokens (implicit sequential bias)
        all_tokens = self.add_lp(dtokens)  # (B, L + num_lp, D)

        # Step 3: MAMBA encoding
        mamba_out = self.mamba_encoder(all_tokens)  # (B, L + num_lp, D)

        # Step 4: Extract LP tokens (these carry positional information)
        mamba_out = self.extract_lp_tokens(mamba_out)  # (B, num_lp, D)

        # Step 5: Decode to continuous representation
        pred = self.hyponet(coord, mamba_out)  # (B, ..., output_dim)

        return pred

    def get_architecture_summary(self):
        """
        Print architecture summary.
        """
        print("=" * 80)
        print("MAMBA-GINR Architecture Summary")
        print("=" * 80)
        print(f"Encoder Dimension: {self.dim}")
        print(f"Number of Learnable Position Tokens: {self.num_lp}")
        print(f"LP Placement Strategy: {self.type}")
        if self.type == 'n_group':
            print(f"Group Size: {self.n_group}")
        print(f"Input Sequence Length: {self.input_len}")
        print(f"Total Sequence Length: {self.input_len + self.num_lp if self.input_len else 'Variable'}")
        print("\nComponents:")
        print(f"  - Tokenizer: {self.tokenizer.__class__.__name__}")
        print(f"  - MAMBA Encoder: {len(self.mamba_encoder.blocks)} blocks")
        print(f"  - Hyponet: {self.hyponet.__class__.__name__}")
        print("=" * 80)


class SimpleMambaGINR(nn.Module):
    """
    Simplified MAMBA-GINR for demonstration purposes.

    This version shows the core innovation clearly without complex dependencies.
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=512,
        output_dim=3,
        num_lp=64,
        mamba_depth=6,
        lp_type='equidistant'
    ):
        super().__init__()

        from mamba_ssm import Mamba

        self.hidden_dim = hidden_dim
        self.num_lp = num_lp

        # Simple linear tokenizer
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable position tokens
        self.lps = nn.Parameter(torch.randn(num_lp, hidden_dim))

        # MAMBA encoder
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim) for _ in range(mamba_depth)
        ])

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.lp_type = lp_type

    def forward(self, x, coords):
        """
        Simplified forward pass.

        Args:
            x: Input features (B, N, input_dim)
            coords: Query coordinates (B, M, coord_dim)

        Returns:
            Output predictions (B, M, output_dim)
        """
        B = x.shape[0]

        # Tokenize
        tokens = self.input_proj(x)  # (B, N, D)

        # Add LPs (simplified - just prepend)
        lps = self.lps.unsqueeze(0).expand(B, -1, -1)  # (B, num_lp, D)
        tokens_with_lps = torch.cat([lps, tokens], dim=1)  # (B, num_lp + N, D)

        # MAMBA encoding
        x_enc = tokens_with_lps
        for mamba in self.mamba_layers:
            x_enc = mamba(x_enc) + x_enc  # Residual connection

        # Extract LP tokens
        lp_features = x_enc[:, :self.num_lp, :]  # (B, num_lp, D)

        # Pool LP features
        global_features = lp_features.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Decode
        M = coords.shape[1]
        global_features_expanded = global_features.expand(-1, M, -1)  # (B, M, D)
        decoder_input = torch.cat([global_features_expanded, coords], dim=-1)
        output = self.decoder(decoder_input)

        return output


if __name__ == "__main__":
    print("MAMBA-GINR Core Innovation: Implicit Sequential Bias via Learnable Position Tokens")
    print("\nKey Innovation Points:")
    print("1. Learnable position tokens (LPs) inserted into input sequence")
    print("2. Strategic placement (equidistant/middle/grouped) provides positional bias")
    print("3. LPs are updated through backpropagation to encode optimal positional info")
    print("4. After encoding, LPs serve as conditioning for continuous decoder")
    print("5. Enables arbitrary-scale generation without explicit positional encodings")
