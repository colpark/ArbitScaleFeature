"""
ASF - Arbitrary Scale Feature (MAMBA-GINR Innovation)

This package contains the isolated implementation of MAMBA-GINR's core innovation:
Implicit Sequential Bias via Learnable Position Tokens.

Key Components:
- mamba_encoder: Bidirectional Mamba state space model encoder
- implicit_sequential_bias: Learnable position token mechanism
- mamba_ginr: Complete MAMBA-GINR architecture

Innovation:
Instead of using fixed positional encodings, MAMBA-GINR uses learnable position
tokens (LPs) that are strategically interleaved with input tokens. These LPs
provide implicit sequential bias and enable arbitrary-scale generation.
"""

from .mamba_encoder import BiMamba, MambaEncoder
from .implicit_sequential_bias import ImplicitSequentialBias, SequentialBiasVisualization
from .mamba_ginr import MambaGINR, SimpleMambaGINR

__version__ = "1.0.0"
__all__ = [
    "BiMamba",
    "MambaEncoder",
    "ImplicitSequentialBias",
    "SequentialBiasVisualization",
    "MambaGINR",
    "SimpleMambaGINR",
]
