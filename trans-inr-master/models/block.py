import torch
from torch import nn, Tensor
from typing import Optional

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
except ImportError:
    RMSNorm = None
    layer_norm_fn = None


class QuadBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states_tuple: tuple, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs):
        x_h, x_v = hidden_states_tuple

        # Apply normalization and residual logic
        if not self.fused_add_norm:
            residual_h = (x_h + residual) if residual is not None else x_h
            residual_v = (x_v + residual) if residual is not None else x_v
            x_h = self.norm(residual_h.to(dtype=self.norm.weight.dtype))
            x_v = self.norm(residual_v.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual_h = residual_h.to(torch.float32)
                residual_v = residual_v.to(torch.float32)
        else:
            x_h, residual_h = layer_norm_fn(
                x_h, self.norm.weight, self.norm.bias, residual=residual,
                prenorm=True, residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm)
            )
            x_v, residual_v = layer_norm_fn(
                x_v, self.norm.weight, self.norm.bias, residual=residual,
                prenorm=True, residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        # Apply quad-directional mixer
        x_h, x_v = self.mixer((x_h, x_v), inference_params=inference_params, **mixer_kwargs)

        # Apply MLP
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual_h = x_h + residual_h
                residual_v = x_v + residual_v
                x_h = self.norm2(residual_h.to(dtype=self.norm2.weight.dtype))
                x_v = self.norm2(residual_v.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual_h = residual_h.to(torch.float32)
                    residual_v = residual_v.to(torch.float32)
            else:
                x_h, residual_h = layer_norm_fn(
                    x_h, self.norm2.weight, self.norm2.bias, residual=residual_h,
                    prenorm=True, residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
                x_v, residual_v = layer_norm_fn(
                    x_v, self.norm2.weight, self.norm2.bias, residual=residual_v,
                    prenorm=True, residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            x_h = self.mlp(x_h)
            x_v = self.mlp(x_v)

        return (x_h, x_v), residual_h  # or return both residual_h and residual_v if you want separate ones
