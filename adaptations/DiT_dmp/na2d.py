from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from natten.context import is_fna_enabled
from natten.functional import na2d, na2d_av, na2d_qk
from natten.functional import na1d, na1d_av, na1d_qk
from natten.types import CausalArg2DTypeOrDed, Dimension2DTypeOrDed
from natten.types import CausalArg1DTypeOrDed, Dimension1DTypeOrDed
from natten.utils import check_all_args, log
import math

logger = log.get_logger(__name__)
class NeighborhoodAttention1D(nn.Module):
    """
    Neighborhood Attention 1D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension1DTypeOrDed,
        dilation: Dimension1DTypeOrDed = 1,
        is_causal: CausalArg1DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            1, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 1
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * self.kernel_size[0] - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, k_neighbors: int) -> Tensor:
        kernel_size = int(math.sqrt(k_neighbors)) + 1
        # if kernel_size == 17: kernel_size = 15
        if x.dim() != 3:
            raise ValueError(
                f"NeighborhoodAttention1D expected a rank-3 input tensor; got {x.dim()=}."
            )

        B, L, C = x.shape
        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )

            qkv = (
                self.qkv(x)
                .reshape(B, L, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 1, 3, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = na1d(
                q,
                k,
                v,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, L, C)

        else:
            qkv = (
                self.qkv(x)
                .reshape(B, L, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na1d_qk(
                q,
                k,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na1d_av(
                attn,
                v,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 1, 3).reshape(B, L, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )

class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension2DTypeOrDed,
        dilation: Dimension2DTypeOrDed = 1,
        is_causal: CausalArg2DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            2, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 2
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, k_neighbors: int) -> Tensor:
        kernel_size = int(math.sqrt(k_neighbors))
        if kernel_size == 8:
            kernel_size = 7
        elif kernel_size == 4:
            kernel_size = 5
        elif kernel_size == 2:
            kernel_size = 3
        else:
            raise f"Invalid kernel size: {kernel_size}"

        x = x.unflatten(1, (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))))
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape
        # import pdb; pdb.set_trace()

        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )

            qkv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 1, 2, 4, 5)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = na2d(
                q,
                k,
                v,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, H, W, C)

        else:
            qkv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 4, 1, 2, 5)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na2d_av(
                attn,
                v,
                kernel_size=kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        final = self.proj_drop(self.proj(x))

        return final.flatten(1,2)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )