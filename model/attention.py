import math
import os
from typing import Tuple
from itertools import repeat
import collections.abc
import torch
from torch import nn, Tensor
from model.layers import Mlp, modulate


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    B, L, S = query.size(0), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask


    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    attn_bias = attn_bias.unsqueeze(1)  # [b, seq, seq] -> [b, head, seq, seq]
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    # fused_attn: torch.jit.Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            use_fused_attn: bool = False,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                mask = mask.expand_as(attn)
                print(mask.shape)
                attn = attn.masked_fill_(mask == 0, -1e9)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, use_fused_attn=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossDiTBlock(nn.Module):

    def __init__(self, x_hidden_size, y_hidden_size, num_head, mlp_ratio=4.0):
        super().__init__()
        self.num_head = num_head
        self.norm1 = nn.LayerNorm(x_hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(y_hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(x_hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_1 = nn.MultiheadAttention(x_hidden_size, num_head, kdim=y_hidden_size, vdim=y_hidden_size,
                                                  dropout=0, batch_first=True)
        mlp_hidden_dim = int(x_hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=x_hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_hidden_size, 6 * x_hidden_size, bias=True)
        )
        self.adaLN_modulation_y = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_hidden_size, 6 * y_hidden_size, bias=True)
        )

    def forward(self, x, c, y, attn_mask=None):
        shift_mca_x, scale_mca_x, gate_mca_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_x(c).chunk(6, dim=-1)
        shift_mca_y, scale_mca_y, gate_mca_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_y(c).chunk(6, dim=-1)
        x_norm1 = modulate(self.norm1(x), shift_mca_x, scale_mca_x)
        y_norm2 = modulate(self.norm2(y), shift_mca_y, scale_mca_y)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_head, 1, 1)
        x = x + gate_mca_x.unsqueeze(1) * self.cross_attn_1(x_norm1, y_norm2, y_norm2, attn_mask=attn_mask,
                                                            key_padding_mask=None, need_weights=False)[0]
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp_x, scale_mlp_x))
        return x