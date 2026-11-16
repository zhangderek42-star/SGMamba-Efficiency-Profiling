
import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from .drop import DropPath
except ImportError:
    from drop import DropPath


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        hidden = max(1, num_feat // squeeze_factor)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):

    def __init__(
        self,
        num_feat: int,
        is_light_sr: bool = False,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
    ):
        super().__init__()

        if is_light_sr:
            compress_ratio = 2
            hidden = max(1, num_feat // compress_ratio)
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, hidden, 1, 1, 0),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
                nn.GELU(),
                nn.Conv2d(hidden, num_feat, 1, 1, 0),
                nn.Conv2d(
                    num_feat,
                    num_feat,
                    3,
                    1,
                    padding=2,
                    groups=num_feat,
                    dilation=2,
                ),
                ChannelAttention(num_feat, squeeze_factor),
            )
        else:
            hidden = max(1, num_feat // compress_ratio)
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, hidden, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(hidden, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SS2D(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(
                self.d_inner,
                self.dt_rank + 2 * d_state,
                bias=False,
                **factory_kwargs,
            )
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([m.weight for m in x_proj], dim=0)
        )

        dt_projs = [
            self._make_dt_proj(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([m.weight for m in dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([m.bias for m in dt_projs], dim=0)
        )

        self.A_logs = self._init_A(
            d_state, self.d_inner, copies=4, merge=True, device=device
        )
        self.Ds = self._init_D(
            self.d_inner, copies=4, merge=True, device=device
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _make_dt_proj(
        dt_rank,
        d_inner,
        dt_scale,
        dt_init,
        dt_min,
        dt_max,
        dt_init_floor,
        **factory_kwargs,
    ):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        init_std = (dt_rank ** -0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(proj.weight, init_std)
        elif dt_init == "random":
            nn.init.uniform_(proj.weight, -init_std, init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            proj.bias.copy_(inv_dt)
        proj.bias._no_reinit = True
        return proj

    @staticmethod
    def _init_A(d_state, d_inner, copies=1, merge=True, device=None):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _init_D(d_inner, copies=1, merge=True, device=None):
        D = torch.ones(d_inner, device=device)

        if copies > 1:
            D = repeat(D, "d -> r d", r=copies)
            if merge:
                D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _scan(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        row = x.reshape(B, C, L)
        col = x.transpose(2, 3).contiguous().reshape(B, C, L)

        seq = torch.stack((row, col), dim=1)
        seq = torch.cat((seq, torch.flip(seq, dims=[-1])), dim=1)

        projected = torch.einsum(
            "b k d l, k c d -> b k c l",
            seq,
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts,
            self.dt_projs_weight,
        )

        xs = seq.float().reshape(B, -1, L)
        dts = dts.contiguous().float().reshape(B, -1, L)
        Bs = Bs.float().reshape(B, K, -1, L)
        Cs = Cs.float().reshape(B, K, -1, L)

        As = -torch.exp(self.A_logs.float()).reshape(-1, self.d_state)
        Ds = self.Ds.float().reshape(-1)
        delta_bias = self.dt_projs_bias.float().reshape(-1)

        out = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        ).reshape(B, K, -1, L)

        inv = torch.flip(out[:, 2:4], dims=[-1])

        row_fwd = out[:, 0]
        row_rev = inv[:, 0]

        col_fwd = (
            out[:, 1]
            .reshape(B, -1, W, H)
            .transpose(2, 3)
            .contiguous()
            .reshape(B, -1, L)
        )
        col_rev = (
            inv[:, 1]
            .reshape(B, -1, W, H)
            .transpose(2, 3)
            .contiguous()
            .reshape(B, -1, L)
        )

        return row_fwd, row_rev, col_fwd, col_rev

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self._scan(x)
        y = y1 + y2 + y3 + y4

        y = y.transpose(1, 2).contiguous().reshape(B, H, W, self.d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)


class VSSBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        expand: float = 2.0,
        is_light_sr: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        self.conv_blk = CAB(hidden_dim, is_light_sr=is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        B, L, C = x.shape
        H, W = x_size

        if L != H * W:
            raise ValueError(
                f"Token length {L} does not match H*W={H*W}."
            )

        x = x.reshape(B, H, W, C)

        y = self.ln_1(x)
        x = x * self.skip_scale + self.drop_path(self.self_attention(y))

        y = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        y = self.conv_blk(y).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + y

        return x.reshape(B, H * W, C)


class VMamba(nn.Module):

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: float = 2.0,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        is_light_sr: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.block = VSSBlock(
            hidden_dim=dim,
            drop_path=drop_path,
            d_state=d_state,
            expand=expand,
            attn_drop_rate=attn_drop_rate,
            is_light_sr=is_light_sr,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"VMamba expects BCHW input, got shape {tuple(x.shape)}."
            )

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2).contiguous()

        tokens = self.block(tokens, (H, W))

        out = tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


__all__ = ["VMamba", "VSSBlock", "SS2D", "CAB", "ChannelAttention"]


if __name__ == "__main__":
    model = VMamba(dim=64)
    print(model.__class__.__name__)
