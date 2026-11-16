import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from ntuple import to_2tuple
from weight_init import trunc_normal_


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class MSGM(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.in_proj = nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0)

        self.conv3x3 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features)

        self.conv5x5 = nn.Conv2d(in_features, in_features, kernel_size=5, stride=1, padding=2, groups=in_features)

        self.out_proj = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.in_proj(x)
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x = x1 * F.gelu(x2)
        x = self.out_proj(x)
        x = x.view(B, -1, H*W).permute(0,2,1).contiguous()

        return x

class AHFRM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lf_in_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.extract_hf = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.fuse = nn.Conv2d(dim*2, dim, 1, 1, 0)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 6, kernel_size=1),
            nn.BatchNorm2d(dim // 6),
            nn.GELU(),
            nn.Conv2d(dim // 6, dim, kernel_size=1),
        )

    def forward(self, x_hf, x_lf, x_size):
        assert x_hf.shape == x_lf.shape, "x_hf and x_lf must have the same shape"
        B, L, C = x_hf.shape
        H, W = x_size
        x_hf = x_hf.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x_lf_in = x_lf.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x_lf = self.lf_in_proj(x_lf_in)

        x_hf_down = self.down(x_hf)
        x_hf = x_hf - F.interpolate(x_hf_down, size=x_size, mode='bilinear', align_corners=True)
        x_hf = self.extract_hf(x_hf)

        x_lf_down = self.down(x_lf)
        x_broken_hf = x_lf - F.interpolate(x_lf_down, size=x_size, mode='bilinear', align_corners=True)

        restored_hf = torch.cat([x_hf, x_broken_hf], dim=1)
        restored_hf = self.fuse(restored_hf)
        
        x = x_lf_in + self.scale(x_lf_in) * restored_hf
        x = self.out_proj(x).view(B, -1, H*W).permute(0,2,1).contiguous()

        return x
    
class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

def scan_image(x: torch.Tensor,
               window_size: int,
               mode: str = 'horizontal') -> torch.Tensor:
    b, h, w, c = x.shape

    if mode == 'horizontal':
        x2 = x
    elif mode == 'vertical':
        x2 = x.permute(0, 2, 1, 3)
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

    windows = window_partition(x2, window_size)
    rows = x2.shape[1] // window_size
    cols = x2.shape[2] // window_size
    num_windows = rows * cols

    windows = windows.view(b, num_windows, window_size, window_size, c)

    seq = windows.reshape(b, -1, c)  
    return seq


def merge_scanned(seq: torch.Tensor,
                  window_size: int,
                  h: int,
                  w: int,
                  mode: str = 'horizontal') -> torch.Tensor:
    b, L, c = seq.shape

    if mode == 'horizontal':
        H2, W2 = h, w
    else:
        H2, W2 = w, h
    rows = H2 // window_size
    cols = W2 // window_size
    num_windows = rows * cols

    windows = seq.view(b, num_windows, window_size, window_size, c)

    windows = windows.reshape(-1, window_size, window_size, c)

    x2 = window_reverse(windows, window_size, H2, W2)

    if mode == 'vertical':
        x_rec = x2.permute(0, 2, 1, 3)
    else:
        x_rec = x2

    return x_rec


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


class WISSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, mode, reverse, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden//2, d_state=self.d_state, expand=1, mode=mode, reverse=reverse)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )
     
    def forward(self, x, x_size, token=None):
        B, n, C = x.shape
        H, W = x_size

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)

        y = self.selectiveScan(x, x_size)
        y = self.out_proj(self.out_norm(y))

        return y


class MGSSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan_last(d_model=hidden//4, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )
     
    def forward(self, x, x_size, token=None):
        B, n, C = x.shape
        H, W = x_size

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)

        y = self.selectiveScan(x, x_size)
        y = self.out_proj(self.out_norm(y))

        return y
    
class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            mode="horizontal",
            reverse=True,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        assert mode == 'horizontal' or mode == 'vertical', "scan mode should be 'horizontal' or 'vertical'"
        self.scan_mode = mode
        self.scan_win_size = [16, 32]
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 8) if dt_rank == "auto" else dt_rank
        self.reverse = reverse

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
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
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, size):
        B, L, C2 = x.shape
        H, W = size
        assert L == H * W, "selective scan: input feature has wrong size"
        C = C2 // 2
        K = 2  
        x = x.reshape(B, H, W, C2)
        x_ws, x_wl = torch.chunk(x, chunks=2, dim=-1)
        x_ws = scan_image(x_ws, self.scan_win_size[0], self.scan_mode)
        x_wl = scan_image(x_wl, self.scan_win_size[1], self.scan_mode)
        if self.reverse:
            x_wl = torch.flip(x_wl, dims=[1])
        else:
            x_ws = torch.flip(x_ws, dims=[1])

        xs = torch.cat([x_ws, x_wl], dim=-1)

        xs = xs.permute(0, 2, 1).view(B, 2, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        x_ws, x_wl = out_y[:, 0].permute(0, 2, 1).contiguous(), out_y[:, 1].permute(0, 2, 1).contiguous()
        if self.reverse:
            x_wl = torch.flip(x_wl, dims=[1])
        else:
            x_ws = torch.flip(x_ws, dims=[1])
        x_ws = merge_scanned(x_ws, self.scan_win_size[0], H, W, self.scan_mode)
        x_wl = merge_scanned(x_wl, self.scan_win_size[1], H, W, self.scan_mode)

        return x_ws.reshape(B, -1, C), x_wl.reshape(B, -1, C)

    def forward(self, x: torch.Tensor, size, **kwargs):
        B, L, C2 = x.shape
        H, W = size
        x = x.reshape(B, H, W, C2)

        h_ori, w_ori = x.size()[1], x.size()[2]
        mod = 32
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [1])], 1)[:, :h, :, :]
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :w, :]
        x = x.reshape(B, h*w, C2)

        y_ws, y_wl = self.forward_core(x, (h, w))
        y = torch.cat([y_ws, y_wl], dim=-1)

        y = y.reshape(B, h, w, C2)
        y = y[:, :h_ori, :w_ori, :].reshape(B, L, C2)

        return y

class Selective_Scan_last(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.scan_win_size = [2, 4]
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
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
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, size):

        B, L, C = x.shape
        H, W = size
        assert L == H * W
        K = 4

        x_img = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        chunks = torch.chunk(x_img, 4, dim=1)

        C4 = C // 4
        h_fwd = chunks[0].view(B, C4, -1)
        v_fwd = chunks[1].permute(0, 1, 3, 2).contiguous().view(B, C4, -1)
        h_rev = torch.flip(chunks[2].view(B, C4, -1), dims=[2])
        v_rev = torch.flip(
            chunks[3].permute(0, 1, 3, 2).contiguous().view(B, C4, -1),
            dims=[2]
        )

        xs = torch.stack([h_fwd, v_fwd, h_rev, v_rev], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor, size, **kwargs):
        B, L, C = x.shape
        H, W = size
        assert L == H * W
        y1, y2, y3, y4 = self.forward_core(x, size)
        assert y1.dtype == torch.float32
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, L, C)
        return y

class T_WSMBlock(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 mode,
                 reverse,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = WISSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            mlp_ratio=mlp_ratio,
            mode=mode,
            reverse=reverse,
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)
        self.convffn2 =  GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)


    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x
    
class GSML(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 convffn_kernel_size,
                 mlp_ratio,
                 norm_layer=nn.LayerNorm,
                 is_last=True,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.assm = MGSSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            mlp_ratio=mlp_ratio,
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)


    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x_aca = self.assm(self.norm1(x), x_size) + x
        x = x_aca + self.convffn(self.norm2(x_aca), x_size)
        x = shortcut * self.scale + x

        return x

class WindowBlock(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 first_mode,
                 reverse, 
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mode = first_mode

        self.first_layer = T_WSMBlock(
                 dim=dim,
                 d_state=d_state,
                 input_resolution=input_resolution,
                 num_heads=num_heads,
                 window_size=window_size,
                 shift_size=0,
                 convffn_kernel_size=convffn_kernel_size,
                 mlp_ratio=mlp_ratio,
                 mode='horizontal' if first_mode == 'horizontal' else 'vertical',
                 reverse=reverse,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,)
        
        self.second_layer = T_WSMBlock(
                 dim=dim,
                 d_state=d_state,
                 input_resolution=input_resolution,
                 num_heads=num_heads,
                 window_size=window_size,
                 shift_size=window_size//2,
                 convffn_kernel_size=convffn_kernel_size,
                 mlp_ratio=mlp_ratio,
                 mode='vertical' if first_mode == 'horizontal' else 'horizontal',
                 reverse=reverse,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,)
        
    def forward(self, x, x_size, params):
        B, L, C = x.shape
        H, W = x_size
        assert L == H * W, "input feature has wrong size"

        x_1 = self.first_layer(x, x_size, params)
        x_2 = self.second_layer(x_1, x_size, params)

        return x_2

class BasicBlock(nn.Module):

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            first_mode = 'horizontal' if (i % 2 == 0) else 'vertical'
            reverse = True if (i % 2 == 0) else False
            self.layers.append(
                WindowBlock(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    first_mode=first_mode,
                    reverse=reverse, 
                )
            )

        self.layers.append(
            GSML(
                dim=dim,
                d_state=d_state,
                input_resolution=input_resolution,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
            ),
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.conv = MSGM(in_features=dim, out_features=dim)
        self.hf_restore = AHFRM(dim=dim)

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        x_hf = self.conv(x, x_size)
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.hf_restore(x_hf, x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class T_PMGroup(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv', ):
        super(T_PMGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class T_PMambaSR(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 out_chans=31,
                 embed_dim=48,
                 d_state=8,
                 depths=(2, 2, 2, 2,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 convffn_kernel_size=5,
                 mlp_ratio=1.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_range = img_range
        if in_chans == 3:
            input_mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        else:
            input_mean = torch.zeros(1, in_chans, 1, 1)
        output_mean = torch.zeros(1, out_chans, 1, 1)
        self.register_buffer('input_mean', input_mean, persistent=False)
        self.register_buffer('output_mean', output_mean, persistent=False)

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = T_PMGroup(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        x = (x - self.input_mean.type_as(x)) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        x_first = self.conv_first(x)
        features = self.conv_after_body(self.forward_features(x_first, params)) + x_first
        x = self.conv_last(features)

        x = x / self.img_range + self.output_mean.type_as(x)

        x = x[..., :h_ori, :w_ori]

        return x


def _build_test_model():
    return T_PMambaSR(
        img_size=64,
        in_chans=3,
        out_chans=31,
        embed_dim=48,
        d_state=8,
        depths=[2, 2, 2, 2],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        img_range=1.0,
        resi_connection="1conv",
    )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test standalone RGB-to-HSI T-PMambaSR.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example: cuda, cuda:0, or cpu.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "This T-PMambaSR implementation uses mamba_ssm.selective_scan_fn, "
            "which requires a CUDA device for the forward-pass test."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    model = _build_test_model().to(device).eval()
    image = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {parameters / 1e6:.3f} M")
    print(f"Input shape: {tuple(image.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Forward time: {elapsed_ms:.2f} ms")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"Peak allocated memory: {peak_memory:.1f} MiB")
