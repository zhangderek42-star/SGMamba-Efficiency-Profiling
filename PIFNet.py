import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.ops import DeformConv2d

from PIF_mamba import VMamba
class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dw_conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.pw_conv(x)
        return x * identity

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SEModule(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class FCNormAttnNorm(nn.Module):
    def __init__(self, dim, fc_ratio=1.0, norm_type='bn', lora_alpha=0.1):
        super().__init__()
        hidden_dim = int(dim * fc_ratio)
        self.alpha = lora_alpha

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        if norm_type == 'ln':
            self.norm1 = nn.GroupNorm(1, hidden_dim)
            self.norm2 = nn.GroupNorm(1, hidden_dim)
            self.norm3 = nn.GroupNorm(1, hidden_dim)
            self.norm4 = nn.GroupNorm(1, hidden_dim)
        else:
            self.norm1 = nn.BatchNorm2d(hidden_dim)
            self.norm2 = nn.BatchNorm2d(hidden_dim)

        self.lka = LargeKernelAttention(hidden_dim)
        self.se = SEModule(hidden_dim)

        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.out_proj.weight.requires_grad = False

        self.lora_A = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        init.kaiming_uniform_(self.lora_A.weight)
        init.zeros_(self.lora_B.weight)

    def forward(self, x, x_f):
        residual = x
        x = self.norm1(self.conv1(x))
        x = self.lka(x) + self.se(x)
        x = self.conv2(x)
        x = self.norm2(self.conv3(x))
        x = self.lka(x) + self.se(x + x_f)

        lora_feat = self.lora_B(self.lora_A(x_f)) * self.alpha

        x = self.conv4(x + lora_feat)
        return residual + 0.1 * x


class InvertedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)


class InBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            VMamba(dim),
            VMamba(dim),
            VMamba(dim),
        ])

    def forward(self, z1, z2):
        z2 = z2 + self.blocks[0](z1)
        log_scale = torch.tanh(self.blocks[1](z2))
        z1 = z1 * torch.exp(log_scale) + self.blocks[2](z2)
        return z1, z2


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, HL, LH, HH


def iwt_init(x):
    r = 2
    B, C4, H, W = x.size()
    C = C4 // 4

    x1 = x[:, :C, :, :] / 2
    x2 = x[:, C:2*C, :, :] / 2
    x3 = x[:, 2*C:3*C, :, :] / 2
    x4 = x[:, 3*C:4*C, :, :] / 2

    out = torch.zeros([B, C, H * r, W * r], dtype=x.dtype, device=x.device)

    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return out


class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class PIFNet(nn.Module):
    def __init__(self, hsi_bands=31, msi_bands=3, dim=64):
        super().__init__()
        self.hsi_bands = hsi_bands
        self.msi_bands = msi_bands
        self.dim = dim

        self.input_conv = nn.Conv2d(
            self.msi_bands, self.hsi_bands, kernel_size=3, padding=1
        )

        self.hsi_head = Head(self.hsi_bands, dim)
        self.msi_head = Head(self.msi_bands, dim)
        self.prior_conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.tail = Head(dim * 2, self.hsi_bands)

        self.conv_proj = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.conv_proj_trans = nn.Conv2d(dim, dim * 3, kernel_size=1)

        self.inblocks = nn.ModuleList([InBlock(dim) for _ in range(4)])

        self.spatial_blocks = nn.ModuleList([FCNormAttnNorm(dim, fc_ratio=1.0, norm_type='ln') for _ in range(4)])

        self.sem_blocks = nn.ModuleList([SemanticAlignBlock(dim, dim, kernel_size=3, upscale_factor=2) for _ in range(4)])

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError(
                f"PIFNet expects a BCHW tensor, got shape {tuple(image.shape)}"
            )
        if image.shape[1] != self.msi_bands:
            raise ValueError(
                f"PIFNet expects {self.msi_bands} input channels, "
                f"got {image.shape[1]}"
            )
        if image.shape[-2] % 2 != 0 or image.shape[-1] % 2 != 0:
            raise ValueError("Input height and width must be even for the Haar DWT")

        hrmsi = image
        hrhsi = self.input_conv(image)

        hsi_feat = self.hsi_head(hrhsi)
        msi_feat = self.msi_head(hrmsi)
        res_prior = F.relu(self.prior_conv(hsi_feat - msi_feat))

        x_LL, x_HL, x_LH, x_HH = dwt_init(hsi_feat)
        z1 = x_LL
        z2 = self.conv_proj(torch.cat([x_HL, x_LH, x_HH], dim=1))

        for i in range(4):
            z1, z2 = self.inblocks[i](z1, z2)

        z2 = self.conv_proj_trans(z2)
        feat = iwt_init(torch.cat([z1, z2], dim=1))

        for i in range(4):
            feat = self.spatial_blocks[i](feat, res_prior)
            feat = self.sem_blocks[i](feat)

        out = self.tail(torch.cat([feat, msi_feat], dim=1))

        return out


class SemanticAlignBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor):
        super().__init__()
        self.deform_conv1 = DeformConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.deform_conv2 = DeformConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv_offset1 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.conv_offset2 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        for conv in (self.conv_offset1, self.conv_offset2):
            init.zeros_(conv.weight)
            init.zeros_(conv.bias)
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        max_offset = 4.0

        offset1 = self.conv_offset1(x)
        if not torch.isfinite(offset1).all():
            raise FloatingPointError("SemanticAlignBlock.offset1 contains NaN/Inf")
        offset1 = max_offset * torch.tanh(offset1 / max_offset)

        x = self.deform_conv1(x, offset1)
        x = self.relu(x)

        offset2 = self.conv_offset2(x)
        if not torch.isfinite(offset2).all():
            raise FloatingPointError("SemanticAlignBlock.offset2 contains NaN/Inf")
        offset2 = max_offset * torch.tanh(offset2 / max_offset)

        x = self.deform_conv2(x, offset2)
        return x


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PIF-Net requires CUDA because mamba_ssm selective_scan is CUDA-only."
        )

    device = torch.device("cuda")
    model = PIFNet().to(device).eval()

    image = torch.rand(1, 3, 64, 64, device=device)

    with torch.no_grad():
        output_hrhsi = model(image)

    print(f"Image input  : {tuple(image.shape)}")
    print(f"HR-HSI output: {tuple(output_hrhsi.shape)}")
    print(f"Output type  : {type(output_hrhsi).__name__}")


if __name__ == "__main__":
    main()
