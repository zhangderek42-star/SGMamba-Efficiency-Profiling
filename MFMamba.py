import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import os
import warnings
warnings.filterwarnings('ignore')

import mfmamba_net_common as common
from mfmamba_mambair import MambaIR as MambaNet


class MDCB(nn.Module):
    def __init__(self, ch_in, ch_out, bias=True, activation=nn.ReLU(inplace=True)):
        super(MDCB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = common.default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_1,  bias=bias)
        self.conv_3_2 = common.default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_1,  bias=bias)
        self.conv_5_1 = common.default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_2,  bias=bias)
        self.conv_5_2 = common.default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_2,  bias=bias)
        self.confusion_3 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_5 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_bottle = nn.Conv2d(ch_in * 3 + ch_out * 2, ch_out, 1, padding=0, bias=True)
        self.activation = activation

    def forward(self, x):
        input_1 = x
        output_3_1 = self.activation(self.conv_3_1(input_1))
        output_3_1 += x
        output_5_1 = self.activation(self.conv_5_1(input_1))
        output_5_1 += x
        input_2 = torch.cat([input_1, output_3_1, output_5_1], 1)

        input_2_3 = self.confusion_3(input_2)
        input_2_5 = self.confusion_5(input_2)

        output_3_2 = self.activation(self.conv_3_2(input_2_3))
        output_5_2 = self.activation(self.conv_5_2(input_2_5))
        input_3 = torch.cat([input_1, output_3_1, output_5_1, output_3_2, output_5_2], 1)
        output = self.confusion_bottle(input_3)
        return output
    

class IMUB_Head(nn.Module):
    def __init__(self, ch_in, ch_out, activation=nn.ReLU(), bias=False, down_times=2, fe_num=16):
        super(IMUB_Head, self).__init__()

        self.down_times = down_times

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=fe_num, kernel_size=1, stride=1,padding=0, bias=bias),
            activation,
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=fe_num, out_channels=fe_num, kernel_size=3, stride=1,padding=1, bias=bias),
            activation,
        )

        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=fe_num * 3, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=bias),
            activation,
        )

        self.conv_sum = nn.Sequential(
            nn.Conv2d(in_channels=ch_out * (down_times + 1), out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=bias),
            activation,
        )

        self.activation = activation

    def forward(self, x):
        b, _, h, w = x.size()
        input, features, result = [], [], []
        input.append(x)

        for i in range(self.down_times):
            input.append(F.upsample(x, size=(x.shape[2] // (2*i + 2), x.shape[3] // (2*i + 2)), mode='bilinear'))

        for item in input:

            x1 = self.conv1_1(item)
            x1 = self.conv1_2(x1) + x1

            x2 = self.conv2_1(item)
            x2 = self.conv2_2(x2) + x2
            x2 = self.conv2_3(x2) + x2

            x3 = self.conv3_1(item)
            x3 = self.conv3_2(x3) + x3
            x3 = self.conv3_3(x3) + x3
            x3 = self.conv3_4(x3) + x3

            out = torch.cat([x1, x2,x3], dim=1)
            out = self.conv_end(out)
            features.append(out)

        for item in features:
            if item.shape[2] != h or item.shape[3] != w:
                result.append(F.upsample(item, size=(h, w), mode='bilinear'))
            else:
                result.append(item)

        out = result[0]
        for i in range(1, len(result)):
            out = torch.cat([out, result[i]], dim=1)

        del input, features, result
        out = self.conv_sum(out)

        return out

class IDB(nn.Module):

    def __init__(self, in_planes, planes, bias, activation):
        super(IDB, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1, stride=1, bias=bias),
            activation,
            common.involution(channels=planes, kernel_size=3, stride=2),
            activation,
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, padding=0, stride=1, bias=bias),
            activation,
        )

    def forward(self, x):
        out = self.cnn(x)
        return out


class IMUB(nn.Module):

    def __init__(self, in_planes, planes, bias, activation):
        super(IMUB, self).__init__()

        self.cnn = nn.Sequential(
            IMUB_Head(ch_in=in_planes, ch_out=planes, bias=bias, activation=activation),
            common.ConvUpsampler(ch_in=planes, ch_out=planes, activation=activation, bias=bias),
            activation,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x

class SkipProjection(nn.Module):

    def __init__(self, in_planes):
        super(SkipProjection, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

class MFmamba(nn.Module):

    upscale = 1
    in_channels = 3
    out_channels = 31

    def __init__(self):
        super(MFmamba, self).__init__()

        self.depth = 4
        self.fe_num = 128
        self.grow_rate = 32
        self.bias = True
        self.activation = common.Mish()

        self.root = self.make_head()

        self.layers = self._make_layers(depth=self.depth, in_planes=self.fe_num)

        self.end_convs = nn.Sequential(
            common.Default_Conv(ch_in=self.fe_num, ch_out=self.out_channels)
        )

    def make_head(self):
        return nn.Sequential(
            MDCB(self.in_channels, 32, bias=self.bias, activation=self.activation),
            MDCB(32, self.fe_num, bias=self.bias, activation=self.activation),
            MambaNet(
                img_size=64,
                patch_size=1,
                in_chans=self.fe_num,
                embed_dim=48,
                depths=(5, 5, 5, 5),
                mlp_ratio=1.,
                drop_rate=0.,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                upscale=self.upscale,
                img_range=1.,
                upsampler='same_resolution',
                resi_connection='1conv',
            ),
        )

    def _make_layers(self, depth, in_planes):

        layers = []
        in_planes = in_planes
        planes = in_planes + self.grow_rate
        for i in range(depth):
            layer_list = []
            if i == 0:
                layers_planes = planes
                layer_list.append(IDB(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                layer_list.append(IMUB(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                layer_list.append(SkipProjection(layers_planes - self.grow_rate))
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
            else:
                layers_planes = planes
                for j in range(i + 1):
                    if j == 0:
                        layer_list.append(IDB(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                    layer_list.append(IMUB(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                    layer_list.append(SkipProjection(layers_planes - self.grow_rate))
                    layers_planes -= self.grow_rate
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        if input_tensor.ndim != 4:
            raise ValueError(f'MFmamba input must have shape [B, 3, H, W], got {tuple(input_tensor.shape)}')
        if input_tensor.shape[1] != self.in_channels:
            raise ValueError(f'MFmamba expects 3 input channels, got {input_tensor.shape[1]}')
        if not torch.is_floating_point(input_tensor):
            raise TypeError(f'MFmamba expects a floating-point input tensor, got {input_tensor.dtype}')

        input_height, input_width = input_tensor.shape[-2:]
        if input_height < 64 or input_width < 64:
            raise ValueError(
                'MFmamba input height and width must be at least 64, '
                f'got {(input_height, input_width)}'
            )
        if input_height % 16 != 0 or input_width % 16 != 0:
            raise ValueError(
                'MFmamba input height and width must be divisible by 16, '
                f'got {(input_height, input_width)}'
            )

        features = []
        x = self.root(input_tensor)
        features.append([x])

        for i in range(self.depth):
            if i == 0:
                y_11 = self.layers[0][0](x)
                y_12 = self.layers[0][1](y_11) + self.layers[0][2](x) + x
                features.append([y_11, y_12])
            else:
                features_layer = []
                layer_step = 0
                feature_step = 0
                for j in range(i + 1):
                    if j == 0:
                        y_n1 = self.layers[i][0](features[i][0])
                        y_n2 = self.layers[i][1](y_n1) + self.layers[i][2](features[i][0])+features[i][0]
                        features_layer.append(y_n1)
                        features_layer.append(y_n2)
                        layer_step = 3
                        feature_step = 1
                    else:
                        y_nm = self.layers[i][layer_step](features_layer[-1]) + self.layers[i][layer_step + 1](features[i][feature_step])+features[i][feature_step]
                        features_layer.append(y_nm)
                        layer_step += 2
                        feature_step += 1
                features.append(features_layer)

        output = self.end_convs[0](features[-1][-1])
        expected_size = (input_height, input_width)
        if output.shape[1] != self.out_channels or output.shape[-2:] != expected_size:
            raise RuntimeError(
                'MFmamba produced an invalid output shape: expected '
                f'[B, {self.out_channels}, {expected_size[0]}, {expected_size[1]}], '
                f'got {tuple(output.shape)}'
            )
        return output

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        pass

    def load_compatible_state_dict(self, state_dict):
        current_state = self.state_dict()
        is_legacy_x2 = any(key.startswith('root.2.conv_before_upsample.') for key in state_dict)
        has_legacy_output_head = (
            'end_convs.0.conv.weight' in state_dict
            and state_dict['end_convs.0.conv.weight'].shape
            != current_state['end_convs.0.conv.weight'].shape
        )
        ignored_prefixes = (
            'end_convs.1.',
            'end_convs.2.',
            'end_convs.3.',
            'root.2.conv_before_upsample.',
            'root.2.upsample.',
        )

        filtered_state = {}
        unexpected = []
        mismatched = []
        for key, value in state_dict.items():
            if key not in current_state:
                if key.startswith(ignored_prefixes):
                    continue
                unexpected.append(key)
                continue
            if value.shape != current_state[key].shape:
                if is_legacy_x2 and key == 'root.2.conv_last.weight':
                    continue
                if has_legacy_output_head and key == 'end_convs.0.conv.weight':
                    continue
                mismatched.append(
                    f'{key}: checkpoint {tuple(value.shape)} != model {tuple(current_state[key].shape)}'
                )
                continue
            filtered_state[key] = value

        incompatible = self.load_state_dict(filtered_state, strict=False)
        allowed_missing = {'root.2.conv_last.weight'} if is_legacy_x2 else set()
        if has_legacy_output_head:
            allowed_missing.add('end_convs.0.conv.weight')
        missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                'Checkpoint does not match the same-resolution MFmamba model. '
                f'Missing keys: {missing}; unexpected keys: {unexpected}; '
                f'shape mismatches: {mismatched}'
            )
        return self

    def load_training_checkpoint(self, checkpoint_dir):
        self.train_epoch = 1
        if not os.path.isdir(checkpoint_dir):
            print(f'Checkpoint directory does not exist: {checkpoint_dir}')
            return self

        checkpoint_names = sorted(
            name for name in os.listdir(checkpoint_dir) if name.endswith('.pkl')
        )
        if not checkpoint_names:
            print(f'No training checkpoint found in: {checkpoint_dir}')
            return self

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[-1])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_compatible_state_dict(checkpoint['model_state_dict'])
        self.train_epoch = checkpoint['epoch']
        print(f'Training checkpoint loaded: {checkpoint_path} (epoch {self.train_epoch})')
        return self


def main(input_tensor, model=None):
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f'main expects a torch.Tensor, got {type(input_tensor).__name__}')

    if model is None:
        model = MFmamba().to(input_tensor.device)
        model.eval()
    return model(input_tensor)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFmamba().to(device).eval()
    input_tensor = torch.rand(1, 3, 64, 64, device=device)

    with torch.no_grad():
        output_tensor = main(input_tensor, model=model)

    expected_shape = (1, 31, 64, 64)
    if tuple(output_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"MFmamba self-test failed: expected {expected_shape}, "
            f"got {tuple(output_tensor.shape)}"
        )

    print(f"Device: {device}")
    print(f"Input shape:  {tuple(input_tensor.shape)}")
    print(f"Output shape: {tuple(output_tensor.shape)}")
