"""
# PyTorch version of HSRnet — Spectral Super-Resolution (RGB → HSI)
# Modified from: Hyperspectral Image Super-resolution via Deep Spatio-spectral Attention Convolutional Neural Networks
# Original author: Jin-Fan Hu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py
import scipy.io as sio

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class HSRnet(nn.Module):
    def __init__(self, in_channels=3, num_spectral=31, num_res=10, num_fm=64):
        super().__init__()

        self.sa_conv = nn.Conv2d(in_channels, 1, 6, 1, padding=0)

        self.entry_conv = nn.Conv2d(in_channels, num_fm, 3, 1, 1)

        self.res_convs = nn.ModuleList()
        for _ in range(num_res):
            self.res_convs.append(nn.ModuleList([
                nn.Conv2d(num_fm, num_fm, 3, 1, 1),
                nn.Conv2d(num_fm, num_fm, 3, 1, 1),
            ]))

        self.final_conv = nn.Conv2d(num_fm, num_spectral, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb):
        sa = torch.sigmoid(self.sa_conv(F.pad(rgb, [2, 3, 2, 3])))

        rs = F.leaky_relu(self.entry_conv(rgb), negative_slope=0.2)

        for conv1, conv2 in self.res_convs:
            rs1 = F.leaky_relu(conv1(rs), negative_slope=0.2)
            rs1 = conv2(rs1)
            rs = rs + rs1

        rs = sa * rs
        rs = self.final_conv(rs)

        return rs


def train():
    train_batch_size = 32
    test_batch_size = 32
    image_size = 64
    bands = 31
    iterations = 150001
    train_data_name = 'train4(20-11)(pRGB).mat'
    test_data_name = 'validation4(20-11)(pRGB).mat'
    restore = False
    method = 'Adam'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HSRnet().to(device)

    if method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.1)

    if restore:
        print('Loading Model...')
        ckpt = torch.load(os.path.join(model_directory, 'model-latest.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_data = h5py.File(train_data_name, 'r')
    test_data = h5py.File(test_data_name, 'r')

    gt1 = np.array(train_data['gt'][...], dtype=np.float32) / (2 ** 16 - 1)
    rgb1 = np.array(train_data['rgb1'][...], dtype=np.float32) / (2 ** 8 - 1)
    N = gt1.shape[0]

    gt1 = gt1.transpose(0, 3, 1, 2)
    rgb1 = rgb1.transpose(0, 3, 1, 2)

    gt2 = np.array(test_data['gt'][...], dtype=np.float32) / (2 ** 16 - 1)
    rgb2 = np.array(test_data['rgb1'][...], dtype=np.float32) / (2 ** 8 - 1)
    N2 = gt2.shape[0]

    gt2 = gt2.transpose(0, 3, 1, 2)
    rgb2 = rgb2.transpose(0, 3, 1, 2)

    mse_train = []
    mse_valid = []

    model.train()
    for i in range(iterations):
        batch_index = np.random.randint(0, N, size=train_batch_size)

        train_gt = torch.from_numpy(gt1[batch_index]).to(device)
        rgb_batch = torch.from_numpy(rgb1[batch_index]).to(device)

        optimizer.zero_grad()
        output = model(rgb_batch)
        loss = torch.mean(torch.abs(output - train_gt))
        loss.backward()

        if method != 'Adam':
            lr = optimizer.param_groups[0]['lr']
            clip_value = 0.1 / lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        if method != 'Adam':
            scheduler.step()

        if i % 1000 == 0:
            mse_train.append(loss.item())
            print("Iter: " + str(i) + " loss: " + str(loss.item()))

        if i % 10000 == 0 and i != 0:
            os.makedirs(model_directory, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': i,
            }, os.path.join(model_directory, 'model-' + str(i) + '.pth'))
            print("Save Model")

        if i % 2000 == 0 and i != 0:
            model.eval()
            with torch.no_grad():
                batch_index2 = np.random.randint(0, N2, size=test_batch_size)
                test_gt_batch = torch.from_numpy(gt2[batch_index2]).to(device)
                test_rgb_batch = torch.from_numpy(rgb2[batch_index2]).to(device)

                test_output = model(test_rgb_batch)
                test_loss = torch.mean(torch.abs(test_output - test_gt_batch))
                mse_valid.append(test_loss.item())
                print("Iter: " + str(i) + " Valid loss: " + str(test_loss.item()))
            model.train()


def test():
    test_data_file = 'test_cave_demo.mat'
    N = 1
    sz = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HSRnet().to(device)

    ckpt = torch.load(os.path.join(model_directory, 'model-150000.pth'), map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    OUT = np.zeros((N, sz, sz, 31))

    data = sio.loadmat(test_data_file)
    Inrgb = data['rgb1'][...]

    with torch.no_grad():
        for i in range(N):
            rgb = Inrgb[i, :, :, :].astype(np.float32) / (2 ** 8 - 1)
            rgb = rgb[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

            rgb_t = torch.from_numpy(rgb).to(device)

            output = model(rgb_t)
            output = torch.clamp(output, 0, 1)

            OUT[i] = output.cpu().numpy().transpose(0, 2, 3, 1)[0]
            print('testing image' + str(i + 1) + ' is done!')

    sio.savemat('output-HSRnet-cave.mat', {'output': OUT})


if __name__ == '__main__':
    model_directory = 'models(cave)'
    test()
