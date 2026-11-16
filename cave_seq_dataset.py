import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional as F
from torch import nn
import torch
import os
import sys
import torch.utils.data as data


from kmeans_index_class import KMeansIndex


import h5py
from tqdm import tqdm, trange

seed=42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



class CAVE_seq_dataset(data.Dataset):
    def __init__(self, path, R, training_size, stride, num, type='sim'):
        super(CAVE_seq_dataset, self).__init__()

        imglist = os.listdir(path)

        self.length_num=0
        self.kmeans_index_model = KMeansIndex(n_clusters=5)
        self.kmeans_index_model.cuda()

        for i in range(num):
            img = loadmat(path + imglist[i])
            img1 = img["orig"]
            for j in range(0, img1.shape[0] - training_size + 1, stride):
                for k in range(0, img1.shape[1] - training_size + 1, stride):
                    self.length_num+=1
        self.train_hrhs_all = torch.zeros((self.length_num, 31, training_size, training_size))
        self.train_hrms_all = torch.zeros((self.length_num, 3, training_size, training_size))
        self.train_labels_all_r1 = torch.zeros((self.length_num, 1, training_size, training_size), dtype=torch.long)
        resolution2 = int(training_size // 2)
        assert resolution2 * 2 == training_size, "training_size must be even"
        self.train_labels_all_r2 = torch.zeros((self.length_num, 1, resolution2, resolution2), dtype=torch.long)
        resolution3 = int(resolution2 // 2)
        assert resolution3 * 2 == resolution2, "training_size must be divisible by 4"
        self.train_labels_all_r3 = torch.zeros((self.length_num, 1, resolution3, resolution3), dtype=torch.long)

        self.resolution_list = [training_size, resolution2, resolution3]

        

        index = 0
        if type == 'sim':
            for i in trange(num, desc="Loading training images"):
                img = loadmat(path + imglist[i])
                img1 = img["orig"]
                HRHSI = np.transpose(img1, (2, 0, 1))
                MSI = np.tensordot(R, HRHSI, axes=([1], [0]))

                batched_MSI = torch.as_tensor(MSI).unsqueeze(0).cuda()
                image_index = self.kmeans_index_model.image2index(batched_MSI)

                for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                    for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                        temp_hrhs = HRHSI[:, j : j + training_size, k : k + training_size]
                        temp_hrms = MSI[:, j : j + training_size, k : k + training_size]

                        temp_image_index_r1 = self.kmeans_index_model.slice_index(image_index, j, k, self.resolution_list[0])
                        temp_image_index_r2 = self.kmeans_index_model.slice_index(image_index, j, k, self.resolution_list[0])
                        temp_image_index_r3 = self.kmeans_index_model.slice_index(image_index, j, k, self.resolution_list[0])

                        temp_result_index_r1 = self.kmeans_index_model.index2index(temp_image_index_r1, resolution=[self.resolution_list[0]])[0]
                        temp_result_index_r2 = self.kmeans_index_model.index2index(temp_image_index_r2, resolution=[self.resolution_list[1]])[0]
                        temp_result_index_r3 = self.kmeans_index_model.index2index(temp_image_index_r3, resolution=[self.resolution_list[2]])[0]


                        self.train_hrhs_all[index,:,:,:]=torch.Tensor(temp_hrhs)
                        self.train_hrms_all[index,:,:,:]=torch.Tensor(temp_hrms)

                        self.train_labels_all_r1[index,0,:,:]=temp_result_index_r1[0,:,:].cpu()
                        self.train_labels_all_r2[index,0,:,:]=temp_result_index_r2[0,:,:].cpu()
                        self.train_labels_all_r3[index,0,:,:]=temp_result_index_r3[0,:,:].cpu()

                        index += 1
        elif type == 'real':
            for i in trange(num, desc="Loading training images"):
                img = loadmat(path + imglist[i])
                img1 = img["orig"]
                HRHSI = np.transpose(img1, (2, 0, 1))
                bmp_path = path.rstrip('/').rsplit('/', 1)[0] + '/bmp/' + imglist[i].replace('_ms.mat', '_RGB.bmp')
                bmp = plt.imread(bmp_path)
                bmp = bmp.astype(np.float32) / 255.0

                MSI = bmp.transpose(2, 0, 1)

                for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                    for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                        temp_hrhs = HRHSI[:, j : j + training_size, k : k + training_size]
                        temp_hrms = MSI[:, j : j + training_size, k : k + training_size]
                        self.train_hrhs_all[index,:,:,:]=torch.Tensor(temp_hrhs)
                        self.train_hrms_all[index,:,:,:]=torch.Tensor(temp_hrms)
                        index += 1
        else:
            raise ValueError("Unknown type: {}".format(type))


    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]


        train_labels_r1 = self.train_labels_all_r1[index, :, :, :]
        train_labels_r2 = self.train_labels_all_r2[index, :, :, :]
        train_labels_r3 = self.train_labels_all_r3[index, :, :, :]

        index_dict = {}
        index_dict[0] = train_labels_r1[0]
        index_dict[1] = train_labels_r2[0]
        index_dict[2] = train_labels_r3[0]

        return train_hrhs, train_hrms, index_dict


    def __len__(self):
        return self.train_hrhs_all.shape[0]




def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F


