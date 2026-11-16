from batch_kmeans_index import KMeansIndexer, SequenceIndexer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import torch
import torch.nn as nn



def create_F():
    F = np.array([
        [2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


class KMeansIndex(nn.Module):
    def __init__(self, n_clusters=5):
        super(KMeansIndex, self).__init__()
        self.n_clusters = n_clusters
        self.kmeans_indexer = KMeansIndexer(n_clusters=n_clusters)
        self.sequence_indexer = SequenceIndexer()
    

    def image2index(self, image):
        
        b, c, h, w = image.shape
        
        _, index_seq = self.kmeans_indexer.unfold(image, "distance")
        image_index = index_seq.reshape(b, h, w).contiguous()
        return image_index
    
    def index2index(self, image_index, resolution=[64,32,16]):
        result_index = []

        for res in resolution:
            b, h, w = image_index.shape
            l = h*w
            image_flatten = image_index.reshape(b, l).contiguous()
            index_seq_flatten = self.sequence_indexer.transform_rescale(
                image_flatten,
                input_resolution=h,
                output_resolution=res
            )
            index_seq = index_seq_flatten.reshape(b, res, res).contiguous()
            result_index.append(index_seq)

        return result_index
    
    def slice_index(self, image_index, start_x, start_y, size):
        b, h, w = image_index.shape
        l = h*w
        image_flatten = image_index.reshape(b, l).contiguous()
        index_seq_flatten = self.sequence_indexer.transform_slice(
            image_flatten,
            x_start=start_x,
            y_start=start_y,
            x_end=start_x+size,
            y_end=start_y+size,
            h=h,
            w=w
        )
        index_seq = index_seq_flatten.reshape(b, size, size).contiguous()
        return index_seq
    
    def unfold(self, image_tensor, image_index):
        b, c, h, w = image_tensor.shape
        b1, h1, w1 = image_index.shape
        assert b==b1 and h==h1 and w==w1, "Image tensor and image index must have the same batch size and spatial dimensions"
        image_seq = image_tensor.reshape(b, c, h, w).contiguous()
        image_index_seq = image_index.reshape(b, h*w).contiguous()
        unfolded_image_seq, _ = self.kmeans_indexer.unfold_custom_index(image_seq, image_index_seq)
        return unfolded_image_seq
    
    def fold(self, scanned_image_seq, image_index):
        b, c, l = scanned_image_seq.shape
        b1, h1, w1 = image_index.shape
        assert b==b1 and l==h1*w1, "Scanned image sequence and image index must have the same batch size and sequence length"
        image_index_seq = image_index.reshape(b, l).contiguous()
        folded_image_seq = self.kmeans_indexer.fold(scanned_image_seq, image_index_seq)
        folded_image_tensor = folded_image_seq.reshape(b, c, h1, w1).contiguous()
        return folded_image_tensor
        
