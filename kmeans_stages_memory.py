from kmeans_index_class import KMeansIndex, create_F
from ca_transform3 import ca_transform
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import os
import gc


class StageMemory:
    def __init__(self, name, stage_memory):
        self.name=name
        self.stage_memory=stage_memory

    def __enter__(self):
        if self.stage_memory is not None:
            torch.cuda.synchronize()
            self.baseline=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stage_memory is not None:
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()
            value=self.stage_memory[self.name]
            value['peak']=max(value['peak'], peak)
            value['extra']=max(value['extra'], peak-self.baseline)


class IndexReorderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kmeans_index_model=KMeansIndex(n_clusters=5)
        self.ca_transform=ca_transform(seq_length=32)

    def forward(self, x, model_inputs, stage_memory=None, save_outputs=False):
        with StageMemory('index_build', stage_memory):
            image_tensor=torch.as_tensor(x, dtype=torch.float32)
            b,c,h,w=image_tensor.shape
            input_pixels=image_tensor.view(b,c,-1).permute(0,2,1).contiguous()

        with StageMemory('kmeans', stage_memory):
            kmeans_result=self.kmeans_index_model.kmeans_indexer.model(input_pixels)

        with StageMemory('centroid_distance_sort', stage_memory):
            index_inner_center=torch.argsort(kmeans_result.inertia[1], dim=1)
            labels_inner_center=torch.gather(kmeans_result.labels, 1, index_inner_center)

        with StageMemory('index_build', stage_memory):
            index_outer_center=torch.argsort(labels_inner_center, dim=1, stable=True)
            index_gather=torch.gather(index_inner_center, 1, index_outer_center)
            result_pixels=torch.gather(
                input_pixels, 1, index_gather.unsqueeze(-1).expand(-1,-1,c))
            image_seq=result_pixels.permute(0,2,1).contiguous()
            image_index=index_gather.reshape(b,h,w).contiguous()
            result_index=self.kmeans_index_model.index2index(
                image_index, resolution=[64,32,16])
            index_dict={key: result_index[key].long().cuda() for key in range(3)}

        del input_pixels, kmeans_result, index_inner_center, labels_inner_center
        del index_outer_center, index_gather, result_pixels, image_seq, image_index, result_index

        reordered_outputs=[]
        restored_outputs=[]
        for key, inputs in zip((0,1,2,1,0), model_inputs):
            image_index=index_dict[key]
            b,h,w=image_index.shape

            with StageMemory('reorder', stage_memory):
                x_unfold=self.kmeans_index_model.unfold(inputs['mamba_feature'], image_index)
                xs=torch.stack([x_unfold, x_unfold.flip(dims=[-1])], dim=1)

            with StageMemory('inverse_reorder', stage_memory):
                y1=self.kmeans_index_model.fold(inputs['mamba_scan'][:,0], image_index)
                y2=self.kmeans_index_model.fold(
                    inputs['mamba_scan'][:,1].flip(dims=[-1]), image_index)

            with StageMemory('index_build', stage_memory):
                natural_index=torch.arange(h*w, device=image_index.device).view(1,h,w)
                natural_index=natural_index.expand(b,-1,-1).contiguous()

            with StageMemory('reorder', stage_memory):
                x_unfold2=self.kmeans_index_model.unfold(inputs['mamba_feature'], natural_index)
                xs2=torch.stack([x_unfold2, x_unfold2.flip(dims=[-1])], dim=1)

            with StageMemory('inverse_reorder', stage_memory):
                y3=self.kmeans_index_model.fold(inputs['natural_scan'][:,0], natural_index)
                y4=self.kmeans_index_model.fold(
                    inputs['natural_scan'][:,1].flip(dims=[-1]), natural_index)

            with StageMemory('reorder', stage_memory):
                ca_input=self.ca_transform.fold(inputs['ca_feature'], image_index=image_index)

            with StageMemory('inverse_reorder', stage_memory):
                ca_output=self.ca_transform.unfold(
                    inputs['ca_scan'], original_B=b, original_H=h,
                    original_W=w, image_index=image_index)

            if save_outputs:
                reordered_outputs.append((xs, xs2, ca_input))
                restored_outputs.append((y1, y2, y3, y4, ca_output))
            del x_unfold, x_unfold2, xs, xs2, y1, y2, y3, y4
            del ca_input, ca_output, natural_index

        return x, index_dict, reordered_outputs, restored_outputs


seed=1039
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device=3
torch.cuda.set_device(device)
path=os.path.dirname(os.path.abspath(__file__)) + '/'
R=create_F()
img=loadmat(path + 'clay_ms.mat')['orig'][:256,:256,:]
HRHSI=np.transpose(img, (2,0,1))
MSI=np.tensordot(R, HRHSI, axes=([1],[0]))
hrms256=torch.Tensor(MSI).unsqueeze(0).cuda()
hrms=F.interpolate(hrms256.unsqueeze(1), size=(3,64,64),
                   mode='trilinear', align_corners=False).squeeze(1)
del hrms256
assert tuple(hrms.shape)==(1,3,64,64)
R2=torch.Tensor(R).cuda()

model_inputs=[]
for key in (0,1,2,1,0):
    resolution=(64,32,16)[key]
    channels=(31,62,124)[key]
    mamba_channels=int(1.2 * channels)
    num_pixels=resolution * resolution
    model_inputs.append({
        'mamba_feature': torch.randn(1,mamba_channels,resolution,resolution).cuda(),
        'mamba_scan': torch.randn(1,2,mamba_channels,num_pixels).cuda(),
        'natural_scan': torch.randn(1,2,mamba_channels,num_pixels).cuda(),
        'ca_feature': torch.randn(1,resolution,resolution,channels).cuda(),
        'ca_scan': torch.randn(num_pixels//32,channels,32).cuda(),
    })

stage_names={
    'kmeans': 'K-means',
    'centroid_distance_sort': 'Centroid distance sorting',
    'index_build': 'Index construction',
    'reorder': 'Reordering',
    'inverse_reorder': 'Inverse reordering',
}
stage_memory={name: {'peak': 0, 'extra': 0} for name in stage_names}
cnn=IndexReorderModel().cuda().eval()

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
baseline_mem=torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    outputs=cnn(hrms, model_inputs)
torch.cuda.synchronize()
peak_mem=torch.cuda.max_memory_allocated()
extra_peak_mem=peak_mem-baseline_mem
del outputs

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
with torch.inference_mode():
    a2,index_dict,reordered_outputs,restored_outputs=cnn(hrms, model_inputs, stage_memory)
torch.cuda.synchronize()

mib=1024**2
for name, label in stage_names.items():
    value=stage_memory[name]
    print('{}: {:.4f} MiB'.format(label, value['extra']/mib))

del model_inputs, a2, index_dict, reordered_outputs, restored_outputs, R2
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    kmeans_output=cnn.kmeans_index_model.kmeans_indexer.model(
        hrms.flatten(2).permute(0,2,1).contiguous())
torch.cuda.synchronize()
kmeans_peak_mem=torch.cuda.max_memory_allocated()
del kmeans_output

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    image_index=cnn.kmeans_index_model.image2index(hrms)
    result_index=cnn.kmeans_index_model.index2index(image_index, resolution=[64,32,16])
torch.cuda.synchronize()
index_peak_mem=torch.cuda.max_memory_allocated()

"""
K-means: 1.0503 MiB
Centroid distance sorting: 0.0781 MiB
Index construction: 0.3906 MiB
Reordering: 2.3125 MiB
Inverse reordering: 2.3438 MiB
"""
