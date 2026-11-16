from kmeans_index_class import KMeansIndex, create_F
from ca_transform3 import ca_transform
from scipy.io import loadmat
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import os
import math


class IndexReorderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kmeans_index_model=KMeansIndex(n_clusters=5)
        self.ca_transform=ca_transform(seq_length=32)

    def forward(self, x, model_inputs):
        with record_function('index_build'):
            image_tensor=torch.as_tensor(x, dtype=torch.float32)
            b,c,h,w=image_tensor.shape
            input_pixels=image_tensor.view(b,c,-1).permute(0,2,1).contiguous()

        with record_function('kmeans'):
            kmeans_result=self.kmeans_index_model.kmeans_indexer.model(input_pixels)

        with record_function('centroid_distance_sort'):
            index_inner_center=torch.argsort(kmeans_result.inertia[1], dim=1)
            labels_inner_center=torch.gather(kmeans_result.labels, 1, index_inner_center)

        with record_function('index_build'):
            index_outer_center=torch.argsort(labels_inner_center, dim=1, stable=True)
            index_gather=torch.gather(index_inner_center, 1, index_outer_center)
            result_pixels=torch.gather(
                input_pixels, 1, index_gather.unsqueeze(-1).expand(-1,-1,c))
            image_seq=result_pixels.permute(0,2,1).contiguous()
            image_index=index_gather.reshape(b,h,w).contiguous()
            result_index=self.kmeans_index_model.index2index(
                image_index, resolution=[64,32,16])
            index_dict={key: result_index[key].long().cuda() for key in range(3)}

        reordered_outputs=[]
        restored_outputs=[]
        for key, inputs in zip((0,1,2,1,0), model_inputs):
            image_index=index_dict[key]
            b,h,w=image_index.shape

            with record_function('reorder'):
                x_unfold=self.kmeans_index_model.unfold(inputs['mamba_feature'], image_index)
                xs=torch.stack([x_unfold, x_unfold.flip(dims=[-1])], dim=1)

            with record_function('inverse_reorder'):
                y1=self.kmeans_index_model.fold(inputs['mamba_scan'][:,0], image_index)
                y2=self.kmeans_index_model.fold(
                    inputs['mamba_scan'][:,1].flip(dims=[-1]), image_index)

            with record_function('index_build'):
                natural_index=torch.arange(h*w, device=image_index.device).view(1,h,w)
                natural_index=natural_index.expand(b,-1,-1).contiguous()

            with record_function('reorder'):
                x_unfold2=self.kmeans_index_model.unfold(inputs['mamba_feature'], natural_index)
                xs2=torch.stack([x_unfold2, x_unfold2.flip(dims=[-1])], dim=1)

            with record_function('inverse_reorder'):
                y3=self.kmeans_index_model.fold(inputs['natural_scan'][:,0], natural_index)
                y4=self.kmeans_index_model.fold(
                    inputs['natural_scan'][:,1].flip(dims=[-1]), natural_index)

            with record_function('reorder'):
                ca_input=self.ca_transform.fold(inputs['ca_feature'], image_index=image_index)

            with record_function('inverse_reorder'):
                ca_output=self.ca_transform.unfold(
                    inputs['ca_scan'], original_B=b, original_H=h,
                    original_W=w, image_index=image_index)

            reordered_outputs.append((xs, xs2, ca_input))
            restored_outputs.append((y1, y2, y3, y4, ca_output))

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

cnn=IndexReorderModel().cuda().eval()
with torch.no_grad(), torch.jit.optimized_execution(False):
    cnn(hrms, model_inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_flops=True) as prof:
        a2,index_dict,reordered_outputs,restored_outputs=cnn(hrms, model_inputs)
        torch.cuda.synchronize()

stage_names={
    'kmeans': 'K-means',
    'centroid_distance_sort': 'Centroid distance sorting',
    'index_build': 'Index construction',
    'reorder': 'Reordering',
    'inverse_reorder': 'Inverse reordering',
}
stage_flops={name: {'profiler': 0, 'extra': 0} for name in stage_names}
stage_ops={name: {'sort': 0, 'gather': 0, 'bmm': 0} for name in stage_names}

for item in prof.events():
    parent=item.cpu_parent
    while parent is not None and parent.name not in stage_names:
        parent=parent.cpu_parent
    if parent is None:
        continue
    stage=parent.name
    stage_flops[stage]['profiler']+=item.flops or 0
    if item.name in ('aten::sort', 'aten::gather', 'aten::bmm'):
        stage_ops[stage][item.name.split('::')[1]]+=1

profiler_flops=sum(value['profiler'] for value in stage_flops.values())
total_flops=profiler_flops

for name, label in stage_names.items():
    value=stage_flops[name]
    print('{}: {:.8f} GFLOPs'.format(label, value['profiler']/1e9))

"""
K-means: 0.00092160 GFLOPs
Centroid distance sorting: 0.00000000 GFLOPs
Index construction: 0.00000000 GFLOPs
Reordering: 0.00000000 GFLOPs
Inverse reordering: 0.00000000 GFLOPs
"""
