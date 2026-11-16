# SGMamba Efficiency Profiling

This repository contains the code for analyzing the computational efficiency of SGMamba.

## K-means indexing

To evaluate the cost of K-means indexing, run the following scripts:

- [kmeans_stages_flops.py](kmeans_stages_flops.py) measures the computation in GFLOPs.
- [kmeans_stages_time.py](kmeans_stages_time.py) measures the average running time in milliseconds.
- [kmeans_stages_memory.py](kmeans_stages_memory.py) measures the additional peak GPU memory in MiB.

Each script reports results separately for K-means, centroid distance sorting, index construction, reordering and inverse reordering.

## Model profiling

The profiling scripts support the following models: `HSCNN+`, `HSRnet`, `AWAN`, `MST++`, `HSACS`, `LTRN`, `MambaIRv2`, `M3SR`, `T_PMamba`, `MFMamba`, `PIFNet` and `SGMamba`. Choose one with `--model` and run the script for the metric you want to measure:

- [profile_flops_params.py](profile_flops_params.py) measures FLOPs in GFLOPs and the parameter count in millions.
- [profile_inference_time.py](profile_inference_time.py) measures the average inference time in milliseconds.
- [profile_memory.py](profile_memory.py) measures the peak GPU memory usage in MiB.

For example, to measure the FLOPs and parameter count of SGMamba on GPU 8:

```bash
python profile_flops_params.py --model SGMamba --device 8
```
