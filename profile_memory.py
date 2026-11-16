import argparse
import gc
import json
import os
from pathlib import Path

from profile_flops_params import MODEL_NAMES, SCRIPT_DIR, gpu_index


def measure_peak_memory(options):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.device)

    import torch
    from model_all import create_model

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Physical GPU {options.device} is unavailable. Check nvidia-smi."
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    registry_name = MODEL_NAMES[options.model]

    inputs = torch.load(options.input_file, map_location="cpu", weights_only=True)
    image_cpu = inputs["hrms"]
    if image_cpu.ndim != 4 or image_cpu.shape[1] != 3:
        raise ValueError(
            f"hrms must have shape (B, 3, H, W), got {tuple(image_cpu.shape)}."
        )

    model = create_model(registry_name).to(device).eval()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)

    image = image_cpu.to(device)
    if registry_name == "Mamba++_ca2":
        indices = {
            key: value.to(device) for key, value in inputs["hrms_index_dict"].items()
        }
        forward_args = [inputs["R"].to(device), image, indices]
    else:
        forward_args = [image]

    torch.cuda.synchronize(device)
    with torch.inference_mode():
        output = model(*forward_args)
    torch.cuda.synchronize(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    del output

    return {
        "model": options.model,
        "Memory_MiB": peak_mem / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure peak GPU memory for a single inference forward."
    )
    parser.add_argument(
        "--model", required=True, choices=MODEL_NAMES,
        help="Available models: " + ", ".join(MODEL_NAMES),
    )
    parser.add_argument(
        "--device", type=gpu_index, default=8,
        help="Physical GPU index, as shown by nvidia-smi (default: 8).",
    )
    parser.add_argument(
        "--input-file", type=Path, default=SCRIPT_DIR / "flops_params.pt",
        help="Saved inputs containing hrms and, for SGMamba, R/hrms_index_dict.",
    )
    options = parser.parse_args()
    result = measure_peak_memory(options)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
