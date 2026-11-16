import argparse
import json
import os
import statistics
from pathlib import Path

from profile_flops_params import MODEL_NAMES, SCRIPT_DIR, gpu_index


def benchmark_model(options):
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
    image = inputs["hrms"].to(device)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"hrms must have shape (B, 3, H, W), got {tuple(image.shape)}.")
    if registry_name == "Mamba++_ca2":
        indices = {
            key: value.to(device) for key, value in inputs["hrms_index_dict"].items()
        }
        forward_args = [inputs["R"].to(device), image, indices]
    else:
        forward_args = [image]

    model = create_model(registry_name).to(device).eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms = []

    with torch.no_grad():
        for _ in range(20):
            model(*forward_args)
        torch.cuda.synchronize(device)

        start.record()
        end.record()
        end.synchronize()

        for _ in range(100):
            start.record()
            output = model(*forward_args)
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
            del output

    time_ms = statistics.fmean(samples_ms)
    if registry_name == "Mamba++_ca2":
        time_ms += 5.5867

    return {
        "model": options.model,
        "Time_ms": time_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure model forward time using CUDA Events."
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
    result = benchmark_model(options)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
