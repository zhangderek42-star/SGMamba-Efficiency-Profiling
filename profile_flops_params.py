import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAMES = {
    "HSCNN+": "HSCNN",
    "HSRnet": "HSRnet",
    "AWAN": "AWAN",
    "MST++": "MST++",
    "HSACS": "HSACS",
    "LTRN": "LTRN",
    "MambaIRv2": "Mambair2",
    "M3SR": "M3SR",
    "T_PMamba": "T_PMamba",
    "MFMamba": "MFmamba",
    "PIFNet": "PIFNet",
    "SGMamba": "Mamba++_ca2",
}


def gpu_index(value):
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("GPU index must be nonnegative.")
    return index


def profile_model(options):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.device)

    import torch

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

    from model_all import create_model
    from deepspeed.profiling.flops_profiler import get_model_profile

    model = create_model(registry_name).to(device).eval()

    with torch.no_grad():
        flops, _, params = get_model_profile(
            model,
            args=forward_args,
            warm_up=3,
            print_profile=False,
            as_string=False,
        )

    return {
        "model": options.model,
        "GFLOPs": flops / 2 / 1e9,
        "Params_M": params / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure model FLOPs and parameters on a selected GPU."
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
    result = profile_model(options)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
