#!/usr/bin/env python3
"""Export BiRefNet checkpoints with an ONNX-compatible deformable convolution."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BIREFNET_ROOT = ROOT / "BiRefNet"


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--implementation", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--opset", type=int, default=21, choices=range(11, 22))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--skip-runtime-check", action="store_true")
    return parser.parse_args()


def configure_birefnet(checkpoint: Path) -> None:
    """Add upstream sources to the import path and select the checkpoint backbone.

    :param checkpoint: Checkpoint whose filename identifies the backbone.
    """
    sys.path.insert(0, str(BIREFNET_ROOT))
    import config

    original_config = config.Config
    backbone = "swin_v1_t" if "swin_v1_tiny" in checkpoint.name else "swin_v1_l"

    class ExportConfig(original_config):
        def __init__(self) -> None:
            super().__init__()
            self.bb = backbone
            channels = {
                "swin_v1_t": [768, 384, 192, 96],
                "swin_v1_l": [1536, 768, 384, 192],
            }[backbone]
            self.lateral_channels_in_collection = [channel * 2 for channel in channels]
            self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:]

    config.Config = ExportConfig


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load a canonical BiRefNet model and its weights.

    :param checkpoint: Path to a plain PyTorch state dictionary.
    :param device: Export device.
    :return: Model in evaluation mode.
    """
    configure_birefnet(checkpoint)
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(check_state_dict(state_dict))
    return model.eval().to(device)


def register_deform_conv(implementation: str, opset: int) -> None:
    """Register one of the two local DeformConv graph lowerings.

    :param implementation: ``baseline`` or memory-optimized lowering.
    :param opset: Target ONNX opset.
    """
    if implementation == "baseline":
        import deform_conv2d_onnx_exporter as exporter
    else:
        import modified_deform_conv2d_onnx_exporter as exporter
    exporter.onnx_opset_version = opset
    exporter.register_deform_conv2d_onnx_op()


def export_model(args: argparse.Namespace) -> None:
    """Export and validate one model.

    :param args: Parsed command-line options.
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but PyTorch cannot access a CUDA device")
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("--image-size must be at least 32 and divisible by 32")

    register_deform_conv(args.implementation, args.opset)
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input_image": {0: "batch", 2: "height", 3: "width"},
            "output_image": {0: "batch", 2: "height", 3: "width"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(example)[-1]
        torch.onnx.export(
            model,
            example,
            str(args.output),
            opset_version=args.opset,
            input_names=["input_image"],
            output_names=["output_image"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )

    import onnx

    onnx.checker.check_model(onnx.load(str(args.output)))
    if not args.skip_runtime_check:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input_image": example.cpu().numpy()})[0]
        np.testing.assert_allclose(actual, expected.cpu().numpy(), rtol=2e-3, atol=2e-4)
    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    export_model(parse_args())
