#!/usr/bin/env python3
"""Export BiRefNet with the standard ONNX opset-22 DeformConv operator."""

import argparse
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

from birefnet_to_onnx import load_model


def parse_cli() -> argparse.Namespace:
    """Parse command-line options.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--skip-runtime-check", action="store_true")
    return parser.parse_args()


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def symbolic_deform_conv(
    graph, input_tensor, weight, offset, mask, bias, stride_h, stride_w, pad_h,
    pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask,
):
    """Lower torchvision DeformConv to the standard opset-22 operator."""
    inputs = [input_tensor, weight, offset]
    if not bias.node().mustBeNone():
        inputs.append(bias)
    elif use_mask:
        inputs.append(graph.op("Constant", value_t=torch.tensor([], dtype=torch.float32)))
    if use_mask:
        inputs.append(mask)
    return graph.op(
        "DeformConv",
        *inputs,
        strides_i=[stride_h, stride_w],
        pads_i=[pad_h, pad_w, pad_h, pad_w],
        dilations_i=[dilation_h, dilation_w],
        group_i=groups,
        offset_group_i=offset_groups,
    )


def export_model(args: argparse.Namespace) -> None:
    """Export and validate one opset-22 model.

    :param args: Parsed command-line options.
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but PyTorch cannot access a CUDA device")
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("--image-size must be at least 32 and divisible by 32")
    register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_deform_conv, 22)

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
        torch.onnx.export(
            model,
            example,
            str(args.output),
            opset_version=22,
            input_names=["input_image"],
            output_names=["output_image"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )

    import onnx

    onnx.checker.check_model(onnx.load(str(args.output)))
    if not args.skip_runtime_check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        output = session.run(None, {"input_image": example.cpu().numpy()})[-1]
        if output.shape != (1, 1, args.image_size, args.image_size):
            raise RuntimeError(f"Unexpected output shape: {output.shape}")
    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    export_model(parse_cli())
