#!/usr/bin/env python3
"""
Export BiRefNet_lite-2K (trained with the provided HF adapter) to ONNX.

- Loads a .safetensors or .bin/.pt state dict from a checkpoint directory or file
- Rebuilds the same adapter around the HF model
- Exports a single output named "logits" shaped (N, 1, H, W)

Requirements:
pip install torch onnx onnxruntime safetensors transformers pillow
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

from model import BiRefNet

# -----------------------
# Global configuration
# -----------------------
CHECKPOINT_PATH: Union[str, Path] = "/home/samuel/ai_data/skin_models/BiRefNet/birefnet_state.pth.tar"  # dir with model.safetensors or a file path
# CHECKPOINT_PATH: Union[str, Path] = "/home/samuel/ai_data/mq_models/birefnet/checkpoint_85.pth.tar"  # dir with model.safetensors or a file path
BASE_MODEL_ID: str = "ZhengPeng7/BiRefNet_lite-2K"
OUTPUT_ONNX_PATH: Union[str, Path] = "/home/samuel/ai_data/skin_models/BiRefNet/birefnet.onnx"
# OUTPUT_ONNX_PATH: Union[str, Path] = "/home/samuel/ai_data/mq_models/birefnet/birefnet_seams_fp16.onnx"

# Input and export knobs
IMAGE_SIZE: int = 1728           # used only when USE_DYNAMIC_AXES=False
BATCH_SIZE: int = 1              # used only when USE_DYNAMIC_AXES=False
USE_DYNAMIC_AXES: bool = False    # True -> dynamic N, H, W
OPSET: int = 21
DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float32  # keep fp32 for widest runtime support


# -----------------------
# Model adapter
# -----------------------
class BiRefNetHFAdapter(nn.Module):
    """
    Wraps the base model to present {'logits': (N, 1, H, W)}.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values)[0]
        out = torch.sigmoid(out)
        # Keep the exported output in the model dtype rather than promoting an
        # FP16 graph through a float32 scalar constant.
        out = torch.where(out > 0.5, out, torch.zeros_like(out))
        return out


def _make_dummy_input(
    batch: int, size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    size = max(32, (size + 31) // 32 * 32)
    return torch.zeros(batch, 3, size, size, device=device, dtype=dtype)


# -----------------------
# Export
# -----------------------
def export_onnx() -> None:
    """
    Build model, load weights, and export to ONNX.
    """
    test_image = cv2.imread("images/00712.png")
    test_image_tensor = (
        cv2.resize(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE)
                   )).astype(np.float32) / 255.0
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # test_image_tensor = (test_image_tensor - mean) / std
    test_image_tensor = torch.from_numpy(test_image_tensor.transpose(2, 0, 1))

    out_path = Path(OUTPUT_ONNX_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    net = BiRefNet()
    net.load_state_dict(checkpoint["state"]["state_dict"])

    model = BiRefNetHFAdapter(net)
    if DTYPE == torch.float16:
        test_image_tensor = test_image_tensor.half()
        model = model.half()

    model.eval()
    with torch.no_grad():
        if USE_DYNAMIC_AXES:
            dummy = _make_dummy_input(1, IMAGE_SIZE, device=torch.device(DEVICE), dtype=DTYPE)
            dynamic_axes = {
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            }
        else:
            dummy = _make_dummy_input(BATCH_SIZE, IMAGE_SIZE, device=torch.device(DEVICE), dtype=DTYPE)
            dynamic_axes = None

        # Quick model test
        test_out = model(test_image_tensor.unsqueeze(0))
        print(f"Model out shape: {test_out.shape}")
        test_out = (test_out.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
        cv2.imwrite(
            "test/hf_test.png",
            np.dstack(
                (test_image,
                 cv2.resize(
                     (test_out.transpose(1, 2, 0) * 255).astype(np.uint8),
                     test_image.shape[::-1][1:]
                 )
                 )
            )
        )

        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )
    print("Model exported to ONNX")
    # Optional sanity check with onnxruntime if present
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        test_in = dummy.detach().cpu().numpy()
        ort_out = sess.run(None, {"pixel_values": test_in})[0]
        if ort_out.ndim != 4 or ort_out.shape[1] != 1:
            raise RuntimeError(f"Unexpected ONNX output shape: {ort_out.shape}")
    except Exception as ex:
        # Do not fail the export if runtime isn't available, but surface real issues.
        msg = str(ex)
        if "No module named" in msg or "Cannot find module" in msg:
            pass
        else:
            raise

    print(f"OK: wrote {out_path.resolve()}")


if __name__ == "__main__":
    export_onnx()
