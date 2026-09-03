#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python3 birefnet_to_onnx.py \
  checkpoints/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth \
  onnx_models/birefnet-lite-optimized-opset21-fp16.onnx \
  --implementation optimized \
  --opset 21 \
  --image-size 1024 \
  --dtype fp16 \
  --skip-runtime-check

python3 birefnet_to_onnx_22.py \
  checkpoints/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth \
  onnx_models/birefnet-lite-opset22-fp16.onnx \
  --image-size 1024 \
  --dtype fp16 \
  --skip-runtime-check

python3 birefnet_to_onnx.py \
  checkpoints/BiRefNet-general-epoch_244.pth \
  onnx_models/birefnet-full-baseline-opset21-fp16.onnx \
  --implementation baseline \
  --opset 21 \
  --image-size 1024 \
  --dtype fp16 \
  --skip-runtime-check

python3 birefnet_to_onnx.py \
  checkpoints/BiRefNet-general-epoch_244.pth \
  onnx_models/birefnet-full-optimized-opset21-fp16.onnx \
  --implementation optimized \
  --opset 21 \
  --image-size 1024 \
  --dtype fp16 \
  --skip-runtime-check

python3 birefnet_to_onnx_22.py \
  checkpoints/BiRefNet-general-epoch_244.pth \
  onnx_models/birefnet-full-opset22-fp16.onnx \
  --image-size 1024 \
  --dtype fp16 \
  --skip-runtime-check
