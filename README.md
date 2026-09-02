# BiRefNet ONNX export and benchmark

This repository exports the full and lite BiRefNet checkpoints at a fixed batch size of 1 and
1024×1024 resolution. The model output is the raw final logit tensor, matching
`BiRefNet/tutorials/BiRefNet_pth2onnx.ipynb`.

Run commands from the repository root with the prepared `.venv` environment.

## Export

The pre-opset-22 exporter supports the baseline and memory-optimized DeformConv lowerings. It
defaults to opset 21:

```bash
.venv/bin/python birefnet_to_onnx.py CHECKPOINT OUTPUT.onnx \
  --implementation baseline

.venv/bin/python birefnet_to_onnx.py CHECKPOINT OUTPUT.onnx \
  --implementation optimized
```

Export with the standard ONNX opset-22 `DeformConv` operator using:

```bash
.venv/bin/python birefnet_to_onnx_22.py CHECKPOINT OUTPUT.onnx
```

Both scripts infer `swin_v1_t` for checkpoint filenames containing `swin_v1_tiny`; other provided
checkpoints use `swin_v1_l`. Use `--image-size` to change the fixed square input size, `--dynamic`
for dynamic batch and spatial axes, or `--skip-runtime-check` to omit the CPU ONNX Runtime check.

Generated models are stored in `onnx_models/`.

## Benchmark

Run the complete batch-size-1 Torch and ONNX Runtime matrix with warmups:

```bash
.venv/bin/python benchmark.py --warmup-runs 3 --runs 10
```

Torch uses `torch.compile` by default. Add `--no-torch-compile` for optimized eager execution.
Each case runs in an isolated process and records average latency, standard deviation, peak process
RSS, and Torch peak CUDA allocation where applicable.

Skip every GPU measurement on a CPU-only host:

```bash
.venv/bin/python benchmark.py --skip-gpu
```

Skip ONNX GPU execution only for selected models by repeating a filename-stem glob:

```bash
.venv/bin/python benchmark.py \
  --disable-onnx-gpu 'birefnet-full-baseline*' \
  --disable-onnx-gpu 'birefnet-full-optimized*'
```

Results are written to `benchmark_results.json` unless changed with `--output`. The checked-in
results are the earlier opset-17 benchmark snapshot and were not regenerated for the opset-21
replacement exports.
