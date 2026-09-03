# BiRefNet ONNX export and benchmark

This repository exports the full and lite BiRefNet checkpoints at a fixed batch size of 1 and
1024×1024 resolution, along with including various benchmark setups.

Benchmark results can be found in [results/README.md](./results/README.md)

## Export

The pre-opset-22 exporter supports the baseline and memory-optimized DeformConv lowerings. It
defaults to opset 21:

```bash
python3 birefnet_to_onnx.py CHECKPOINT OUTPUT.onnx \
  --implementation baseline

python3 birefnet_to_onnx.py CHECKPOINT OUTPUT.onnx \
  --implementation optimized
```

Add `--dtype fp16` to either command to statically cast the model weights and input to FP16. FP16
exports are intended for the GPU-only benchmark paths.

Export with the standard ONNX opset-22 `DeformConv` operator using:

```bash
python3 birefnet_to_onnx_22.py CHECKPOINT OUTPUT.onnx
```

The opset-22 exporter also accepts `--dtype fp16`.

Both scripts infer `swin_v1_t` for checkpoint filenames containing `swin_v1_tiny`; other provided
checkpoints use `swin_v1_l`. Use `--image-size` to change the fixed square input size, `--dynamic`
for dynamic batch and spatial axes, or `--skip-runtime-check` to omit the CPU ONNX Runtime check.

Generated models are stored in `onnx_models/`.

## Benchmark

Run the complete batch-size-1 Torch and ONNX Runtime matrix with warmups:

```bash
python3 benchmark.py --warmup-runs 3 --runs 10
```

Torch uses `torch.compile` by default. Add `--no-torch-compile` for optimized eager execution. CPU
uses FP32. CUDA measures both FP32 weights under modern `torch.amp.autocast("cuda",
dtype=torch.float16)` and a separate static-FP16 weight cast. ONNX FP16 models are likewise run only
on CUDA; FP32 ONNX models run on both CPU and CUDA. Each case runs in an isolated process and
records average latency, standard deviation, peak process RSS, and Torch peak CUDA allocation where
applicable.

For CUDA, the benchmark preloads ONNX Runtime's CUDA and cuDNN libraries with
`ort.preload_dlls(directory="")` before creating the execution provider.

Skip every GPU measurement on a CPU-only host:

```bash
python3 benchmark.py --skip-gpu
```

Skip ONNX GPU execution only for selected models by repeating a filename-stem glob:

```bash
python3 benchmark.py \
  --disable-onnx-gpu 'birefnet-full-baseline*' \
  --disable-onnx-gpu 'birefnet-full-optimized*'
```

Results are written to `benchmark_results.json` unless changed with `--output`.

## AI Acknowledgement

GPT-5.6 Sol in Codex was used to clean up the export code and to prepare the benchmark, with all code human-verified for correctness.
