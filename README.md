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
uses FP32. CUDA measures FP32, FP32 weights under modern `torch.amp.autocast("cuda",
dtype=torch.float16)` and a separate static-FP16 weight cast. ONNX FP16 models are likewise run only
on CUDA; FP32 ONNX models run on both CPU and CUDA. Each case runs in an isolated process and
records average latency, standard deviation, peak process RSS, and Torch peak CUDA allocation where
applicable. FP32 measurements disable TF32 in both backends.

Run only FP32 cases for both backends (CPU and CUDA):

```bash
python3 benchmark.py --fp32-only --output benchmark_fp32_results.json
```

This excludes AMP and static-FP16 cases. Combine with `--skip-gpu` for CPU-only measurements.

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

## OpenVINO GPU benchmark

Install the separate requirements in a dedicated environment (the OpenVINO wheel replaces the
other ONNX Runtime wheels):

```bash
python3 -m venv .venv-openvino
.venv-openvino/bin/pip install -r requirements-openvino.txt
.venv-openvino/bin/python benchmark_openvino.py --device GPU.0 --warmup-runs 3 --runs 10
```

The host needs Intel GPU drivers and access to the GPU render device; containers also need access
to `/dev/dri`. This uses the `onnxruntime-openvino` package and `device_type` option described in the
[ONNX Runtime OpenVINO EP documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html).
Only GPU-targeted cases run: `--device` accepts `GPU` or an indexed target such as `GPU.0` or `GPU.1`.
CPU fallback for unsupported operations is allowed, but there is no separate CPU benchmark. A
missing OpenVINO provider or failed GPU session is reported as a failure.

Opset 22 models are recorded as skipped by default, using the ONNX metadata rather than filenames.
Use `--include-opset22` to attempt them when support is available. Both FP32 and static FP16 models
are tested; `--fp32-only` skips static FP16 inputs. `--models-dir`, `--image-size`, and `--output`
work as in the main benchmark. Default output is `benchmark_openvino_results.json`.

Each model runs in a fresh process. Results include average latency, latency standard deviation,
and peak process RSS in MiB, including session creation and warmups. Latency measures synchronous
`session.run` calls, including host/device transfers, without model loading or input generation.
GPU allocation is left `null`: this benchmark does not measure device memory, and process RSS is
not a substitute for GPU memory, including on shared-memory iGPUs. OpenVINO manages graph
optimization; ORT graph optimizations are disabled for this benchmark.

By default, the EP uses `disable_dynamic_shapes=True` to specialize subgraphs using their concrete
runtime input shapes. This can avoid Intel GPU compilation failures such as
`get_dims() is called for dynamic shape`, even when the ONNX image input has fixed dimensions:
partitioned subgraphs can still have dynamic inputs. Use `--allow-dynamic-shapes` to restore the
EP's dynamic compilation behavior for comparison. This is a workaround, not a guarantee that every
GPU operation will compile. At least one untimed warmup always runs, even with `--warmup-runs 0`,
so deferred compilation is excluded from measured latency. The effective warmup count and shape
setting are recorded in the results.
