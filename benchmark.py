#!/usr/bin/env python3
"""Benchmark batch-one BiRefNet Torch and ONNX inference in isolated processes."""

import argparse
import contextlib
import fnmatch
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse benchmark options.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "onnx_models")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results.json")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--disable-onnx-gpu",
        action="append",
        default=[],
        metavar="GLOB",
        help="Skip matching ONNX filename stems on GPU; may be repeated.",
    )
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=("torch", "onnx"), help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--torch-mode", choices=("fp32", "amp-fp16", "static-fp16"),
        default="fp32", help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def peak_rss_mib() -> float:
    """Return this process's peak resident memory in MiB.

    :return: Peak RSS in MiB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def summarize(name: str, engine: str, device: str, timings: list[float],
              gpu_peak_mib: float | None) -> dict:
    """Build a serializable benchmark result.

    :param name: Model name.
    :param engine: Inference engine.
    :param device: Execution device.
    :param timings: Per-run durations in seconds.
    :param gpu_peak_mib: Peak framework-managed CUDA memory.
    :return: Benchmark result.
    """
    values = np.asarray(timings) * 1000
    return {
        "model": name,
        "engine": engine,
        "device": device,
        "status": "ok",
        "average_ms": float(values.mean()),
        "stddev_ms": float(values.std()),
        "peak_process_rss_mib": peak_rss_mib(),
        "peak_cuda_allocated_mib": gpu_peak_mib,
    }


def benchmark_torch(args: argparse.Namespace) -> dict:
    """Run the optimized Torch benchmark.

    :param args: Worker options.
    :return: Benchmark result.
    """
    import torch
    from birefnet_to_onnx import load_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model = load_model(args.checkpoint.resolve(), device).to(memory_format=torch.channels_last)
    if args.torch_mode != "fp32" and device.type != "cuda":
        raise ValueError(f"{args.torch_mode} is only supported on CUDA")
    dtype = torch.float16 if args.torch_mode == "static-fp16" else torch.float32
    if args.torch_mode == "static-fp16":
        model = model.to(dtype=dtype)
    sample = torch.randn(
        1, 3, args.image_size, args.image_size, device=device, dtype=dtype
    )
    sample = sample.to(memory_format=torch.channels_last)
    if not args.no_torch_compile:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    autocast = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if args.torch_mode == "amp-fp16"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        for _ in range(args.warmup_runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.runs):
            started = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - started)
    gpu_peak = None
    if device.type == "cuda":
        gpu_peak = torch.cuda.max_memory_allocated(device) / 2**20
    execution = "compiled" if not args.no_torch_compile else "optimized-eager"
    engine = f"torch-{execution}-{args.torch_mode}"
    return summarize(args.checkpoint.stem, engine, args.device, timings, gpu_peak)


def benchmark_onnx(args: argparse.Namespace) -> dict:
    """Run an ONNX Runtime benchmark.

    :param args: Worker options.
    :return: Benchmark result.
    """
    import onnxruntime as ort

    if args.device == "cuda":
        ort.preload_dlls(directory="")
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"{provider} is unavailable")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=providers
    )
    if provider not in session.get_providers():
        raise RuntimeError(f"ONNX Runtime did not activate {provider}")
    input_meta = session.get_inputs()[0]
    input_dtype = np.float16 if input_meta.type == "tensor(float16)" else np.float32
    sample = np.random.randn(1, 3, args.image_size, args.image_size).astype(input_dtype)
    feeds = {input_meta.name: sample}
    for _ in range(args.warmup_runs):
        session.run(None, feeds)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        session.run(None, feeds)
        timings.append(time.perf_counter() - started)
    return summarize(args.model.stem, "onnxruntime", args.device, timings, None)


def run_worker(args: argparse.Namespace) -> None:
    """Execute one isolated measurement and print JSON.

    :param args: Worker options.
    """
    result = benchmark_torch(args) if args.engine == "torch" else benchmark_onnx(args)
    print("BENCHMARK_JSON=" + json.dumps(result))


def invoke_worker(args: argparse.Namespace, extra: list[str], label: dict) -> dict:
    """Invoke and collect one benchmark subprocess.

    :param args: Parent benchmark options.
    :param extra: Worker-specific arguments.
    :param label: Fields identifying a failed result.
    :return: Benchmark result.
    """
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--image-size", str(args.image_size), "--warmup-runs", str(args.warmup_runs),
        "--runs", str(args.runs), *extra,
    ]
    if args.no_torch_compile:
        command.append("--no-torch-compile")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    for line in completed.stdout.splitlines():
        if line.startswith("BENCHMARK_JSON="):
            return json.loads(line.partition("=")[2])
    return {
        **label,
        "status": "failed",
        "error": (completed.stderr or completed.stdout).strip()[-2000:],
    }


def skipped(model: str, engine: str, device: str, reason: str) -> dict:
    """Create a skipped result entry."""
    return {
        "model": model, "engine": engine, "device": device,
        "status": "skipped", "reason": reason,
    }


def main(args: argparse.Namespace) -> None:
    """Run the benchmark matrix and write JSON results.

    :param args: Parent benchmark options.
    """
    checkpoints = sorted((ROOT / "checkpoints").glob("*.pth"))
    models = sorted(args.models_dir.glob("*.onnx"))
    results = []
    for checkpoint in checkpoints:
        torch_cases = [("cpu", "fp32"), ("cuda", "amp-fp16"), ("cuda", "static-fp16")]
        for device, torch_mode in torch_cases:
            execution = "compiled" if not args.no_torch_compile else "optimized-eager"
            engine = f"torch-{execution}-{torch_mode}"
            label = {"model": checkpoint.stem, "engine": engine, "device": device}
            if device == "cuda" and args.skip_gpu:
                results.append(skipped(**label, reason="--skip-gpu"))
                continue
            print(f"Benchmarking {label}", flush=True)
            results.append(invoke_worker(
                args,
                [
                    "--engine", "torch", "--device", device,
                    "--torch-mode", torch_mode, "--checkpoint", str(checkpoint),
                ],
                label,
            ))
    for model in models:
        for device in ("cpu", "cuda"):
            label = {"model": model.stem, "engine": "onnxruntime", "device": device}
            is_static_fp16 = model.stem.endswith("-fp16")
            if device == "cpu" and is_static_fp16:
                results.append(skipped(
                    **label, reason="static FP16 models are GPU-only benchmark paths"
                ))
                continue
            disabled = any(
                fnmatch.fnmatch(model.stem, pattern) for pattern in args.disable_onnx_gpu
            )
            if device == "cuda" and (args.skip_gpu or disabled):
                reason = "--skip-gpu" if args.skip_gpu else "matched --disable-onnx-gpu"
                results.append(skipped(**label, reason=reason))
                continue
            print(f"Benchmarking {label}", flush=True)
            results.append(invoke_worker(
                args, ["--engine", "onnx", "--device", device, "--model", str(model)], label
            ))
    payload = {
        "image_size": args.image_size,
        "batch_size": 1,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.runs,
        "torch_compile": not args.no_torch_compile,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    options = parse_args()
    if options.worker:
        run_worker(options)
    else:
        main(options)
