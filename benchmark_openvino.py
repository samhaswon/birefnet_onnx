#!/usr/bin/env python3
"""Benchmark batch-one ONNX Runtime OpenVINO GPU inference in isolated processes."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmark import peak_rss_mib


ROOT = Path(__file__).resolve().parent
PROVIDER = "OpenVINOExecutionProvider"


def parse_args() -> argparse.Namespace:
    """Parse and validate options.

    :return: Benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "onnx_models")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_openvino_results.json")
    parser.add_argument("--device", default="GPU.0", help="GPU or GPU.N; CPU/AUTO are forbidden.")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--include-opset22", action="store_true")
    parser.add_argument(
        "--allow-dynamic-shapes", action="store_true",
        help="Use OpenVINO dynamic compilation instead of specializing subgraph inputs.",
    )
    parser.add_argument("--fp32-only", action="store_true", help="Skip static FP16 exports.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"GPU(?:\.\d+)?", args.device):
        parser.error("--device must be GPU or GPU.N (for example GPU.0)")
    if args.image_size <= 0 or args.runs <= 0 or args.warmup_runs < 0:
        parser.error("image size and runs must be positive; warmup runs must be nonnegative")
    if args.worker and args.model is None:
        parser.error("--worker requires --model")
    return args


def benchmark_model(args: argparse.Namespace) -> dict:
    """Measure synchronous GPU-targeted inference with CPU fallback allowed.

    :param args: Worker options.
    :return: Latency and process memory measurements.
    """
    import onnxruntime as ort

    if PROVIDER not in ort.get_available_providers():
        raise RuntimeError(
            "OpenVINOExecutionProvider unavailable; install requirements-openvino.txt"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    provider_options = {
        "device_type": args.device,
        "disable_dynamic_shapes": "False" if args.allow_dynamic_shapes else "True",
    }
    session = ort.InferenceSession(
        str(args.model), sess_options=options,
        providers=[(PROVIDER, provider_options), "CPUExecutionProvider"],
    )
    session.disable_fallback()
    if PROVIDER not in session.get_providers():
        raise RuntimeError("ONNX Runtime did not activate OpenVINOExecutionProvider")
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Expected one image input")
    input_meta = inputs[0]
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
    if input_meta.type not in dtypes:
        raise ValueError(f"Unsupported image input type: {input_meta.type}")
    shape = (1, 3, args.image_size, args.image_size)
    if len(input_meta.shape) != 4 or any(
        isinstance(actual, int) and actual != requested
        for actual, requested in zip(input_meta.shape, shape)
    ):
        raise ValueError(f"Model input {input_meta.shape} does not match requested shape {shape}")
    sample = np.random.default_rng(0).standard_normal(shape).astype(dtypes[input_meta.type])
    feeds = {input_meta.name: sample}
    for _ in range(max(1, args.warmup_runs)):
        session.run(None, feeds)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "average_ms": float(np.mean(timings)),
        "stddev_ms": float(np.std(timings)),
        "peak_process_rss_mib": peak_rss_mib(),
        "peak_gpu_allocated_mib": None,
        "onnxruntime_version": ort.__version__,
        "input_dtype": input_meta.type,
        "provider_options": provider_options,
    }


def invoke_worker(args: argparse.Namespace, model: Path) -> dict:
    """Run one model in a fresh process.

    :param args: Benchmark options.
    :param model: ONNX model path.
    :return: Measurements or failure details.
    """
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--model", str(model),
        "--device", args.device, "--image-size", str(args.image_size),
        "--warmup-runs", str(args.warmup_runs), "--runs", str(args.runs),
    ]
    if args.allow_dynamic_shapes:
        command.append("--allow-dynamic-shapes")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("BENCHMARK_JSON="):
                return json.loads(line.partition("=")[2])
    return {
        "status": "failed",
        "error": (completed.stderr or completed.stdout or
                  f"Worker exited with code {completed.returncode}").strip()[-4000:],
    }


def main(args: argparse.Namespace) -> None:
    """Select models, launch workers, and write results.

    :param args: Benchmark options.
    """
    import onnx

    models = sorted(args.models_dir.glob("*.onnx"))
    if not models:
        raise FileNotFoundError(f"No ONNX models found in {args.models_dir}")
    results = []
    for model in models:
        label = {"model": model.stem, "engine": "onnxruntime-openvino", "device": args.device}
        try:
            metadata = onnx.load(str(model), load_external_data=False)
            opsets = [entry.version for entry in metadata.opset_import
                      if entry.domain in ("", "ai.onnx")]
            is_fp16 = metadata.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
            del metadata
            label["opsets"] = opsets
            reason = None
            if 22 in opsets and not args.include_opset22:
                reason = "opset 22 is currently unsupported; enable with --include-opset22"
            elif args.fp32_only and is_fp16:
                reason = "--fp32-only"
            if reason:
                result = {"status": "skipped", "reason": reason}
            else:
                print(f"Benchmarking {model.name} on {args.device}", flush=True)
                result = invoke_worker(args, model.resolve())
        except Exception as error:
            result = {"status": "failed", "error": str(error)}
        results.append({**label, **result})
        print(f"{model.name}: {result['status']}", flush=True)
    payload = {
        "image_size": args.image_size, "batch_size": 1,
        "warmup_runs": max(1, args.warmup_runs), "measured_runs": args.runs,
        "provider": PROVIDER, "device": args.device, "cpu_fallback": True,
        "include_opset22": args.include_opset22, "fp32_only": args.fp32_only,
        "disable_dynamic_shapes": not args.allow_dynamic_shapes,
        "memory_note": "Peak process RSS includes session creation and warmups. "
                       "OpenVINO EP exposes no portable peak GPU allocation counter; "
                       "GPU allocation is unmeasured (null).",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        print("BENCHMARK_JSON=" + json.dumps(benchmark_model(arguments)))
    else:
        main(arguments)
