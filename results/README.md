# BiRefNet benchmark results

All timings are average batch-size-1 inference latency at 1024×1024, using 3 warmup runs and
10 measured runs. Lower is better. Speedup compares the FP32 CPU result with the fastest completed
GPU result in the same row.

## Sleepy

Ryzen 7 5700X, RTX 3060 12 GB, 48 GB DDR4-3000.

### Compiled Torch

| Model | CPU FP32 | GPU AMP FP16 | GPU static FP16 | Best GPU speedup |
|---|---:|---:|---:|---:|
| Lite | 7,037.3 ms | 101.6 ms | **78.6 ms** | **89.6×** |
| Full | 11,368.8 ms | 247.0 ms | **215.4 ms** | **52.8×** |

Static FP16 was 22.6% faster than AMP for Lite and 12.8% faster for Full. It also approximately
halved peak CUDA allocation versus AMP: 775 vs 1,501 MiB for Lite and 1,804 vs 2,218 MiB for Full.

### ONNX Runtime

| Model | CPU FP32 | GPU FP32 | GPU FP16 | Best GPU speedup |
|---|---:|---:|---:|---:|
| Lite, baseline opset 21 | 6,655.6 ms | 370.2 ms | **308.6 ms** | **21.6×** |
| Lite, optimized opset 21 | 6,714.0 ms | 386.3 ms | **315.2 ms** | **21.3×** |
| Lite, native opset 22 | 2,149.7 ms | 212.1 ms | **139.6 ms** | **15.4×** |
| Full, baseline opset 21 | 11,030.7 ms | 765.4 ms | **584.0 ms** | **18.9×** |
| Full, optimized opset 21 | 10,832.1 ms | 781.3 ms | **590.9 ms** | **18.3×** |
| Full, native opset 22 | 8,539.9 ms | 606.7 ms | **415.7 ms** | **20.5×** |

Native opset 22 was the fastest ONNX implementation for both model sizes. Its FP16 GPU graph was
2.21× faster than the next-fastest Lite ONNX result and 1.41× faster than the next-fastest Full
ONNX result.

### Fastest Torch vs ONNX Runtime

This table compares the fastest completed configuration from each runtime on each device. The
configuration is included because the winning precision differs between runtimes.

| Model | Device | Fastest Torch | Fastest ONNX | Winner |
|---|---|---:|---:|---:|
| Lite | CPU | 7,037.3 ms (FP32) | **2,149.7 ms** (opset 22 FP32) | **ONNX, 3.27×** |
| Lite | GPU | **78.6 ms** (static FP16) | 139.6 ms (opset 22 FP16) | **Torch, 1.78×** |
| Full | CPU | 11,368.8 ms (FP32) | **8,539.9 ms** (opset 22 FP32) | **ONNX, 1.33×** |
| Full | GPU | **215.4 ms** (static FP16) | 415.7 ms (opset 22 FP16) | **Torch, 1.93×** |

## Legion

Ryzen 7 4800H, GTX 1650 Ti 4 GB, 64 GB DDR4-3200. `DNF` means the GPU run did not finish because
the GPU did not have enough VRAM.

### Compiled Torch

| Model | CPU FP32 | GPU AMP FP16 | GPU static FP16 | Best GPU speedup |
|---|---:|---:|---:|---:|
| Lite | 13,292.9 ms | **1,285.4 ms** | 1,589.0 ms | **10.3×** |
| Full | 15,815.7 ms | **5,766.5 ms** | 6,023.2 ms | **2.7×** |

Unlike the RTX 3060, the GTX 1650 Ti ran AMP faster than static FP16: 19.1% faster for Lite and
4.3% faster for Full. Static FP16 still reduced peak CUDA allocation from 1,501 to 776 MiB for
Lite and from 2,217 to 1,803 MiB for Full.

### ONNX Runtime

| Model | CPU FP32 | GPU FP32 | GPU FP16 | Best GPU speedup |
|---|---:|---:|---:|---:|
| Lite, baseline opset 21 | 7,061.3 ms | DNF | **2,050.5 ms** | **3.4×** |
| Lite, optimized opset 21 | 7,154.9 ms | DNF | **2,068.1 ms** | **3.5×** |
| Lite, native opset 22 | 3,894.7 ms | **577.2 ms** | 1,717.2 ms | **6.7×** |
| Full, baseline opset 21 | 14,170.3 ms | DNF | DNF | DNF |
| Full, optimized opset 21 | 13,845.4 ms | DNF | DNF | DNF |
| Full, native opset 22 | 10,174.2 ms | DNF | **7,147.8 ms** | **1.4×** |

Native opset 22 was again the fastest successful ONNX path. On this older GPU, its Lite FP32 graph
was unexpectedly 3.0× faster than its static-FP16 graph; FP16 should therefore not be assumed to
improve latency on hardware without strong native FP16 throughput.

### Fastest Torch vs ONNX Runtime

| Model | Device | Fastest Torch | Fastest ONNX | Winner |
|---|---|---:|---:|---:|
| Lite | CPU | 13,292.9 ms (FP32) | **3,894.7 ms** (opset 22 FP32) | **ONNX, 3.41×** |
| Lite | GPU | 1,285.4 ms (AMP FP16) | **577.2 ms** (opset 22 FP32) | **ONNX, 2.23×** |
| Full | CPU | 15,815.7 ms (FP32) | **10,174.2 ms** (opset 22 FP32) | **ONNX, 1.55×** |
| Full | GPU | **5,766.5 ms** (AMP FP16) | 7,147.8 ms (opset 22 FP16) | **Torch, 1.24×** |

For Legion, the comparison uses only configurations that completed. In particular, the Full ONNX
GPU value is the sole successful Full ONNX GPU path, while the faster-looking alternatives were
`DNF` because of the 4 GB VRAM limit.
