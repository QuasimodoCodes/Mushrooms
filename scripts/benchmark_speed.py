"""
Inference speed benchmark — all project models.

Tests each model with a single dummy image (batch=1), matching real-world
usage in the Vision API where one image is processed per request.

Measurements:
  - GPU latency  (CUDA, PyTorch models)
  - CPU latency  (all models — TFLite runs CPU-only in production)

Run from project root:
    python scripts/benchmark_speed.py
"""

import os
import sys
import time
import statistics

import torch
import numpy as np

import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

WARMUP_RUNS = 50
BENCH_RUNS  = 200
IMG_SIZE    = 224
NUM_CLASSES = 169

# ── Model weight paths ────────────────────────────────────────────────────────
WEIGHTS = {
    "YOLOv26n-cls":         os.path.join(_ROOT, "..", "docs", "yolo_runs", "yolo26_classifier_v1", "weights", "best.pt"),
    "YOLOv8n-cls":          os.path.join(_ROOT, "..", "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt"),
    "EfficientNet-B0":      os.path.join(_ROOT, "..", "docs", "cnn_runs", "efficientnet_b0_adamw_ce_smooth", "weights", "best.pt"),
    "ConvNeXt-Tiny (PEFT)": os.path.join(_ROOT, "..", "docs", "herman_runs", "convnext_tiny_adamw_ce_smooth_peft", "weights", "best.pt"),
}
TFLITE_PATH = os.path.join(_ROOT, "..", "docs", "yolo_runs", "yolo26_tflite", "weights", "best_float16.tflite")


# ── Helpers ───────────────────────────────────────────────────────────────────

def bench_pytorch(model, device, label):
    """Benchmark a PyTorch model on the given device. Returns (mean_ms, std_ms)."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # benchmark
    times = []
    with torch.no_grad():
        for _ in range(BENCH_RUNS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean = statistics.mean(times)
    std  = statistics.stdev(times)
    print(f"  [{label:>20}] {device.upper():4}  {mean:7.3f} ms  ± {std:.3f} ms")
    return mean, std


def bench_yolo(weight_path, name):
    """Benchmark an Ultralytics YOLO model. Returns (gpu_ms, cpu_ms)."""
    from ultralytics import YOLO
    results = {}
    for device in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        model = YOLO(weight_path)
        dummy_np = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # warmup
        for _ in range(WARMUP_RUNS):
            model.predict(dummy_np, device=device, verbose=False)

        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            model.predict(dummy_np, device=device, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

        mean = statistics.mean(times)
        std  = statistics.stdev(times)
        print(f"  [{name:>20}] {device.upper():4}  {mean:7.3f} ms  ± {std:.3f} ms")
        results[device] = (mean, std)
    return results


def bench_tflite(weight_path):
    """Benchmark the TFLite float16 model on CPU."""
    try:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        interp = Interpreter(model_path=weight_path)
        interp.allocate_tensors()
        inp  = interp.get_input_details()[0]
        out  = interp.get_output_details()[0]

        # TFLite float16 models still take float32 input on CPU
        dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

        # warmup
        for _ in range(WARMUP_RUNS):
            interp.set_tensor(inp["index"], dummy)
            interp.invoke()

        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            interp.set_tensor(inp["index"], dummy)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        mean = statistics.mean(times)
        std  = statistics.stdev(times)
        print(f"  [{'YOLOv26n TFLite f16':>20}]  CPU  {mean:7.3f} ms  ± {std:.3f} ms")
        return mean, std

    except Exception as e:
        print(f"  [{'YOLOv26n TFLite f16':>20}]  CPU  SKIPPED ({e})")
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Inference Speed Benchmark")
    print(f"  {WARMUP_RUNS} warmup + {BENCH_RUNS} timed runs | batch=1 | {IMG_SIZE}×{IMG_SIZE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    results = {}

    # ── EfficientNet-B0 ───────────────────────────────────────────────────────
    print("\n[EfficientNet-B0]")
    cnn_mod = _load_module("cnn_model", os.path.join(_ROOT, "training", "cnn", "model.py"))
    eff = cnn_mod.build_efficientnet_b0(num_classes=NUM_CLASSES)
    eff.load_state_dict(torch.load(WEIGHTS["EfficientNet-B0"], map_location="cpu"))
    eff_gpu = eff_cpu = None
    if torch.cuda.is_available():
        eff_gpu, _ = bench_pytorch(eff, "cuda", "EfficientNet-B0")
    eff_cpu, _ = bench_pytorch(eff, "cpu", "EfficientNet-B0")
    results["EfficientNet-B0"] = {"gpu": eff_gpu, "cpu": eff_cpu}

    # ── ConvNeXt-Tiny PEFT ────────────────────────────────────────────────────
    print("\n[ConvNeXt-Tiny PEFT]")
    cnx_mod = _load_module("cnx_model", os.path.join(_ROOT, "training", "convnext", "model.py"))
    cnx = cnx_mod.build_convnext_tiny(num_classes=NUM_CLASSES)
    cnx.load_state_dict(torch.load(WEIGHTS["ConvNeXt-Tiny (PEFT)"], map_location="cpu"))
    cnx_gpu = cnx_cpu = None
    if torch.cuda.is_available():
        cnx_gpu, _ = bench_pytorch(cnx, "cuda", "ConvNeXt-Tiny")
    cnx_cpu, _ = bench_pytorch(cnx, "cpu", "ConvNeXt-Tiny")
    results["ConvNeXt-Tiny (PEFT)"] = {"gpu": cnx_gpu, "cpu": cnx_cpu}

    # ── YOLO models ───────────────────────────────────────────────────────────
    print("\n[YOLOv26n-cls (PyTorch)]")
    yolo26_res = bench_yolo(WEIGHTS["YOLOv26n-cls"], "YOLOv26n-cls")
    results["YOLOv26n-cls"] = {"gpu": yolo26_res.get("cuda", (None,))[0],
                                "cpu": yolo26_res.get("cpu",  (None,))[0]}

    print("\n[YOLOv8n-cls (PyTorch)]")
    yolo8_res = bench_yolo(WEIGHTS["YOLOv8n-cls"], "YOLOv8n-cls")
    results["YOLOv8n-cls"] = {"gpu": yolo8_res.get("cuda", (None,))[0],
                               "cpu": yolo8_res.get("cpu",  (None,))[0]}

    # ── TFLite ────────────────────────────────────────────────────────────────
    print("\n[YOLOv26n TFLite float16 — CPU only (production format)]")
    tfl_mean, _ = bench_tflite(TFLITE_PATH)
    results["YOLOv26n TFLite f16"] = {"gpu": None, "cpu": tfl_mean}

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY  (batch=1, ms/image, lower = faster)")
    print("=" * 65)
    header = f"  {'Model':<26}  {'Top-1':>6}  {'GPU (ms)':>10}  {'CPU (ms)':>10}"
    print(header)
    print("  " + "-" * 60)

    top1 = {
        "YOLOv8n-cls":          "86.8%",
        "YOLOv26n-cls":         "88.1%",
        "EfficientNet-B0":      "89.5%",
        "ConvNeXt-Tiny (PEFT)": "91.1%",
        "YOLOv26n TFLite f16":  "88.1%",
    }

    for name, vals in results.items():
        gpu_s = f"{vals['gpu']:>8.3f}" if vals["gpu"] is not None else "     N/A"
        cpu_s = f"{vals['cpu']:>8.3f}" if vals["cpu"] is not None else "     N/A"
        acc   = top1.get(name, "—")
        print(f"  {name:<26}  {acc:>6}  {gpu_s}  {cpu_s}")

    print("=" * 65)
    print("  * TFLite runs CPU-only (production deployment on Cloud Run)")
    print("  * YOLO times include Ultralytics preprocessing overhead")


if __name__ == "__main__":
    main()
