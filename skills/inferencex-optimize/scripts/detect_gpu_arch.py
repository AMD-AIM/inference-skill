#!/usr/bin/env python3
"""Detect GPU model and write architecture specs JSON for roofline analysis.

Outputs: GPU_ARCH_DETECTED=<name> and GPU_ARCH_JSON=<path> to stdout.
"""

import argparse
import json
import os
import subprocess
import sys

# Peak theoretical dense TFLOPS per GPU (no sparsity). Sources:
# MI300X: AMD datasheet + Hot Chips 2024 (CDNA3, gfx942, 750W)
# MI325X: same CDNA3 die as MI300X at higher clocks, 256GB HBM3E (CDNA3, gfx942, 1000W)
# MI350X: AMD MI350X GPU datasheet June 2025 (CDNA4, gfx950, 1000W air-cooled)
# MI355X: AMD press specs (CDNA4, gfx950, 1400W liquid-cooled, ~8% higher clocks than MI350X)
# H100:   NVIDIA H100 datasheet (Hopper, sm_90, 700W)
# H200:   NVIDIA H200 datasheet (Hopper, sm_90, same die as H100, 141GB HBM3e, 700W)
# B200:   NVIDIA B200 datasheet (Blackwell, sm_100, 1000W)
PLATFORM_SPECS = {
    "MI300X": {
        "name": "MI300X", "arch": "gfx942", "mem_bw_gbps": 5325, "memory_gb": 192, "tdp_w": 750,
        "max_achievable_tflops": {
            "matrix_fp16": 1307, "matrix_bf16": 1307, "matrix_tf32": 654,
            "matrix_fp32": 163, "matrix_fp64": 163, "matrix_fp8": 2615,
            "matrix_int8": 2615, "vector_fp16": 163, "vector_bf16": 163,
            "vector_fp32": 163, "vector_fp64": 82,
        },
    },
    "MI325X": {
        "name": "MI325X", "arch": "gfx942", "mem_bw_gbps": 6000, "memory_gb": 256, "tdp_w": 1000,
        "max_achievable_tflops": {
            "matrix_fp16": 1307, "matrix_bf16": 1307, "matrix_tf32": 654,
            "matrix_fp32": 163, "matrix_fp64": 163, "matrix_fp8": 2615,
            "matrix_int8": 2615, "vector_fp16": 163, "vector_bf16": 163,
            "vector_fp32": 163, "vector_fp64": 82,
        },
    },
    "MI350X": {
        "name": "MI350X", "arch": "gfx950", "mem_bw_gbps": 8000, "memory_gb": 288, "tdp_w": 1000,
        "max_achievable_tflops": {
            "matrix_fp16": 2307, "matrix_bf16": 2310, "matrix_fp32": 144,
            "matrix_fp64": 72, "matrix_fp8": 4614, "matrix_fp6": 9228,
            "matrix_fp4": 9228, "matrix_int8": 4614, "vector_fp16": 144,
            "vector_bf16": 144, "vector_fp32": 144, "vector_fp64": 72,
        },
    },
    "MI355X": {
        "name": "MI355X", "arch": "gfx950", "mem_bw_gbps": 8000, "memory_gb": 288, "tdp_w": 1400,
        "max_achievable_tflops": {
            "matrix_fp16": 2500, "matrix_bf16": 2500, "matrix_fp32": 158,
            "matrix_fp64": 79, "matrix_fp8": 5000, "matrix_fp6": 10000,
            "matrix_fp4": 10000, "matrix_int8": 5000, "vector_fp16": 158,
            "vector_bf16": 158, "vector_fp32": 158, "vector_fp64": 79,
        },
    },
    "H100": {
        "name": "H100", "arch": "sm_90", "mem_bw_gbps": 3350, "memory_gb": 80, "tdp_w": 700,
        "max_achievable_tflops": {
            "matrix_fp16": 990, "matrix_bf16": 990, "matrix_tf32": 495,
            "matrix_fp32": 67, "matrix_fp64": 34, "matrix_fp8": 1979,
            "matrix_int8": 1979, "vector_fp16": 134, "vector_bf16": 134,
            "vector_fp32": 67, "vector_fp64": 34,
        },
    },
    "H200": {
        "name": "H200", "arch": "sm_90", "mem_bw_gbps": 4800, "memory_gb": 141, "tdp_w": 700,
        "max_achievable_tflops": {
            "matrix_fp16": 990, "matrix_bf16": 990, "matrix_tf32": 495,
            "matrix_fp32": 67, "matrix_fp64": 34, "matrix_fp8": 1979,
            "matrix_int8": 1979, "vector_fp16": 134, "vector_bf16": 134,
            "vector_fp32": 67, "vector_fp64": 34,
        },
    },
    "B200": {
        "name": "B200", "arch": "sm_100", "mem_bw_gbps": 8000, "memory_gb": 192, "tdp_w": 1000,
        "max_achievable_tflops": {
            "matrix_fp16": 2250, "matrix_bf16": 2250, "matrix_tf32": 1200,
            "matrix_fp32": 80, "matrix_fp64": 40, "matrix_fp8": 4500,
            "matrix_fp4": 9000, "matrix_int8": 4500, "vector_fp16": 160,
            "vector_bf16": 160, "vector_fp32": 80, "vector_fp64": 40,
        },
    },
}


def detect_gpu():
    gpu_name = None

    # Try AMD detection first
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            lower = line.lower()
            for key in sorted(PLATFORM_SPECS.keys(), key=len, reverse=True):
                if key.lower() in lower:
                    gpu_name = key
                    break
            if gpu_name:
                break
    except Exception:
        pass

    if not gpu_name:
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "gfx" in line.lower():
                    if "gfx942" in line.lower():
                        gpu_name = "MI300X"
                    elif "gfx950" in line.lower():
                        gpu_name = "MI355X"
                    break
        except Exception:
            pass

    # Try NVIDIA detection
    if not gpu_name:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            name = result.stdout.strip().split("\n")[0].upper()
            if "B200" in name:
                gpu_name = "B200"
            elif "H200" in name:
                gpu_name = "H200"
            elif "H100" in name:
                gpu_name = "H100"
        except Exception:
            pass

    return gpu_name


def main():
    parser = argparse.ArgumentParser(description="Detect GPU and write arch specs JSON")
    parser.add_argument("--output", required=True, help="Output path for gpu_arch.json")
    args = parser.parse_args()

    gpu_name = detect_gpu()

    if gpu_name and gpu_name in PLATFORM_SPECS:
        spec = PLATFORM_SPECS[gpu_name]
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"GPU_ARCH_DETECTED={gpu_name}")
        print(f"GPU_ARCH_JSON={args.output}")
    else:
        print("GPU_ARCH_DETECTED=unknown")
        print(
            "WARNING: Could not detect GPU model for roofline analysis. "
            "Roofline data will be omitted."
        )


if __name__ == "__main__":
    main()
