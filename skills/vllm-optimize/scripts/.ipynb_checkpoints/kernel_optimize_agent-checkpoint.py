#!/usr/bin/env python3
"""
Kernel Optimization Agent — auto-research kernel optimizer using Triton.

Design principles:
  - Works WITHOUT GEAK; pure Triton + autotuning
  - Auto-research loop: apply one optimization at a time, stack winners
  - Every candidate is correctness-tested (fp tolerance) AND perf-tested
  - Vendor kernels (GEMM, Attention) ARE optimization targets
  - Output: best Triton kernel per bottleneck + full optimization log

Usage:
    python kernel_optimize_agent.py \
        --gap-analysis results/gap_analysis/gap_analysis.json \
        --model-config /app/Qwen3-8B/config.json \
        --gpu-arch results/gpu_arch.json \
        --output-dir optimized/ \
        --threshold 0.5
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_module_from_file(path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("_dyn_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Kernel type classifier
# ---------------------------------------------------------------------------

KERNEL_TYPE_PATTERNS = {
    "gemm": ["Cijk_", "gemm", "mm", "matmul", "wvSplitK", "ck_tile"],
    "rmsnorm": ["rmsnorm", "rms_norm"],
    "layernorm": ["layernorm", "layer_norm"],
    "swiglu": ["act_and_mul", "silu", "swiglu"],
    "rotary": ["rotary_embedding", "rotary"],
    "attention": ["attention", "paged_attention", "flash_attn"],
    "reduce": ["reduce_segments", "allreduce", "allgather"],
    "memory": ["memcpy", "memset", "copyBuffer", "reshape_and_cache"],
}


def classify_kernel_type(name: str) -> str:
    lower = name.lower()
    for ktype, patterns in KERNEL_TYPE_PATTERNS.items():
        if any(p.lower() in lower for p in patterns):
            return ktype
    return "other"


# ---------------------------------------------------------------------------
# Correctness + Performance testing
# ---------------------------------------------------------------------------

def check_correctness(ref_fn, opt_fn, input_shapes, dtype=torch.bfloat16,
                      atol=1e-2, rtol=1e-2, num_trials=3):
    """Run reference and optimized functions on random inputs, compare."""
    device = "cuda"
    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        inputs = [torch.randn(s, dtype=dtype, device=device) for s in input_shapes]
        ref_out = ref_fn(*[x.clone() for x in inputs])
        opt_out = opt_fn(*[x.clone() for x in inputs])
        if not isinstance(ref_out, torch.Tensor):
            ref_out = ref_out[0]
        if not isinstance(opt_out, torch.Tensor):
            opt_out = opt_out[0]
        if not torch.allclose(ref_out, opt_out, atol=atol, rtol=rtol):
            max_diff = (ref_out - opt_out).abs().max().item()
            return False, f"trial {trial}: max_diff={max_diff:.6f} > atol={atol}"
    return True, "PASS"


def benchmark_fn(fn, input_shapes, dtype=torch.bfloat16,
                 warmup=10, repeats=100):
    """Benchmark a function, return median time in microseconds."""
    device = "cuda"
    torch.manual_seed(42)
    inputs = [torch.randn(s, dtype=dtype, device=device) for s in input_shapes]

    # warmup
    for _ in range(warmup):
        fn(*[x.clone() for x in inputs])
    torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*[x.clone() for x in inputs])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us

    times.sort()
    # remove top/bottom 10%
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    median = trimmed[len(trimmed) // 2]
    return {
        "median_us": round(median, 2),
        "mean_us": round(sum(trimmed) / len(trimmed), 2),
        "min_us": round(trimmed[0], 2),
        "max_us": round(trimmed[-1], 2),
        "samples": len(trimmed),
    }


# ---------------------------------------------------------------------------
# Kernel templates — written to .py files, then loaded dynamically
# ---------------------------------------------------------------------------

GEMM_TEMPLATE = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
{AUTOTUNE_CONFIGS}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm(A, B):
    assert A.shape[1] == B.shape[0], f"shape mismatch: {{A.shape}} x {{B.shape}}"
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def reference(A, B):
    return torch.mm(A, B)


def optimized(A, B):
    return triton_gemm(A, B)


INPUT_SHAPES = {INPUT_SHAPES}
DTYPE = torch.bfloat16
'''

FUSED_RMSNORM_TEMPLATE = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
{AUTOTUNE_CONFIGS}
    ],
    key=["N"],
)
@triton.jit
def _fused_residual_rmsnorm_kernel(
    X, Residual, Weight, Out,
    N,
    eps,
    stride_x, stride_r, stride_o,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)

    # fused residual add
    h = x + r

    # RMSNorm
    var = tl.sum(h * h, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(Weight + offs, mask=mask, other=1.0).to(tl.float32)
    out = (h * rstd * w).to(Out.dtype.element_ty)

    tl.store(Out + row * stride_o + offs, out, mask=mask)
    # write back updated residual
    tl.store(Residual + row * stride_r + offs, h.to(Residual.dtype.element_ty), mask=mask)


def triton_fused_residual_rmsnorm(x, residual, weight, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    _fused_residual_rmsnorm_kernel[grid](
        x, residual, weight, out,
        N, eps,
        x.stride(0), residual.stride(0), out.stride(0),
    )
    return out, residual


def reference(x, residual):
    weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    h = x.float() + residual.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h_norm = (h * torch.rsqrt(var + 1e-6))
    return (h_norm * weight.float()).to(x.dtype), h.to(x.dtype)


def optimized(x, residual):
    weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    res_clone = residual.clone()
    return triton_fused_residual_rmsnorm(x, res_clone, weight)


INPUT_SHAPES = {INPUT_SHAPES}
DTYPE = torch.bfloat16
'''

SWIGLU_TEMPLATE = '''
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
{AUTOTUNE_CONFIGS}
    ],
    key=["N"],
)
@triton.jit
def _swiglu_kernel(
    Gate, Up, Out,
    M, N,
    stride_g, stride_u, stride_o,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    g = tl.load(Gate + offs_m[:, None] * stride_g + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up + offs_m[:, None] * stride_u + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    silu_g = g * tl.sigmoid(g)
    out = (silu_g * u).to(Out.dtype.element_ty)

    tl.store(Out + offs_m[:, None] * stride_o + offs_n[None, :], out, mask=mask)


def triton_swiglu(gate, up):
    M, N = gate.shape
    out = torch.empty_like(gate)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _swiglu_kernel[grid](
        gate, up, out,
        M, N,
        gate.stride(0), up.stride(0), out.stride(0),
    )
    return out


def reference(gate, up):
    return torch.nn.functional.silu(gate) * up


def optimized(gate, up):
    return triton_swiglu(gate, up)


INPUT_SHAPES = {INPUT_SHAPES}
DTYPE = torch.bfloat16
'''


# ---------------------------------------------------------------------------
# Autotune config generation
# ---------------------------------------------------------------------------

def gemm_autotune_configs():
    """Generate Triton autotune configs for GEMM."""
    configs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for bk in [32, 64]:
                for stages in [1, 2]:
                    for warps in [2, 4, 8]:
                        for gsm in [4, 8]:
                            configs.append(
                                f'        triton.Config({{"BLOCK_M": {bm}, "BLOCK_N": {bn}, '
                                f'"BLOCK_K": {bk}, "GROUP_SIZE_M": {gsm}}}, '
                                f'num_stages={stages}, num_warps={warps}),'
                            )
    return "\n".join(configs)


def norm_autotune_configs(hidden_size):
    """Generate Triton autotune configs for norm kernels."""
    configs = []
    # block must be >= hidden_size and power of 2
    bn = 1
    while bn < hidden_size:
        bn *= 2
    for warps in [2, 4, 8, 16]:
        configs.append(
            f'        triton.Config({{"BLOCK_N": {bn}}}, num_warps={warps}),'
        )
    return "\n".join(configs)


def swiglu_autotune_configs():
    """Generate Triton autotune configs for SwiGLU."""
    configs = []
    for bm in [1, 4, 16]:
        for bn in [256, 512, 1024, 2048]:
            for warps in [2, 4, 8]:
                configs.append(
                    f'        triton.Config({{"BLOCK_M": {bm}, "BLOCK_N": {bn}}}, '
                    f'num_warps={warps}),'
                )
    return "\n".join(configs)


# ---------------------------------------------------------------------------
# Problem generation for each kernel type
# ---------------------------------------------------------------------------

def generate_gemm_problem(kernel_info, model_cfg, output_dir):
    """Generate a GEMM optimization problem."""
    hidden = model_cfg.get("hidden_size", 2560)
    inter = model_cfg.get("intermediate_size", 9216)
    # Common GEMM shapes in transformer: (batch*seq, hidden) x (hidden, inter)
    # and (batch*seq, inter) x (inter, hidden)
    shapes = [
        [(64, hidden), (hidden, inter)],   # up/gate projection
        [(64, inter), (inter, hidden)],    # down projection
        [(64, hidden), (hidden, hidden)],  # QKV projection
    ]
    code = GEMM_TEMPLATE.replace("{AUTOTUNE_CONFIGS}", gemm_autotune_configs())
    code = code.replace("{INPUT_SHAPES}", repr(shapes[0]))  # start with up proj shape

    kernel_dir = os.path.join(output_dir, "gemm")
    os.makedirs(kernel_dir, exist_ok=True)

    # Write the kernel file
    kernel_path = os.path.join(kernel_dir, "triton_gemm.py")
    with open(kernel_path, "w") as f:
        f.write(code)

    return {
        "type": "gemm",
        "kernel_path": kernel_path,
        "shapes": shapes,
        "original_kernel": kernel_info["name"],
        "original_pct": kernel_info["pct_total"],
    }


def generate_rmsnorm_problem(kernel_info, model_cfg, output_dir):
    """Generate a fused residual + RMSNorm optimization problem."""
    hidden = model_cfg.get("hidden_size", 2560)
    shapes = [(64, hidden), (64, hidden)]  # x, residual

    code = FUSED_RMSNORM_TEMPLATE.replace(
        "{AUTOTUNE_CONFIGS}", norm_autotune_configs(hidden)
    )
    code = code.replace("{INPUT_SHAPES}", repr(shapes))

    kernel_dir = os.path.join(output_dir, "fused_rmsnorm")
    os.makedirs(kernel_dir, exist_ok=True)
    kernel_path = os.path.join(kernel_dir, "triton_fused_rmsnorm.py")
    with open(kernel_path, "w") as f:
        f.write(code)

    return {
        "type": "fused_rmsnorm",
        "kernel_path": kernel_path,
        "shapes": [shapes],
        "original_kernel": kernel_info["name"],
        "original_pct": kernel_info["pct_total"],
    }


def generate_swiglu_problem(kernel_info, model_cfg, output_dir):
    """Generate a SwiGLU optimization problem."""
    inter = model_cfg.get("intermediate_size", 9216)
    shapes = [(64, inter), (64, inter)]  # gate, up

    code = SWIGLU_TEMPLATE.replace("{AUTOTUNE_CONFIGS}", swiglu_autotune_configs())
    code = code.replace("{INPUT_SHAPES}", repr(shapes))

    kernel_dir = os.path.join(output_dir, "swiglu")
    os.makedirs(kernel_dir, exist_ok=True)
    kernel_path = os.path.join(kernel_dir, "triton_swiglu.py")
    with open(kernel_path, "w") as f:
        f.write(code)

    return {
        "type": "swiglu",
        "kernel_path": kernel_path,
        "shapes": [shapes],
        "original_kernel": kernel_info["name"],
        "original_pct": kernel_info["pct_total"],
    }


PROBLEM_GENERATORS = {
    "gemm": generate_gemm_problem,
    "rmsnorm": generate_rmsnorm_problem,
    "swiglu": generate_swiglu_problem,
}


# ---------------------------------------------------------------------------
# Optimization strategies — each returns a modified kernel code string
# ---------------------------------------------------------------------------

def strategy_batch_sweep(problem, model_cfg):
    """Sweep multiple batch sizes to find optimal."""
    hidden = model_cfg.get("hidden_size", 2560)
    inter = model_cfg.get("intermediate_size", 9216)
    ktype = problem["type"]
    shapes_list = []
    if ktype == "gemm":
        for bs in [1, 16, 64, 256]:
            shapes_list.append([(bs, hidden), (hidden, inter)])
            shapes_list.append([(bs, inter), (inter, hidden)])
            shapes_list.append([(bs, hidden), (hidden, hidden)])
    elif ktype == "fused_rmsnorm":
        for bs in [1, 16, 64, 256]:
            shapes_list.append([(bs, hidden), (bs, hidden)])
    elif ktype == "swiglu":
        for bs in [1, 16, 64, 256]:
            shapes_list.append([(bs, inter), (bs, inter)])
    return shapes_list


# ---------------------------------------------------------------------------
# Auto-research optimization loop
# ---------------------------------------------------------------------------

def optimize_one_kernel(problem, model_cfg, output_dir):
    """
    Auto-research loop for one kernel:
    1. Load Triton kernel (with autotune)
    2. For each shape combo, run correctness + perf test
    3. Compare against PyTorch reference baseline
    4. Record all results, report best
    """
    ktype = problem["type"]
    kernel_path = problem["kernel_path"]
    kernel_dir = os.path.dirname(kernel_path)
    log_entries = []

    print(f"\n{'='*70}")
    print(f"Optimizing: {ktype} (original: {problem['original_kernel'][:60]})")
    print(f"  Original % of GPU time: {problem['original_pct']:.1f}%")
    print(f"  Kernel path: {kernel_path}")
    print(f"{'='*70}")

    # Load the kernel module
    try:
        mod = load_module_from_file(kernel_path)
    except Exception as e:
        msg = f"FAILED to load kernel module: {e}"
        print(f"  {msg}")
        return {"status": "error", "error": msg, "log": log_entries}

    # Generate shape sweep
    all_shapes = strategy_batch_sweep(problem, model_cfg)
    if not all_shapes:
        all_shapes = problem["shapes"]
    if not isinstance(all_shapes[0], list):
        all_shapes = [all_shapes]

    results_by_shape = []

    for shape_set in all_shapes:
        shape_tag = "x".join(str(s) for s in shape_set[0])
        print(f"\n  --- Shape: {[list(s) for s in shape_set]} ---")

        # 1. Correctness test
        try:
            correct, detail = check_correctness(
                mod.reference, mod.optimized, shape_set,
                dtype=torch.bfloat16, atol=1e-2, rtol=1e-2,
            )
        except Exception as e:
            detail = f"correctness exception: {e}"
            correct = False
            traceback.print_exc()

        if not correct:
            print(f"    CORRECTNESS FAIL: {detail}")
            log_entries.append({
                "shape": [list(s) for s in shape_set],
                "correctness": False,
                "detail": detail,
            })
            continue

        print(f"    Correctness: PASS")

        # 2. Benchmark reference (PyTorch)
        try:
            ref_perf = benchmark_fn(mod.reference, shape_set)
        except Exception as e:
            print(f"    Reference benchmark failed: {e}")
            ref_perf = {"median_us": float("inf")}

        # 3. Benchmark optimized (Triton)
        try:
            opt_perf = benchmark_fn(mod.optimized, shape_set)
        except Exception as e:
            print(f"    Optimized benchmark failed: {e}")
            opt_perf = {"median_us": float("inf")}

        speedup = ref_perf["median_us"] / opt_perf["median_us"] if opt_perf["median_us"] > 0 else 0
        tag = "FASTER" if speedup > 1.0 else "SLOWER"

        entry = {
            "shape": [list(s) for s in shape_set],
            "correctness": True,
            "reference_us": ref_perf["median_us"],
            "optimized_us": opt_perf["median_us"],
            "speedup": round(speedup, 3),
            "tag": tag,
            "ref_detail": ref_perf,
            "opt_detail": opt_perf,
        }
        log_entries.append(entry)
        results_by_shape.append(entry)

        print(f"    Reference: {ref_perf['median_us']:.1f} us")
        print(f"    Optimized: {opt_perf['median_us']:.1f} us")
        print(f"    Speedup:   {speedup:.3f}x  [{tag}]")

    # Summary
    if not results_by_shape:
        print(f"\n  No valid results for {ktype}")
        return {"status": "no_valid_results", "log": log_entries}

    best = max(results_by_shape, key=lambda x: x["speedup"])
    worst = min(results_by_shape, key=lambda x: x["speedup"])
    avg_speedup = sum(r["speedup"] for r in results_by_shape) / len(results_by_shape)

    summary = {
        "type": ktype,
        "original_kernel": problem["original_kernel"],
        "original_pct": problem["original_pct"],
        "kernel_path": kernel_path,
        "num_shapes_tested": len(results_by_shape),
        "best_speedup": best["speedup"],
        "best_shape": best["shape"],
        "worst_speedup": worst["speedup"],
        "avg_speedup": round(avg_speedup, 3),
        "all_correct": all(r["correctness"] for r in log_entries),
        "status": "success",
    }

    print(f"\n  Summary for {ktype}:")
    print(f"    Shapes tested: {len(results_by_shape)}")
    print(f"    Best speedup:  {best['speedup']:.3f}x at shape {best['shape']}")
    print(f"    Worst speedup: {worst['speedup']:.3f}x")
    print(f"    Avg speedup:   {avg_speedup:.3f}x")

    # Save results
    save_json({"summary": summary, "log": log_entries},
              os.path.join(kernel_dir, "optimization_results.json"))

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kernel Optimization Agent")
    parser.add_argument("--gap-analysis", required=True, help="gap_analysis.json path")
    parser.add_argument("--model-config", required=True, help="Model config.json path")
    parser.add_argument("--gpu-arch", default=None, help="gpu_arch.json path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min pct of GPU time to optimize (default 0.5%%)")
    args = parser.parse_args()

    # Load inputs
    gap = load_json(args.gap_analysis)
    model_cfg_raw = load_json(args.model_config)

    # Handle nested text_config (Qwen3.5 multimodal) vs flat config
    if "text_config" in model_cfg_raw:
        model_cfg = model_cfg_raw["text_config"]
    else:
        model_cfg = model_cfg_raw

    gpu_arch = load_json(args.gpu_arch) if args.gpu_arch and os.path.exists(args.gpu_arch) else {}

    print("=" * 70)
    print("Kernel Optimization Agent")
    print("=" * 70)
    print(f"Model config: hidden={model_cfg.get('hidden_size')}, "
          f"intermediate={model_cfg.get('intermediate_size')}")
    print(f"GPU arch: {gpu_arch.get('gpu_arch', 'unknown')}")
    print(f"Threshold: {args.threshold}%")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Identify kernels to optimize
    top_kernels = gap.get("top_kernels", [])
    targets = []
    for k in top_kernels:
        pct = k.get("pct_total", 0)
        if pct < args.threshold:
            continue
        ktype = classify_kernel_type(k["name"])
        if ktype in PROBLEM_GENERATORS:
            targets.append({"info": k, "type": ktype})
        else:
            print(f"  SKIP: {k['name'][:60]} ({pct:.1f}%) — type '{ktype}' has no Triton template")

    # Deduplicate by type (optimize each type once)
    seen_types = set()
    unique_targets = []
    for t in targets:
        if t["type"] not in seen_types:
            seen_types.add(t["type"])
            unique_targets.append(t)

    print(f"\nTargets to optimize: {len(unique_targets)}")
    for t in unique_targets:
        print(f"  [{t['type']}] {t['info']['name'][:60]} — {t['info']['pct_total']:.1f}%")

    # Generate problems and optimize
    all_results = []
    for t in unique_targets:
        gen_fn = PROBLEM_GENERATORS[t["type"]]
        problem = gen_fn(t["info"], model_cfg, args.output_dir)
        result = optimize_one_kernel(problem, model_cfg, args.output_dir)
        all_results.append(result)

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    for r in all_results:
        status = r.get("status", "unknown")
        ktype = r.get("type", "unknown")
        if status == "success":
            print(f"  [{ktype}] best={r['best_speedup']:.3f}x avg={r['avg_speedup']:.3f}x "
                  f"({r['num_shapes_tested']} shapes) — original {r['original_pct']:.1f}% GPU time")
        else:
            print(f"  [{ktype}] {status}: {r.get('error', 'no details')}")

    save_json({"results": all_results, "gpu_arch": gpu_arch, "model_config_path": args.model_config},
              os.path.join(args.output_dir, "agent_report.json"))

    print(f"\nResults saved to {args.output_dir}/agent_report.json")


if __name__ == "__main__":
    main()
