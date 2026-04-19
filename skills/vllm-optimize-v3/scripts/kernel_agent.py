#!/usr/bin/env python3
"""
Kernel Optimization Agent — scaffold and toolbox.

Provides tool commands that the AI agent calls during Phase 4:

  setup         — create workspaces, measure baselines, collect HW counters
  status        — show current optimization state for a target
  benchmark     — measure optimized vs reference on real shapes
  correctness   — verify torch.allclose(atol=1e-2, rtol=1e-2) on real shapes
  rocprof       — collect HW counters via rocprofv3
  accept        — promote kernel to current best, update log
  reject        — log rejection with reason
  serving-test  — simulate dynamic-M serving to detect autotune regression

The AI agent reads status, writes attempt_N.py, calls correctness+benchmark,
then accept or reject. This script handles all measurement; the AI handles strategy.

Usage (setup):
    python kernel_agent.py setup \\
        --targets    problems/targets.json \\
        --real-shapes results/real_shapes.json \\
        --model-config /app/Qwen3-8B/config.json \\
        --gpu-arch   results/gpu_arch.json \\
        --output-dir optimized/ \\
        --knowledge-base references/TRITON_KNOWLEDGE.md

Usage (tools):
    python kernel_agent.py benchmark --kernel optimized/gemm/attempt_1.py --shapes '[[...]]'
    python kernel_agent.py correctness --kernel ... --shapes '[[...]]'
    python kernel_agent.py serving-test --kernel ... --n 4096 --k 4096 --m-values 1,2,4,8,16,32,64
"""

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sqlite3
import sys
import time

import torch


# ─── Helpers ───────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_module(path):
    spec = importlib.util.spec_from_file_location("_kmod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def resolve_dtype(kernel_path):
    """Read dtype hint from first comment lines of kernel file."""
    try:
        with open(kernel_path) as f:
            for line in f:
                line = line.strip()
                if "dtype:" in line.lower():
                    if "float16" in line: return torch.float16
                    if "float32" in line: return torch.float32
                    if "bfloat16" in line: return torch.bfloat16
                if line and not line.startswith("#"):
                    break
    except Exception:
        pass
    return torch.bfloat16

def detect_dtype_from_config(cfg_path):
    """Read torch_dtype from model config.json."""
    try:
        cfg = load_json(cfg_path)
        cfg = cfg.get("text_config", cfg)
        v = cfg.get("torch_dtype") or cfg.get("dtype", "bfloat16")
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(v, torch.bfloat16)
    except Exception:
        return torch.bfloat16

def count_trailing_rejections(steps):
    count = 0
    for s in reversed(steps):
        if s.get("action") == "REJECTED":
            count += 1
        else:
            break
    return count

def load_log(log_path):
    if os.path.exists(log_path):
        return load_json(log_path)
    return {"baseline": {}, "optimization_path": [], "steps": [],
            "best_kernel": None, "best_avg_speedup": 0.0,
            "serving_ready": False, "shape_coverage_pct": None}


# ─── Benchmark ─────────────────────────────────────────────────────────────

def do_benchmark(kernel_path, shapes_list, warmup=10, reps=50):
    """
    Benchmark reference() vs optimized() on real shapes.
    Returns {shape_tag: {reference: {...}, optimized: {...}, speedup: float}}.
    """
    mod   = load_module(kernel_path)
    dtype = resolve_dtype(kernel_path)
    results = {}

    for shapes in shapes_list:
        tag = "_x_".join("x".join(str(d) for d in s) for s in shapes)
        torch.cuda.empty_cache()
        inputs = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]
        perf = {}

        for fn_name in ("reference", "optimized"):
            fn = getattr(mod, fn_name)
            # Warmup
            for _ in range(warmup):
                fn(*[x.clone() for x in inputs])
            torch.cuda.synchronize()
            # Measure
            times = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn(*[x.clone() for x in inputs])
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
            times.sort()
            trim = max(1, len(times) // 10)
            t = times[trim:-trim] if len(times) > 2 * trim else times
            perf[fn_name] = {
                "median_us": round(t[len(t) // 2], 2),
                "mean_us":   round(sum(t) / len(t), 2),
                "min_us":    round(t[0], 2),
                "max_us":    round(t[-1], 2),
            }

        speedup = (perf["reference"]["median_us"] / perf["optimized"]["median_us"]
                   if perf["optimized"]["median_us"] > 0 else 0.0)
        results[tag] = {**perf, "speedup": round(speedup, 4)}

    return results


# ─── Correctness ───────────────────────────────────────────────────────────

def do_correctness(kernel_path, shapes_list, atol=1e-2, rtol=1e-2, trials=3):
    """
    Verify torch.allclose(atol=1e-2, rtol=1e-2) on all shapes.
    Returns {shape_tag: {correct: bool, detail: str}}.
    """
    mod   = load_module(kernel_path)
    dtype = resolve_dtype(kernel_path)
    results = {}

    for shapes in shapes_list:
        tag = "_x_".join("x".join(str(d) for d in s) for s in shapes)
        ok = True; detail = "PASS"
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            inputs = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]
            ref = mod.reference(*[x.clone() for x in inputs])
            opt = mod.optimized(*[x.clone() for x in inputs])
            if not isinstance(ref, torch.Tensor): ref = ref[0]
            if not isinstance(opt, torch.Tensor): opt = opt[0]
            if not torch.allclose(ref.float(), opt.float(), atol=atol, rtol=rtol):
                ok = False
                max_diff = float((ref.float() - opt.float()).abs().max())
                detail = f"trial {trial}: max_diff={max_diff:.6f} (atol={atol})"
                break
        results[tag] = {"correct": ok, "detail": detail}

    return results


# ─── HW Profiling (rocprofv3) ──────────────────────────────────────────────

def do_rocprof(kernel_path, shapes_list, output_dir):
    """Run rocprofv3 kernel trace. Returns HW counters or skip info."""
    rp = "/opt/rocm/bin/rocprofv3"
    if not os.path.exists(rp):
        r = subprocess.run(["which", "rocprofv3"], capture_output=True, text=True)
        if r.returncode != 0:
            return {"status": "skipped", "reason": "rocprofv3 not found"}
        rp = r.stdout.strip()

    prof_dir = os.path.join(output_dir, "rocprof", str(int(time.time())))
    os.makedirs(prof_dir, exist_ok=True)

    shapes_repr = repr([list(s) for s in shapes_list[0]])
    bench_py = os.path.join(prof_dir, "bench.py")
    dtype = resolve_dtype(kernel_path)
    dtype_str = "torch.bfloat16" if dtype == torch.bfloat16 else \
                "torch.float16" if dtype == torch.float16 else "torch.float32"

    with open(bench_py, "w") as f:
        f.write(f"""import torch, importlib.util
spec = importlib.util.spec_from_file_location('k', '{kernel_path}')
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
shapes = {shapes_repr}
inputs = [torch.randn(s, dtype={dtype_str}, device='cuda') for s in shapes]
for _ in range(5): mod.optimized(*[x.clone() for x in inputs])
torch.cuda.synchronize()
for _ in range(50): mod.optimized(*inputs)
torch.cuda.synchronize()
""")

    try:
        r = subprocess.run(
            [rp, "--kernel-trace", "-o", os.path.join(prof_dir, "trace"),
             "-d", prof_dir, "--", sys.executable, bench_py],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"status": "error", "stderr": r.stderr[:300]}

        dbs = [f for f in os.listdir(prof_dir) if f.endswith(".db")]
        if not dbs:
            return {"status": "error", "reason": "no .db output"}

        conn = sqlite3.connect(os.path.join(prof_dir, dbs[0]))
        c    = conn.cursor()
        tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]

        hw_info = []
        ks_tables = [t for t in tables if "kernel_symbol" in t]
        if ks_tables:
            c.execute(f"SELECT display_name, sgpr_count, arch_vgpr_count, accum_vgpr_count, "
                      f"group_segment_size FROM {ks_tables[0]}")
            for row in c.fetchall():
                name = row[0] or "?"
                if any(name.startswith(p) for p in ("void at::native::", "__amd_rocclr_")):
                    continue
                hw_info.append({"kernel": name[:80], "sgpr": row[1], "vgpr": row[2],
                                 "accum_vgpr": row[3], "lds_bytes": row[4]})

        dispatches = []
        kd_tables = [t for t in tables if "kernel_dispatch" in t]
        if kd_tables:
            c.execute(f"SELECT start, end, workgroup_size_x, grid_size_x, grid_size_y, grid_size_z "
                      f"FROM {kd_tables[0]} WHERE end > start ORDER BY (end-start) DESC LIMIT 10")
            for row in c.fetchall():
                dispatches.append({"duration_us": round((row[1]-row[0])/1000, 2),
                                   "workgroup_x": row[2], "grid": f"{row[3]}x{row[4]}x{row[5]}"})
        conn.close()
        return {"status": "success", "hw_info": hw_info[:8], "dispatches": dispatches[:5]}

    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}


# ─── Serving Test ──────────────────────────────────────────────────────────

def do_serving_test(kernel_path, N, K, m_values, iterations):
    """
    Simulate vLLM serving by cycling through M values rapidly.
    Detects autotune regression: if @triton.autotune benchmarks each new (M,N,K)
    at serving time, this test shows massive slowdown vs a fixed-M benchmark.

    Returns: {speedup, verdict, optimized_avg_us, reference_avg_us}
    Verdict: PASS (speedup>=0.90), FAIL_REGRESSION, FAIL_SEVERE_REGRESSION
    """
    mod   = load_module(kernel_path)
    dtype = resolve_dtype(kernel_path)
    m_cycle = itertools.cycle(m_values)

    weight = torch.randn(N, K, dtype=dtype, device="cuda")  # (out, in) like vLLM

    # Warmup both
    for _ in range(5):
        x = torch.randn(1, K, dtype=dtype, device="cuda")
        mod.optimized(x, weight.t())
        mod.reference(x, weight.t())
    torch.cuda.synchronize()

    # Measure optimized under varying M
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        m = next(m_cycle)
        x = torch.randn(m, K, dtype=dtype, device="cuda")
        mod.optimized(x, weight.t())
    torch.cuda.synchronize()
    opt_us = (time.time() - t0) * 1e6

    # Measure reference under same M sequence
    m_cycle2 = itertools.cycle(m_values)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        m = next(m_cycle2)
        x = torch.randn(m, K, dtype=dtype, device="cuda")
        mod.reference(x, weight.t())
    torch.cuda.synchronize()
    ref_us = (time.time() - t0) * 1e6

    opt_avg = opt_us / iterations
    ref_avg = ref_us / iterations
    speedup = ref_avg / opt_avg if opt_avg > 0 else 0.0

    verdict = ("PASS" if speedup >= 0.90 else
               "FAIL_SEVERE_REGRESSION" if speedup < 0.50 else
               "FAIL_REGRESSION")

    return {
        "speedup": round(speedup, 4),
        "optimized_avg_us": round(opt_avg, 2),
        "reference_avg_us": round(ref_avg, 2),
        "iterations": iterations,
        "m_values": m_values,
        "N": N, "K": K,
        "verdict": verdict,
    }


# ─── Accept / Reject / Status ──────────────────────────────────────────────

def do_accept(kernel_path, name, desc, bench_results, log_path, kernel_dir):
    import shutil
    log = load_log(log_path)
    best_path = os.path.join(kernel_dir, "best_kernel.py")
    shutil.copy2(kernel_path, best_path)

    speedups = [v["speedup"] for v in bench_results.values() if "speedup" in v]
    avg = sum(speedups) / len(speedups) if speedups else 0.0

    log["optimization_path"].append(name)
    log["best_kernel"] = best_path
    log["best_avg_speedup"] = round(avg, 4)
    log["steps"].append({
        "step": len(log["steps"]) + 1, "name": name, "description": desc,
        "action": "ACCEPTED", "avg_speedup": round(avg, 4),
        "per_shape": bench_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_json(log, log_path)
    return {"status": "accepted", "avg_speedup": round(avg, 4)}


def do_reject(name, desc, reason, bench_results, log_path):
    log = load_log(log_path)
    speedups = [v["speedup"] for v in bench_results.values() if "speedup" in v] if bench_results else []
    avg = sum(speedups) / len(speedups) if speedups else 0.0
    log["steps"].append({
        "step": len(log["steps"]) + 1, "name": name, "description": desc,
        "action": "REJECTED", "reason": reason, "avg_speedup": round(avg, 4),
        "per_shape": bench_results or {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_json(log, log_path)
    return {"status": "rejected", "reason": reason}


def do_status(log_path):
    log = load_log(log_path)
    steps = log.get("steps", [])
    accepted  = [s for s in steps if s.get("action") == "ACCEPTED"]
    rejected  = [s for s in steps if s.get("action") == "REJECTED"]
    return {
        "kernel_type":    log.get("kernel_type", "unknown"),
        "dtype":          log.get("dtype", "bfloat16"),
        "gpu_arch":       log.get("gpu_arch", "unknown"),
        "total_attempts": len(steps),
        "accepted":       len(accepted),
        "rejected":       len(rejected),
        "consecutive_rejections": count_trailing_rejections(steps),
        "max_attempts":   log.get("max_attempts", 8),
        "max_consecutive_rejections": log.get("max_consecutive_rejections", 3),
        "best_avg_speedup": log.get("best_avg_speedup", 0.0),
        "best_kernel":    log.get("best_kernel"),
        "serving_ready":  log.get("serving_ready", False),
        "shape_coverage_pct": log.get("shape_coverage_pct"),
        "baseline":       log.get("baseline", {}),
        "baseline_rocprof": log.get("baseline_rocprof", {}),
        "benchmark_shapes": log.get("benchmark_shapes", []),
        "knowledge_base": log.get("knowledge_base"),
        "optimization_path": log.get("optimization_path", []),
        "steps_summary": [
            {"step": s["step"], "name": s["name"], "action": s.get("action"),
             "avg_speedup": s.get("avg_speedup", 0), "description": s.get("description",""),
             "reason": s.get("reason", "")}
            for s in steps
        ],
    }


# ─── Setup ─────────────────────────────────────────────────────────────────

def _write_baseline_kernel(path, ktype, hidden, inter, dtype_name):
    """Write a minimal kernel that defines reference() = optimized() = PyTorch baseline."""
    header = f"# dtype: {dtype_name}\nimport torch\n"
    if ktype == "gemm":
        code = header + "def reference(A, B): return torch.mm(A, B)\ndef optimized(A, B): return torch.mm(A, B)\n"
    elif ktype in ("rmsnorm", "layernorm"):
        code = header + (
            "def reference(x, w):\n"
            "    v = x.float().pow(2).mean(-1, keepdim=True)\n"
            "    return (x.float() * torch.rsqrt(v + 1e-6) * w.float()).to(x.dtype)\n"
            "def optimized(x, w): return reference(x, w)\n")
    elif ktype in ("swiglu", "activation"):
        code = header + (
            "def reference(gate, up): return torch.nn.functional.silu(gate) * up\n"
            "def optimized(gate, up): return reference(gate, up)\n")
    elif ktype == "rotary":
        code = header + (
            "def reference(x, cos, sin): return x * cos + torch.roll(x, 1, -1) * sin\n"
            "def optimized(x, cos, sin): return reference(x, cos, sin)\n")
    else:
        code = header + "def reference(*a): return a[0]\ndef optimized(*a): return reference(*a)\n"
    with open(path, "w") as f:
        f.write(code)


def cmd_setup(args):
    targets_data  = load_json(args.targets)
    model_cfg_raw = load_json(args.model_config) if args.model_config and os.path.exists(args.model_config) else {}
    model_cfg     = model_cfg_raw.get("text_config", model_cfg_raw)
    gpu_arch_data = load_json(args.gpu_arch) if args.gpu_arch and os.path.exists(args.gpu_arch) else {}

    hidden = model_cfg.get("hidden_size", 4096)
    inter  = model_cfg.get("intermediate_size", hidden * 3)

    # dtype: prefer model config over default
    dtype_obj = detect_dtype_from_config(args.model_config) if args.model_config and os.path.exists(args.model_config) else torch.bfloat16
    dtype_name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype_obj]
    dtype_torch = {torch.bfloat16: "torch.bfloat16", torch.float16: "torch.float16", torch.float32: "torch.float32"}[dtype_obj]

    # Real shapes
    real_shapes_data = {}
    if args.real_shapes and os.path.exists(args.real_shapes):
        real_shapes_data = load_json(args.real_shapes)

    # Knowledge base path
    kb_path = args.knowledge_base
    if not kb_path or not os.path.exists(kb_path):
        # Try relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "references", "TRITON_KNOWLEDGE.md"),
            os.path.join(script_dir, "TRITON_KNOWLEDGE.md"),
        ]
        kb_path = next((p for p in candidates if os.path.exists(p)), None)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print(" Kernel Agent — Setup")
    print("=" * 64)
    print(f" dtype={dtype_name}  gpu_arch={gpu_arch_data.get('gpu_arch','?')}")
    print(f" hidden={hidden}  intermediate={inter}")
    if kb_path:
        print(f" Knowledge base: {kb_path}")

    all_targets = []
    for t in targets_data.get("targets", []):
        ktype = t["kernel_type"]
        if not t.get("has_real_shapes"):
            print(f"\n[{ktype}] SKIP — no real shapes from trace")
            continue

        kernel_dir = os.path.join(args.output_dir, ktype)
        os.makedirs(kernel_dir, exist_ok=True)
        log_path = os.path.join(kernel_dir, "optimization_log.json")

        # Use real shapes for GEMM; fall back to model-config shapes for others
        bench_shapes = real_shapes_data.get("benchmark_shapes", [])[:20] if ktype == "gemm" else []
        if not bench_shapes:
            # Generic fallback: create shapes from model config
            if ktype in ("rmsnorm", "layernorm"):
                bench_shapes = [[[bs, hidden], [bs, hidden]] for bs in [16, 64, 256]]
            elif ktype in ("swiglu", "activation"):
                bench_shapes = [[[bs, inter // 2], [bs, inter // 2]] for bs in [1, 16, 64]]
            elif ktype == "gemm":
                bench_shapes = [[[bs, hidden], [hidden, inter]] for bs in [1, 64, 256]]
            else:
                bench_shapes = [[[bs, hidden], [bs, hidden]] for bs in [16, 64]]

        print(f"\n[{ktype}] Setting up workspace...")
        print(f"  Shapes: {len(bench_shapes)} (M values: {sorted(set(s[0][0] for s in bench_shapes))})")

        # Write baseline kernel
        baseline_path = os.path.join(kernel_dir, "baseline.py")
        _write_baseline_kernel(baseline_path, ktype, hidden, inter, dtype_name)

        # Measure baseline
        print(f"  Measuring PyTorch baseline...")
        baseline_perf = {}
        try:
            baseline_perf = do_benchmark(baseline_path, bench_shapes, warmup=5, reps=30)
            for tag, p in baseline_perf.items():
                print(f"    {tag}: {p['reference']['median_us']:.1f}us")
        except Exception as e:
            print(f"    WARN: baseline measurement failed: {e}")

        # HW profiling
        rp = do_rocprof(baseline_path, bench_shapes[:2], kernel_dir)
        if rp.get("status") == "success":
            for hi in rp.get("hw_info", [])[:3]:
                print(f"    HW: VGPR={hi.get('vgpr','?')} SGPR={hi.get('sgpr','?')} "
                      f"LDS={hi.get('lds_bytes','?')} kernel={hi.get('kernel','?')[:40]}")

        log = {
            "kernel_type": ktype,
            "original_kernel": t.get("kernel_name", ktype),
            "original_pct": t.get("pct_total", 0),
            "model_config": {"hidden_size": hidden, "intermediate_size": inter},
            "dtype": dtype_name, "dtype_torch": dtype_torch,
            "gpu_arch": gpu_arch_data.get("gpu_arch", "unknown"),
            "benchmark_shapes": bench_shapes,
            "real_m_values": sorted(set(s[0][0] for s in bench_shapes)),
            "baseline": baseline_perf,
            "baseline_rocprof": rp,
            "knowledge_base": kb_path,
            "max_attempts": args.max_attempts,
            "max_consecutive_rejections": args.max_rejections,
            "optimization_path": [], "steps": [],
            "best_kernel": None, "best_avg_speedup": 0.0,
            "serving_ready": False, "shape_coverage_pct": None,
        }
        save_json(log, log_path)
        all_targets.append({"type": ktype, "log_path": log_path,
                             "kernel_dir": kernel_dir, "bench_shapes": bench_shapes})
        print(f"  Workspace: {kernel_dir}")

    manifest = {
        "targets": all_targets,
        "dtype": dtype_name, "dtype_torch": dtype_torch,
        "gpu_arch": gpu_arch_data,
        "knowledge_base": kb_path,
        "max_attempts": args.max_attempts,
        "max_consecutive_rejections": args.max_rejections,
    }
    save_json(manifest, os.path.join(args.output_dir, "manifest.json"))
    print(f"\nSetup complete. {len(all_targets)} target(s) ready.")
    print(f"Manifest: {args.output_dir}/manifest.json")


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kernel Optimization Agent")
    sub = parser.add_subparsers(dest="cmd")

    # setup
    p = sub.add_parser("setup")
    p.add_argument("--targets",       required=True)
    p.add_argument("--real-shapes",   default=None)
    p.add_argument("--model-config",  default=None)
    p.add_argument("--gpu-arch",      default=None)
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--knowledge-base",default=None)
    p.add_argument("--max-attempts",  type=int, default=8)
    p.add_argument("--max-rejections",type=int, default=3)

    # benchmark
    p = sub.add_parser("benchmark")
    p.add_argument("--kernel",  required=True)
    p.add_argument("--shapes",  required=True)

    # correctness
    p = sub.add_parser("correctness")
    p.add_argument("--kernel",  required=True)
    p.add_argument("--shapes",  required=True)

    # rocprof
    p = sub.add_parser("rocprof")
    p.add_argument("--kernel",     required=True)
    p.add_argument("--shapes",     required=True)
    p.add_argument("--output-dir", default=".")

    # accept
    p = sub.add_parser("accept")
    p.add_argument("--kernel",     required=True)
    p.add_argument("--name",       required=True)
    p.add_argument("--desc",       default="")
    p.add_argument("--results",    required=True)
    p.add_argument("--log-path",   required=True)
    p.add_argument("--kernel-dir", required=True)

    # reject
    p = sub.add_parser("reject")
    p.add_argument("--name",     required=True)
    p.add_argument("--desc",     default="")
    p.add_argument("--reason",   required=True)
    p.add_argument("--results",  default="{}")
    p.add_argument("--log-path", required=True)

    # status
    p = sub.add_parser("status")
    p.add_argument("--log-path", required=True)

    # serving-test
    p = sub.add_parser("serving-test")
    p.add_argument("--kernel",    required=True)
    p.add_argument("--n",         type=int, default=4096)
    p.add_argument("--k",         type=int, default=4096)
    p.add_argument("--m-values",  default="1,2,4,8,16,32,64,128")
    p.add_argument("--iterations",type=int, default=200)

    args = parser.parse_args()

    if args.cmd == "setup":
        cmd_setup(args)

    elif args.cmd == "benchmark":
        shapes = json.loads(args.shapes)
        print(json.dumps(do_benchmark(args.kernel, shapes), indent=2))

    elif args.cmd == "correctness":
        shapes = json.loads(args.shapes)
        r = do_correctness(args.kernel, shapes)
        print(json.dumps(r, indent=2))
        all_ok = all(v["correct"] for v in r.values())
        if not all_ok:
            print("\nCORRECTNESS FAILED — do not proceed to benchmark", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "rocprof":
        shapes = json.loads(args.shapes)
        print(json.dumps(do_rocprof(args.kernel, shapes, args.output_dir), indent=2))

    elif args.cmd == "accept":
        r = do_accept(args.kernel, args.name, args.desc,
                      json.loads(args.results), args.log_path, args.kernel_dir)
        print(json.dumps(r, indent=2))

    elif args.cmd == "reject":
        r = do_reject(args.name, args.desc, args.reason,
                      json.loads(args.results), args.log_path)
        print(json.dumps(r, indent=2))

    elif args.cmd == "status":
        print(json.dumps(do_status(args.log_path), indent=2))

    elif args.cmd == "serving-test":
        m_values = [int(x) for x in args.m_values.split(",")]
        r = do_serving_test(args.kernel, args.n, args.k, m_values, args.iterations)
        print(json.dumps(r, indent=2))
        print(f"\nServing test verdict: {r['verdict']}")
        if r["verdict"] != "PASS":
            print("ACTION REQUIRED: fix autotune regression before integrating", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
