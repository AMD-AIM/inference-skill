#!/usr/bin/env python3
"""
Kernel Optimization Agent — scaffold & toolbox.

This script does NOT contain optimization strategies. It provides:
  1. Workspace setup (dirs, baseline measurement)
  2. Tool functions that a sub-agent calls to evaluate kernels:
     - benchmark(kernel_path, shapes)  → median_us per shape
     - correctness(kernel_path, shapes) → pass/fail per shape
     - rocprof(kernel_path, shapes) → HW counters (VGPR, SGPR, LDS, dispatch duration)
     - accept(kernel_path) → promote to current best, update log
     - reject(reason) → log rejection, keep current best
  3. Optimization log maintenance (JSON)

The actual optimization decisions are made by a sub-agent that:
  - Reads the current state (baseline perf, HW counters, optimization history)
  - Writes a new Triton kernel file
  - Calls the tool functions to evaluate it
  - Decides whether to accept or reject
  - Repeats until convergence

Usage (setup mode — run once to create workspace and baseline):
    python kernel_optimize_agent.py setup \\
        --gap-analysis results/gap_analysis/gap_analysis.json \\
        --model-config /app/Qwen3-8B/config.json \\
        --gpu-arch results/gpu_arch.json \\
        --output-dir optimized/ \\
        --threshold 0.5

Usage (tool mode — called by sub-agent):
    python kernel_optimize_agent.py benchmark --kernel optimized/gemm/attempt_3.py --shapes '[[64,4096],[4096,12288]]'
    python kernel_optimize_agent.py correctness --kernel optimized/gemm/attempt_3.py --shapes '[[64,4096],[4096,12288]]'
    python kernel_optimize_agent.py rocprof --kernel optimized/gemm/attempt_3.py --shapes '[[64,4096],[4096,12288]]'
    python kernel_optimize_agent.py accept --kernel optimized/gemm/attempt_3.py --name "v3_split_k"
    python kernel_optimize_agent.py reject --name "v3_split_k" --reason "regressed on bs=256"
    python kernel_optimize_agent.py status --kernel-type gemm
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import sqlite3
import glob

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_module(path):
    spec = importlib.util.spec_from_file_location("_k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

KERNEL_TYPE_PATTERNS = {
    "gemm":      ["Cijk_", "gemm", "mm", "matmul", "wvSplitK", "ck_tile"],
    "rmsnorm":   ["rmsnorm", "rms_norm", "fused_add_rms"],
    "layernorm": ["layernorm", "layer_norm"],
    "swiglu":    ["act_and_mul", "silu", "swiglu"],
    "rotary":    ["rotary_embedding", "rotary"],
    "attention": ["attention", "paged_attention", "flash_attn"],
}

def classify_kernel_type(name):
    lower = name.lower()
    for ktype, pats in KERNEL_TYPE_PATTERNS.items():
        if any(p.lower() in lower for p in pats):
            return ktype
    return "other"


# ═══════════════════════════════════════════════════════════════════════════
# Tool: benchmark
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_dtype(kernel_path):
    """Try to read dtype from kernel file comment or log, default bf16."""
    try:
        with open(kernel_path) as f:
            for line in f:
                if "Model dtype:" in line:
                    if "float16" in line: return torch.float16
                    if "float32" in line: return torch.float32
                    return torch.bfloat16
                if line.strip() and not line.startswith("#"):
                    break
    except Exception:
        pass
    return torch.bfloat16


def do_benchmark(kernel_path, shapes_list, warmup=10, repeats=50):
    """Benchmark a kernel file. The file must define `reference(...)` and `optimized(...)`."""
    mod = load_module(kernel_path)
    dtype = _resolve_dtype(kernel_path)
    results = {}
    for shapes in shapes_list:
        tag = "x".join(str(s) for s in shapes[0]) + "_x_" + "x".join(str(s) for s in shapes[1])
        torch.cuda.empty_cache()
        inputs = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]
        perf = {}
        for fn_name in ["reference", "optimized"]:
            fn = getattr(mod, fn_name)
            for _ in range(warmup):
                fn(*[x.clone() for x in inputs])
            torch.cuda.synchronize()
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn(*[x.clone() for x in inputs])
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
            times.sort()
            trim = max(1, len(times) // 10)
            trimmed = times[trim:-trim] if len(times) > 2 * trim else times
            perf[fn_name] = {
                "median_us": round(trimmed[len(trimmed) // 2], 2),
                "mean_us": round(sum(trimmed) / len(trimmed), 2),
                "min_us": round(trimmed[0], 2),
                "max_us": round(trimmed[-1], 2),
            }
        speedup = perf["reference"]["median_us"] / perf["optimized"]["median_us"] \
            if perf["optimized"]["median_us"] > 0 else 0
        results[tag] = {**perf, "speedup": round(speedup, 4)}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Tool: correctness
# ═══════════════════════════════════════════════════════════════════════════

def do_correctness(kernel_path, shapes_list, atol=1e-2, rtol=1e-2, trials=3):
    """Correctness test. Returns {shape_tag: {correct: bool, detail: str}}."""
    mod = load_module(kernel_path)
    dtype = _resolve_dtype(kernel_path)
    results = {}
    for shapes in shapes_list:
        tag = "x".join(str(s) for s in shapes[0]) + "_x_" + "x".join(str(s) for s in shapes[1])
        ok = True
        detail = "PASS"
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            inputs = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]
            ref = mod.reference(*[x.clone() for x in inputs])
            opt = mod.optimized(*[x.clone() for x in inputs])
            if not isinstance(ref, torch.Tensor): ref = ref[0]
            if not isinstance(opt, torch.Tensor): opt = opt[0]
            if not torch.allclose(ref, opt, atol=atol, rtol=rtol):
                ok = False
                detail = f"trial {trial}: max_diff={float((ref - opt).abs().max()):.6f}"
                break
        results[tag] = {"correct": ok, "detail": detail}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Tool: rocprof (rocprofv3 kernel trace)
# ═══════════════════════════════════════════════════════════════════════════

def do_rocprof(kernel_path, shapes_list, output_dir):
    """Run rocprofv3 kernel trace. Returns HW info (VGPR, SGPR, LDS, dispatch times)."""
    rocprofv3 = "/opt/rocm/bin/rocprofv3"
    if not os.path.exists(rocprofv3):
        r = subprocess.run(["which", "rocprofv3"], capture_output=True, text=True)
        if r.returncode != 0:
            return {"status": "skipped", "reason": "rocprofv3 not in PATH"}
        rocprofv3 = r.stdout.strip()

    prof_dir = os.path.join(output_dir, "rocprof_runs", str(int(time.time())))
    os.makedirs(prof_dir, exist_ok=True)

    shapes_repr = repr([list(s) for s in shapes_list[0]])
    bench_py = os.path.join(prof_dir, "bench.py")
    with open(bench_py, "w") as f:
        f.write(f"import torch, importlib.util, os\n"
                f"spec = importlib.util.spec_from_file_location('k', '{kernel_path}')\n"
                f"mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
                f"shapes = {shapes_repr}\n"
                f"# Pre-allocate inputs BEFORE profiling starts\n"
                f"inputs = [torch.randn(s, dtype=torch.bfloat16, device='cuda') for s in shapes]\n"
                f"# Warmup (outside profiled region — rocprofv3 traces the whole process,\n"
                f"# but we do enough reps that warmup kernels are negligible)\n"
                f"for _ in range(5): mod.optimized(*[x.clone() for x in inputs])\n"
                f"torch.cuda.synchronize()\n"
                f"# Measured region — these dispatches dominate the trace\n"
                f"for _ in range(50): mod.optimized(*inputs)\n"
                f"torch.cuda.synchronize()\n")

    try:
        r = subprocess.run(
            [rocprofv3, "--kernel-trace",
             "-o", os.path.join(prof_dir, "trace"),
             "-d", prof_dir,
             "--", sys.executable, bench_py],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"status": "error", "stderr": r.stderr[:500]}

        dbs = glob.glob(os.path.join(prof_dir, "*.db"))
        if not dbs:
            return {"status": "error", "reason": "no .db output"}

        conn = sqlite3.connect(dbs[0])
        c = conn.cursor()
        tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

        hw_info = []
        ks = [t for t in tables if "kernel_symbol" in t]
        if ks:
            c.execute(f"SELECT display_name, sgpr_count, arch_vgpr_count, accum_vgpr_count, "
                      f"group_segment_size, private_segment_size FROM {ks[0]}")
            # Filter: skip PyTorch init kernels, keep actual compute kernels
            skip_prefixes = ("void at::native::", "__amd_rocclr_")
            for row in c.fetchall():
                name = (row[0] or "?")
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                hw_info.append({
                    "kernel": name[:100],
                    "sgpr": row[1], "vgpr": row[2], "accum_vgpr": row[3],
                    "lds_bytes": row[4], "scratch_bytes": row[5],
                })
            # If we filtered everything, include top entries unfiltered
            if not hw_info:
                c.execute(f"SELECT display_name, sgpr_count, arch_vgpr_count, accum_vgpr_count, "
                          f"group_segment_size, private_segment_size FROM {ks[0]}")
                for row in c.fetchall():
                    hw_info.append({
                        "kernel": (row[0] or "?")[:100],
                        "sgpr": row[1], "vgpr": row[2], "accum_vgpr": row[3],
                        "lds_bytes": row[4], "scratch_bytes": row[5],
                    })

        dispatches = []
        kd = [t for t in tables if "kernel_dispatch" in t]
        if kd:
            # Join dispatch with kernel symbol to get kernel names
            ks_id_col = "kernel_id"
            try:
                c.execute(f"PRAGMA table_info({kd[0]})")
                kd_cols = [r[1] for r in c.fetchall()]
                if "kernel_id" not in kd_cols:
                    ks_id_col = None
            except Exception:
                ks_id_col = None

            c.execute(f"SELECT start, end, workgroup_size_x, workgroup_size_y, workgroup_size_z, "
                      f"grid_size_x, grid_size_y, grid_size_z"
                      f"{', kernel_id' if ks_id_col else ''} FROM {kd[0]} "
                      f"WHERE end > start ORDER BY (end - start) DESC LIMIT 30")
            for row in c.fetchall():
                dur_us = round((row[1] - row[0]) / 1000, 2)
                entry = {
                    "duration_us": dur_us,
                    "workgroup": f"{row[2]}x{row[3]}x{row[4]}",
                    "grid": f"{row[5]}x{row[6]}x{row[7]}",
                }
                # Try to get kernel name
                if ks_id_col and len(row) > 8 and ks:
                    try:
                        c.execute(f"SELECT display_name FROM {ks[0]} WHERE id = ?", (row[8],))
                        kr = c.fetchone()
                        if kr and kr[0]:
                            entry["kernel"] = kr[0][:100]
                    except Exception:
                        pass
                dispatches.append(entry)
        conn.close()
        return {"status": "success", "hw_info": hw_info[:15], "dispatches": dispatches[:10]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Tool: accept / reject / status
# ═══════════════════════════════════════════════════════════════════════════

def load_opt_log(log_path):
    if os.path.exists(log_path):
        return load_json(log_path)
    return {"baseline": {}, "optimization_path": [], "steps": [], "best_kernel": None, "best_avg_speedup": 0}

def do_accept(kernel_path, name, description, benchmark_results, log_path, kernel_dir):
    """Accept a kernel as the new best. Copy to best_kernel.py, update log."""
    import shutil
    log = load_opt_log(log_path)
    best_path = os.path.join(kernel_dir, "best_kernel.py")
    shutil.copy2(kernel_path, best_path)

    speedups = [v["speedup"] for v in benchmark_results.values() if "speedup" in v]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    log["optimization_path"].append(name)
    log["best_kernel"] = best_path
    log["best_avg_speedup"] = round(avg_speedup, 4)
    log["steps"].append({
        "step": len(log["steps"]) + 1,
        "name": name,
        "description": description,
        "action": "ACCEPTED",
        "avg_speedup": round(avg_speedup, 4),
        "per_shape": benchmark_results,
        "kernel_file": kernel_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_json(log, log_path)
    return {"status": "accepted", "avg_speedup": round(avg_speedup, 4),
            "path": log["optimization_path"]}

def do_reject(name, description, reason, benchmark_results, log_path):
    """Reject a kernel attempt. Log it."""
    log = load_opt_log(log_path)
    speedups = [v["speedup"] for v in benchmark_results.values() if "speedup" in v] if benchmark_results else []
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    log["steps"].append({
        "step": len(log["steps"]) + 1,
        "name": name,
        "description": description,
        "action": "REJECTED",
        "reason": reason,
        "avg_speedup": round(avg_speedup, 4),
        "per_shape": benchmark_results or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_json(log, log_path)
    return {"status": "rejected", "reason": reason}

def do_status(log_path):
    """Return current optimization status."""
    log = load_opt_log(log_path)
    accepted = [s for s in log["steps"] if s["action"] == "ACCEPTED"]
    rejected = [s for s in log["steps"] if s["action"] == "REJECTED"]
    return {
        "kernel_type": log.get("kernel_type", "unknown"),
        "dtype": log.get("dtype", "bfloat16"),
        "gpu_arch": log.get("gpu_arch", "unknown"),
        "total_attempts": len(log["steps"]),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "consecutive_rejections": _count_trailing_rejections(log["steps"]),
        "max_attempts": log.get("max_attempts", 8),
        "max_consecutive_rejections": log.get("max_consecutive_rejections", 3),
        "optimization_path": log["optimization_path"],
        "best_avg_speedup": log.get("best_avg_speedup", 0),
        "best_kernel": log.get("best_kernel"),
        "baseline": log.get("baseline", {}),
        "baseline_rocprof": log.get("baseline_rocprof", {}),
        "benchmark_shapes": log.get("benchmark_shapes", []),
        "knowledge_base": log.get("knowledge_base"),
        "steps_summary": [
            {"step": s["step"], "name": s["name"], "action": s["action"],
             "avg_speedup": s.get("avg_speedup", 0),
             "description": s.get("description", ""),
             "reason": s.get("reason", "")}
            for s in log["steps"]
        ],
    }

def do_serving_test(kernel_path, N, K, m_values, iterations):
    """Simulate serving: call the kernel with rapidly varying M values.
    This catches autotune regressions that don't show up in fixed-shape benchmarks.

    Returns: {optimized_avg_us, reference_avg_us, speedup, verdict, per_m_results}
    """
    mod = load_module(kernel_path)
    dtype = _resolve_dtype(kernel_path)
    import itertools

    m_cycle = itertools.cycle(m_values)

    # Pre-allocate weight (fixed)
    weight = torch.randn(N, K, dtype=dtype, device="cuda")  # (out, in) like vLLM

    # Warmup both
    for _ in range(5):
        x = torch.randn(1, K, dtype=dtype, device="cuda")
        mod.optimized(x, weight.t())
        mod.reference(x, weight.t())
    torch.cuda.synchronize()

    # Test optimized: varying M
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iterations):
        m = next(m_cycle)
        x = torch.randn(m, K, dtype=dtype, device="cuda")
        mod.optimized(x, weight.t())
    torch.cuda.synchronize()
    opt_total = (time.time() - t0) * 1e6  # us

    # Test reference: same sequence of M values
    m_cycle2 = itertools.cycle(m_values)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iterations):
        m = next(m_cycle2)
        x = torch.randn(m, K, dtype=dtype, device="cuda")
        mod.reference(x, weight.t())
    torch.cuda.synchronize()
    ref_total = (time.time() - t0) * 1e6

    opt_avg = opt_total / iterations
    ref_avg = ref_total / iterations
    speedup = ref_avg / opt_avg if opt_avg > 0 else 0

    verdict = "PASS" if speedup >= 0.9 else "FAIL_REGRESSION"
    if speedup < 0.5:
        verdict = "FAIL_SEVERE_REGRESSION"

    return {
        "optimized_total_us": round(opt_total, 1),
        "reference_total_us": round(ref_total, 1),
        "optimized_avg_us": round(opt_avg, 2),
        "reference_avg_us": round(ref_avg, 2),
        "speedup": round(speedup, 4),
        "iterations": iterations,
        "m_values": m_values,
        "N": N, "K": K,
        "verdict": verdict,
    }


def _count_trailing_rejections(steps):
    count = 0
    for s in reversed(steps):
        if s["action"] == "REJECTED":
            count += 1
        else:
            break
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Setup command — run once to create workspace
# ═══════════════════════════════════════════════════════════════════════════

def _detect_dtype(model_cfg_raw):
    """Detect model dtype from config. Returns a torch dtype string and torch dtype."""
    for key in ["torch_dtype"]:
        val = model_cfg_raw.get(key)
        if val:
            DTYPE_MAP = {
                "bfloat16": ("bfloat16", "torch.bfloat16"),
                "float16": ("float16", "torch.float16"),
                "float32": ("float32", "torch.float32"),
            }
            if val in DTYPE_MAP:
                return DTYPE_MAP[val]
    # Check text_config
    tc = model_cfg_raw.get("text_config", {})
    for key in ["dtype", "torch_dtype"]:
        val = tc.get(key)
        if val and val in ("bfloat16", "float16", "float32"):
            DTYPE_MAP = {"bfloat16": ("bfloat16", "torch.bfloat16"),
                         "float16": ("float16", "torch.float16"),
                         "float32": ("float32", "torch.float32")}
            return DTYPE_MAP[val]
    return ("bfloat16", "torch.bfloat16")  # fallback


def _derive_shapes(ktype, model_cfg):
    """Derive benchmark shapes from model dimensions. Generic — works for any model."""
    hidden = model_cfg.get("hidden_size", 4096)
    inter = model_cfg.get("intermediate_size", hidden * 3)
    head_dim = model_cfg.get("head_dim", 128)
    nheads = model_cfg.get("num_attention_heads", 32)
    nkv = model_cfg.get("num_key_value_heads", nheads)
    batch_sizes = [1, 64, 256]

    if ktype == "gemm":
        # All linear projections in a transformer: QKV proj, up/gate proj, down proj
        return [[[bs, hidden], [hidden, inter]] for bs in batch_sizes] + \
               [[[bs, hidden], [hidden, hidden]] for bs in batch_sizes]
    elif ktype in ("rmsnorm", "layernorm"):
        return [[[bs, hidden], [bs, hidden]] for bs in [16, 64, 256]]
    elif ktype == "swiglu":
        return [[[bs, inter], [bs, inter]] for bs in [1, 16, 64, 256]]
    elif ktype == "rotary":
        return [[[bs, nheads, head_dim], [bs, nheads, head_dim]] for bs in [16, 64, 256]]
    elif ktype == "attention":
        seq_lens = [128, 512]
        return [[[bs, nheads, sl, head_dim], [bs, nkv, sl, head_dim]] for bs in [1, 4] for sl in seq_lens]
    else:
        # Generic fallback: treat as elementwise with 2 inputs of shape [bs, hidden]
        return [[[bs, hidden], [bs, hidden]] for bs in batch_sizes]


def cmd_setup(args):
    gap = load_json(args.gap_analysis)
    model_cfg_raw = load_json(args.model_config)
    model_cfg = model_cfg_raw.get("text_config", model_cfg_raw)
    gpu_arch = load_json(args.gpu_arch) if args.gpu_arch and os.path.exists(args.gpu_arch) else {}

    hidden = model_cfg.get("hidden_size", 4096)
    inter = model_cfg.get("intermediate_size", 12288)
    dtype_name, dtype_torch = _detect_dtype(model_cfg_raw)

    # Find knowledge base path (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_candidates = [
        os.path.join(script_dir, "..", "references", "TRITON_OPTIMIZATION_KNOWLEDGE.md"),
        os.path.join(script_dir, "TRITON_OPTIMIZATION_KNOWLEDGE.md"),
    ]
    knowledge_path = None
    for kp in knowledge_candidates:
        if os.path.exists(kp):
            knowledge_path = os.path.abspath(kp)
            break

    print("═" * 72)
    print(" Kernel Optimization Agent — Setup")
    print("═" * 72)
    print(f" hidden={hidden}, intermediate={inter}, dtype={dtype_name}, gpu={gpu_arch.get('gpu_arch', '?')}")
    if knowledge_path:
        print(f" Knowledge base: {knowledge_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Identify targets
    targets = []
    seen = set()
    for k in gap.get("top_kernels", []):
        pct = k.get("pct_total", 0)
        if pct < args.threshold:
            continue
        ktype = classify_kernel_type(k["name"])
        if ktype not in seen and ktype != "other" and ktype != "memory" and ktype != "reduce":
            seen.add(ktype)
            targets.append({"type": ktype, "name": k["name"], "pct": pct,
                            "raw_kernel_info": k})

    # For each target, create workspace and measure baseline
    all_targets = []
    for t in targets:
        ktype = t["type"]
        kernel_dir = os.path.join(args.output_dir, ktype)
        os.makedirs(kernel_dir, exist_ok=True)
        log_path = os.path.join(kernel_dir, "optimization_log.json")

        # Use REAL shapes from trace if available; fall back to model config
        bench_shapes = None
        if args.real_shapes and os.path.exists(args.real_shapes) and ktype == "gemm":
            try:
                real = load_json(args.real_shapes)
                # Use the actual shapes from profiling — these are the shapes
                # that REALLY run during inference, not guesses from model config
                bench_shapes = real.get("benchmark_shapes", None)
                if bench_shapes:
                    # Deduplicate
                    seen = set()
                    unique = []
                    for s in bench_shapes:
                        key = str(s)
                        if key not in seen:
                            seen.add(key)
                            unique.append(s)
                    bench_shapes = unique
                    print(f"  [{ktype}] Using REAL shapes from trace: {len(bench_shapes)} unique shapes")
                    print(f"  [{ktype}] Real M values: {sorted(set(s[0][0] for s in bench_shapes))}")
            except Exception as e:
                print(f"  [{ktype}] Failed to load real shapes: {e}")
                bench_shapes = None

        if not bench_shapes:
            bench_shapes = _derive_shapes(ktype, model_cfg)
            print(f"  [{ktype}] Using model-config-derived shapes (no trace data)")

        # Write a minimal baseline kernel (just torch reference) so we can measure
        baseline_path = os.path.join(kernel_dir, "baseline.py")
        _write_baseline_kernel(baseline_path, ktype, hidden, inter, model_cfg, dtype_name)

        # Measure baseline
        print(f"\n  [{ktype}] Measuring PyTorch baseline...")
        try:
            baseline_perf = do_benchmark(baseline_path, bench_shapes)
            for tag, p in baseline_perf.items():
                print(f"    {tag}: ref={p['reference']['median_us']:.1f}us")
        except Exception as e:
            print(f"    FAILED: {e}")
            baseline_perf = {}

        # Run initial rocprof
        print(f"  [{ktype}] Running rocprofv3 baseline trace...")
        rp = do_rocprof(baseline_path, bench_shapes, kernel_dir)
        if rp.get("status") == "success":
            for hi in rp.get("hw_info", []):
                print(f"    HW: {hi['kernel'][:50]}  VGPR={hi['vgpr']}  SGPR={hi['sgpr']}  LDS={hi['lds_bytes']}")
            for di in rp.get("dispatches", [])[:3]:
                print(f"    Dispatch: {di['duration_us']}us  wg={di['workgroup']}  grid={di['grid']}")
        else:
            print(f"    rocprof: {rp.get('status')} {rp.get('reason', '')}")

        # Save initial log
        log = {
            "kernel_type": ktype,
            "original_kernel": t["name"],
            "original_pct": t["pct"],
            "model_config": {"hidden_size": hidden, "intermediate_size": inter},
            "dtype": dtype_name,
            "dtype_torch": dtype_torch,
            "gpu_arch": gpu_arch.get("gpu_arch", "unknown"),
            "benchmark_shapes": bench_shapes,
            "baseline": baseline_perf,
            "baseline_rocprof": rp,
            "knowledge_base": knowledge_path,
            "max_attempts": args.max_attempts,
            "max_consecutive_rejections": args.max_rejections,
            "optimization_path": [],
            "best_kernel": None,
            "best_avg_speedup": 0,
            "steps": [],
        }
        save_json(log, log_path)

        all_targets.append({
            "type": ktype,
            "name": t["name"],
            "pct": t["pct"],
            "kernel_dir": kernel_dir,
            "log_path": log_path,
            "bench_shapes": bench_shapes,
        })
        print(f"  [{ktype}] Workspace ready: {kernel_dir}")

    # Save manifest
    manifest = {
        "targets": all_targets,
        "model_config": {"hidden_size": hidden, "intermediate_size": inter},
        "dtype": dtype_name,
        "dtype_torch": dtype_torch,
        "gpu_arch": gpu_arch,
        "threshold": args.threshold,
        "knowledge_base": knowledge_path,
        "max_attempts": args.max_attempts,
        "max_consecutive_rejections": args.max_rejections,
    }
    save_json(manifest, os.path.join(args.output_dir, "manifest.json"))
    print(f"\nSetup complete. Manifest: {args.output_dir}/manifest.json")
    print(f"Targets: {len(all_targets)}")
    for t in all_targets:
        print(f"  [{t['type']}] {t['name'][:55]} — {t['pct']:.1f}%")


def _write_baseline_kernel(path, ktype, hidden, inter, model_cfg, dtype_name="bfloat16"):
    """Write a minimal .py that defines reference() and optimized() = reference.
    Uses the model's actual dtype, not hardcoded."""
    header = f"import torch\n# Model dtype: {dtype_name}\n"
    if ktype == "gemm":
        code = header + "def reference(A, B): return torch.mm(A, B)\ndef optimized(A, B): return torch.mm(A, B)\n"
    elif ktype in ("rmsnorm", "layernorm"):
        code = header + (
            "def reference(x, residual):\n"
            "    h = x.float() + residual.float()\n"
            "    var = h.pow(2).mean(-1, keepdim=True)\n"
            "    return (h * torch.rsqrt(var + 1e-6)).to(x.dtype), h.to(x.dtype)\n"
            "def optimized(x, residual): return reference(x, residual)\n"
        )
    elif ktype == "swiglu":
        code = header + (
            "def reference(gate, up): return torch.nn.functional.silu(gate) * up\n"
            "def optimized(gate, up): return reference(gate, up)\n"
        )
    elif ktype == "rotary":
        code = header + (
            "import torch.nn.functional as F\n"
            "def reference(x, freqs):\n"
            "    # Simplified rotary — real impl depends on position encoding\n"
            "    return x * freqs.cos() + torch.roll(x, 1, -1) * freqs.sin()\n"
            "def optimized(x, freqs): return reference(x, freqs)\n"
        )
    else:
        # Generic fallback — identity
        code = header + "def reference(*args): return args[0]\ndef optimized(*args): return reference(*args)\n"
    with open(path, "w") as f:
        f.write(code)


# ═══════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kernel Optimization Agent")
    sub = parser.add_subparsers(dest="command")

    # setup
    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--gap-analysis", required=True)
    p_setup.add_argument("--model-config", required=True)
    p_setup.add_argument("--gpu-arch", default=None)
    p_setup.add_argument("--output-dir", required=True)
    p_setup.add_argument("--threshold", type=float, default=0.5)
    p_setup.add_argument("--max-attempts", type=int, default=8,
                         help="Max optimization attempts per kernel (default: 8)")
    p_setup.add_argument("--max-rejections", type=int, default=3,
                         help="Stop after N consecutive rejections (default: 3)")
    p_setup.add_argument("--real-shapes", default=None,
                         help="Path to real_shapes.json from Phase 5 (extracted from trace)")

    # benchmark
    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--kernel", required=True)
    p_bench.add_argument("--shapes", required=True, help="JSON list of shape pairs")

    # correctness
    p_corr = sub.add_parser("correctness")
    p_corr.add_argument("--kernel", required=True)
    p_corr.add_argument("--shapes", required=True)

    # rocprof
    p_rp = sub.add_parser("rocprof")
    p_rp.add_argument("--kernel", required=True)
    p_rp.add_argument("--shapes", required=True)
    p_rp.add_argument("--output-dir", default=".")

    # accept
    p_acc = sub.add_parser("accept")
    p_acc.add_argument("--kernel", required=True)
    p_acc.add_argument("--name", required=True)
    p_acc.add_argument("--description", default="")
    p_acc.add_argument("--benchmark-results", required=True, help="JSON string")
    p_acc.add_argument("--log-path", required=True)
    p_acc.add_argument("--kernel-dir", required=True)

    # reject
    p_rej = sub.add_parser("reject")
    p_rej.add_argument("--name", required=True)
    p_rej.add_argument("--description", default="")
    p_rej.add_argument("--reason", required=True)
    p_rej.add_argument("--benchmark-results", default="{}")
    p_rej.add_argument("--log-path", required=True)

    # status
    p_stat = sub.add_parser("status")
    p_stat.add_argument("--log-path", required=True)

    # serving-test: simulates dynamic-M serving to catch autotune regressions
    p_srv = sub.add_parser("serving-test",
        help="Test kernel with varying M (simulates serving). Catches autotune regressions.")
    p_srv.add_argument("--kernel", required=True)
    p_srv.add_argument("--n", type=int, default=4096, help="N dimension (out_features)")
    p_srv.add_argument("--k", type=int, default=4096, help="K dimension (in_features)")
    p_srv.add_argument("--m-values", default="1,2,3,5,8,13,21,34,55,64",
                       help="Comma-separated M values to cycle through (simulates dynamic batching)")
    p_srv.add_argument("--iterations", type=int, default=100,
                       help="Total kernel calls across varying M values")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "benchmark":
        shapes = json.loads(args.shapes)
        r = do_benchmark(args.kernel, shapes)
        print(json.dumps(r, indent=2))
    elif args.command == "correctness":
        shapes = json.loads(args.shapes)
        r = do_correctness(args.kernel, shapes)
        print(json.dumps(r, indent=2))
    elif args.command == "rocprof":
        shapes = json.loads(args.shapes)
        r = do_rocprof(args.kernel, shapes, args.output_dir)
        print(json.dumps(r, indent=2))
    elif args.command == "accept":
        br = json.loads(args.benchmark_results)
        r = do_accept(args.kernel, args.name, args.description, br, args.log_path, args.kernel_dir)
        print(json.dumps(r, indent=2))
    elif args.command == "reject":
        br = json.loads(args.benchmark_results) if args.benchmark_results != "{}" else {}
        r = do_reject(args.name, args.description, args.reason, br, args.log_path)
        print(json.dumps(r, indent=2))
    elif args.command == "status":
        r = do_status(args.log_path)
        print(json.dumps(r, indent=2))
    elif args.command == "serving-test":
        r = do_serving_test(args.kernel, args.n, args.k,
                            [int(x) for x in args.m_values.split(",")],
                            args.iterations)
        print(json.dumps(r, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
