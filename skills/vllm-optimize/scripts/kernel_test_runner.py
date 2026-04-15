#!/usr/bin/env python3
"""
Kernel Test Runner — Tests accuracy and benchmarks optimized Triton kernels.

Compares Model (reference) vs ModelNew (optimized) from source and target files.
Automatically tracks best result across multiple attempts.

Usage:
    python kernel_test_runner.py --src problem.py --target problem_opt.py
    python kernel_test_runner.py --src problem.py --target problem_opt.py --goal 1.5

Part of the model-optimize pipeline. Can be used standalone.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import statistics
import torch


def load_module(filepath, module_name):
    """Import a module from file path (required for Triton to get source)."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark optimized kernel")
    parser.add_argument("--src", required=True, help="Source file with Model + get_inputs()")
    parser.add_argument("--target", required=True, help="Target file with ModelNew")
    parser.add_argument("--goal", type=float, default=None, help="Target speedup ratio")
    parser.add_argument("--tracker", default=None, help="Path to best tracker JSON")
    parser.add_argument("--log", default=None, help="Path to history log file")
    args = parser.parse_args()

    src_path = os.path.abspath(args.src)
    tgt_path = os.path.abspath(args.target)

    if not os.path.exists(src_path):
        print(f"ERROR: Source file not found: {src_path}")
        sys.exit(1)
    if not os.path.exists(tgt_path):
        print(f"ERROR: Target file not found: {tgt_path}")
        sys.exit(1)

    # Default tracker/log paths
    tracker_path = args.tracker or tgt_path.replace(".py", "_best.json")
    log_path = args.log or tgt_path.replace(".py", "_history.log")

    # Initialize tracker if not exists
    if not os.path.exists(tracker_path):
        with open(tracker_path, "w") as f:
            json.dump({"best_speedup": 0, "best_code": "", "attempt": 0}, f, indent=2)

    # Load tracker
    with open(tracker_path) as f:
        tracker = json.load(f)
    attempt = tracker["attempt"] + 1

    print(f"\n{'='*50}")
    print(f"Testing Attempt {attempt}")
    print(f"  Source: {src_path}")
    print(f"  Target: {tgt_path}")
    if args.goal:
        print(f"  Goal:   {args.goal}x")
    print(f"{'='*50}\n")

    # Load source module
    try:
        src_mod = load_module(src_path, "src_module")
        for name in dir(src_mod):
            if not name.startswith("_"):
                globals()[name] = getattr(src_mod, name)
    except Exception as e:
        print(f"ERROR loading source: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Read target code snapshot, then load
    try:
        target_code = open(tgt_path).read()
        tgt_mod = load_module(tgt_path, "tgt_module")
        for name in dir(tgt_mod):
            if not name.startswith("_"):
                globals()[name] = getattr(tgt_mod, name)
    except Exception as e:
        print(f"ERROR loading target: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Get init inputs
    try:
        init_inputs = get_init_inputs()
    except Exception:
        init_inputs = []

    # Initialize models
    try:
        Model_cls = src_mod.Model
        ModelNew_cls = tgt_mod.ModelNew
        if init_inputs:
            model_ref = Model_cls(*init_inputs).cuda().eval()
            model_new = ModelNew_cls(*init_inputs).cuda().eval()
        else:
            model_ref = Model_cls().cuda().eval()
            model_new = ModelNew_cls().cuda().eval()
    except Exception as e:
        print(f"ERROR initializing models: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Sync weights from Model to ModelNew
    try:
        ref_state = model_ref.state_dict()
        new_state = model_new.state_dict()
        if ref_state and new_state:
            matched = 0
            for name, param in ref_state.items():
                if name in new_state and new_state[name].shape == param.shape:
                    new_state[name].copy_(param)
                    matched += 1
            if matched > 0:
                model_new.load_state_dict(new_state)
                print(f"[Synced {matched} parameters]")
    except Exception:
        pass

    # Get inputs
    inputs = [x.cuda() if hasattr(x, "cuda") else x for x in get_inputs()]

    # Auto-convert dtype
    input_dtype = None
    for x in inputs:
        if hasattr(x, "dtype") and x.dtype in [torch.float16, torch.bfloat16]:
            input_dtype = x.dtype
            break
    if input_dtype is not None:
        if hasattr(model_ref, "parameters") and any(True for _ in model_ref.parameters()):
            model_ref = model_ref.to(input_dtype)
        if hasattr(model_new, "parameters") and any(True for _ in model_new.parameters()):
            model_new = model_new.to(input_dtype)

    # Check quantized inputs
    def is_low_precision(dtype):
        return any(x in str(dtype) for x in ["int8", "float8", "uint8", "qint8"])
    has_quantized = any(is_low_precision(x.dtype) for x in inputs if hasattr(x, "dtype"))

    # Test accuracy
    with torch.no_grad():
        try:
            out_ref = model_ref(*inputs)
            out_new = model_new(*inputs)
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # Tolerance
    if has_quantized:
        rtol, atol = 5e-2, 5e-2
    elif out_ref.dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 1e-2, 1e-3
    else:
        rtol, atol = 1e-5, 1e-6

    max_diff = (out_ref - out_new).abs().max().item()
    denom = out_ref.abs() + 1e-8
    rel_diff = ((out_ref - out_new).abs() / denom).max().item()
    is_close = torch.allclose(out_ref, out_new, rtol=rtol, atol=atol)

    print(f"=== Accuracy ===")
    print(f"Max abs error: {max_diff:.2e}, Max rel error: {rel_diff:.2e}")
    print(f"Accuracy: {'PASSED' if is_close else 'FAILED'}")

    if not is_close:
        print("\nERROR: Accuracy test failed!")
        _log_attempt(log_path, attempt, failed=True, max_diff=max_diff, rel_diff=rel_diff)
        tracker["attempt"] = attempt
        with open(tracker_path, "w") as f:
            json.dump(tracker, f, indent=2)
        sys.exit(1)

    # Benchmark
    print("\n=== Benchmarking ===")
    for _ in range(20):
        model_ref(*inputs)
        model_new(*inputs)
    torch.cuda.synchronize()

    NUM_ROUNDS, N_PER_ROUND = 5, 100
    ref_times, new_times = [], []

    for r in range(NUM_ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_PER_ROUND):
            model_ref(*inputs)
        torch.cuda.synchronize()
        ref_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_PER_ROUND):
            model_new(*inputs)
        torch.cuda.synchronize()
        new_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)

    t_ref = statistics.median(ref_times)
    t_new = statistics.median(new_times)
    speedup = t_ref / t_new

    print(f"\n=== Performance (median of {NUM_ROUNDS} rounds) ===")
    print(f"PyTorch (ref): {t_ref:.4f} ms (std: {statistics.stdev(ref_times):.4f})")
    print(f"Triton (opt):  {t_new:.4f} ms (std: {statistics.stdev(new_times):.4f})")
    print(f"Speedup: {speedup:.2f}x")

    # Track best
    is_best = speedup > tracker["best_speedup"]
    if is_best:
        print(f"\n*** NEW BEST! {speedup:.2f}x > {tracker['best_speedup']:.2f}x ***")
        tracker["best_speedup"] = speedup
        tracker["best_code"] = target_code
        tracker["best_ref_time"] = t_ref
        tracker["best_opt_time"] = t_new
        tracker["best_attempt"] = attempt
    else:
        print(f"\nNot best. Current best: {tracker['best_speedup']:.2f}x (Attempt {tracker.get('best_attempt', '?')})")

    tracker["attempt"] = attempt
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    _log_attempt(log_path, attempt, speedup=speedup, t_ref=t_ref, t_new=t_new,
                 is_best=is_best, code=target_code)

    print(f"\nResults logged to: {log_path}")
    print(f"Tracker updated: {tracker_path}")

    # Print result as JSON for easy parsing
    result = {"speedup": speedup, "ref_ms": t_ref, "opt_ms": t_new,
              "accuracy": "PASSED", "is_best": is_best, "attempt": attempt}
    print(f"\nRESULT_JSON: {json.dumps(result)}")

    return speedup


def _log_attempt(log_path, attempt, failed=False, **kwargs):
    entry = f"\n## Attempt {attempt} - {time.strftime('%Y-%m-%dT%H:%M:%S')}"
    if failed:
        entry += f" - FAILED (accuracy)\nMax abs: {kwargs.get('max_diff', '?'):.2e}, Rel: {kwargs.get('rel_diff', '?'):.2e}\n"
    else:
        entry += f"{'  *** BEST ***' if kwargs.get('is_best') else ''}\n"
        entry += f"Speedup: {kwargs.get('speedup', 0):.2f}x\n"
        entry += f"Ref: {kwargs.get('t_ref', 0):.4f}ms, Opt: {kwargs.get('t_new', 0):.4f}ms\n"
    with open(log_path, "a") as f:
        f.write(entry)


if __name__ == "__main__":
    main()

