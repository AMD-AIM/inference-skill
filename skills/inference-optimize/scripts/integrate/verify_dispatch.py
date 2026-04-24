#!/usr/bin/env python3
"""Run rocprofv3 kernel-trace and verify dispatch against expected symbols.

Three modes:
  --baseline-only   (Phase 6): run on the already-Phase-2 baseline, capture
                    rocprofv3 JSON to results/baseline_dispatch_trace.json.
  --pre-flight      (Phase 7 step 9): fast sanity check against the
                    rebuilt-fork env that the expected kernel symbols fire
                    before Phase 8 burns wall-clock on full e2e. Writes
                    results/preflight_dispatch_trace.json.
  --post-rebuild    (Phase 8 step 2): full verification. Compares against
                    `expected_dispatch_symbols` and `vendor_baseline_symbols`
                    from the optimization manifest, diffs against
                    baseline_dispatch_trace.json, writes
                    results/dispatch_verification.json.

Usage:
  verify_dispatch.py --output-dir <path> --mode {baseline-only|pre-flight|post-rebuild} \
      [--manifest problems/optimization_manifest.json] \
      [--launcher 'python -c "<minimal vllm decode>"']
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import warnings


def run_rocprofv3(launcher_cmd, output_json):
    """Invoke rocprofv3 --kernel-trace --output-format json on the launcher."""
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--output-format", "json",
        "--output-file", output_json,
        "--",
    ] + launcher_cmd
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _extract_dispatch_records(doc):
    """Return the list of kernel-dispatch records from a rocprofv3 JSON
    document at the canonical documented path for the schema version in
    use, or None if no recognized schema applies.

    Recognized schemas:
      - rocprofv3 >= 0.5 (current AMD ROCm release):
          {"kernel_dispatches": [ {kernel_name, ...}, ... ]}
      - rocprofv3 0.4.x (older):
          {"rocprofiler-sdk-tool":
             {"buffer_records": {"kernel_dispatch": [ {kernel_name, ...} ]}}}

    A direct lookup is used rather than a recursive walker so the same
    kernel name appearing at multiple nesting levels (which has been
    observed across rocprofv3 versions) cannot be double-counted.
    """
    if isinstance(doc, dict):
        # rocprofv3 >= 0.5 -- top-level "kernel_dispatches" list
        kd = doc.get("kernel_dispatches")
        if isinstance(kd, list):
            return kd
        # rocprofv3 0.4.x -- nested under "rocprofiler-sdk-tool"
        sdk = doc.get("rocprofiler-sdk-tool")
        if isinstance(sdk, dict):
            buf = sdk.get("buffer_records")
            if isinstance(buf, dict):
                kd = buf.get("kernel_dispatch")
                if isinstance(kd, list):
                    return kd
    return None


def parse_kernel_counts(trace_path):
    """Aggregate kernel-name -> count from a rocprofv3 JSON trace.

    Reads the documented schema path (see _extract_dispatch_records) and
    counts `kernel_name` occurrences at exactly one nesting level so a
    kernel cannot be double-counted by the structure of the file. Emits
    a UserWarning when no recognized schema path is found rather than
    silently returning empty counts.
    """
    if not os.path.isfile(trace_path):
        return {}
    with open(trace_path) as f:
        doc = json.load(f)
    records = _extract_dispatch_records(doc)
    if records is None:
        warnings.warn(
            f"rocprofv3 trace at {trace_path!r} did not match any known "
            "schema path (kernel_dispatches | rocprofiler-sdk-tool."
            "buffer_records.kernel_dispatch). Kernel counts will be empty; "
            "dispatch verification will fail closed.",
            UserWarning,
            stacklevel=2,
        )
        return {}
    counts = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = rec.get("kernel_name") or rec.get("kernel") or rec.get("name")
        if isinstance(name, str):
            counts[name] = counts.get(name, 0) + 1
    return counts


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def evaluate(manifest, counts):
    """Return per-kernel verification status against the manifest."""
    kernels = manifest.get("optimizations", manifest.get("kernels", []))
    expected_records = []
    vendor_records = []
    redirect_required = 0
    redirect_honored = 0

    for k in kernels:
        if not k.get("optimize", True):
            continue
        for sym in k.get("expected_dispatch_symbols", []) or []:
            n = sum(c for name, c in counts.items() if fnmatch.fnmatchcase(name, sym))
            expected_records.append({
                "kernel": k.get("name"),
                "symbol": sym,
                "count": n,
                "status": "present" if n > 0 else "missing",
            })
        for sym in k.get("vendor_baseline_symbols", []) or []:
            n = sum(c for name, c in counts.items() if fnmatch.fnmatchcase(name, sym))
            vendor_records.append({
                "kernel": k.get("name"),
                "symbol": sym,
                "count": n,
                "status": "absent" if n == 0 else "leaked",
            })
        if k.get("geak_strategy", "").startswith("dispatch_redirect"):
            redirect_required += 1
            target_syms = k.get("expected_dispatch_symbols", []) or []
            vendor_syms = k.get("vendor_baseline_symbols", []) or []
            target_hit = any(
                sum(c for name, c in counts.items() if fnmatch.fnmatchcase(name, s)) > 0
                for s in target_syms
            )
            vendor_hit = any(
                sum(c for name, c in counts.items() if fnmatch.fnmatchcase(name, s)) > 0
                for s in vendor_syms
            )
            if target_hit and not vendor_hit:
                redirect_honored += 1

    expected_total = sum(r["count"] for r in expected_records)
    vendor_leaked = sum(1 for r in vendor_records if r["status"] == "leaked")
    expected_missing = sum(1 for r in expected_records if r["status"] == "missing")

    dispatch_verified = expected_missing == 0 and vendor_leaked == 0 and (
        redirect_required == redirect_honored
    )
    return {
        "expected_symbols": expected_records,
        "vendor_symbols": vendor_records,
        "expected_symbol_total_count": expected_total,
        "vendor_symbol_leaked_count": vendor_leaked,
        "redirect_required_count": redirect_required,
        "redirect_honored_count": redirect_honored,
        "dispatch_verified": dispatch_verified,
    }


DEFAULT_LAUNCHER = (
    "python -c \"from vllm import LLM, SamplingParams; "
    "llm=LLM(model='Qwen/Qwen3.5-9B', dtype='bfloat16', enforce_eager=False); "
    "_=llm.generate(['Hello world.'], SamplingParams(temperature=0.0, max_tokens=8))\""
)


def main():
    parser = argparse.ArgumentParser(description="rocprofv3 dispatch verifier")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["baseline-only", "pre-flight", "post-rebuild"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--launcher", default=DEFAULT_LAUNCHER)
    parser.add_argument("--trace-out", default=None)
    args = parser.parse_args()

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    name_map = {
        "baseline-only": "baseline_dispatch_trace.json",
        "pre-flight": "preflight_dispatch_trace.json",
        "post-rebuild": "postrebuild_dispatch_trace.json",
    }
    trace_path = args.trace_out or os.path.join(results_dir, name_map[args.mode])

    rc, stdout, stderr = run_rocprofv3(["bash", "-lc", args.launcher], trace_path)
    if rc != 0:
        sys.stderr.write(f"rocprofv3 returned {rc}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n")
        result = {"mode": args.mode, "rocprofv3_returncode": rc, "trace_path": trace_path,
                  "dispatch_verified": False}
        out_name = "dispatch_verification.json" if args.mode == "post-rebuild" else name_map[args.mode] + ".meta.json"
        with open(os.path.join(results_dir, out_name), "w") as f:
            json.dump(result, f, indent=2)
        json.dump(result, sys.stdout, indent=2)
        return 2

    counts = parse_kernel_counts(trace_path)

    if args.mode == "baseline-only":
        result = {
            "mode": "baseline-only",
            "rocprofv3_trace_path": trace_path,
            "kernel_counts": counts,
        }
        json.dump(result, sys.stdout, indent=2)
        return 0

    if args.mode == "pre-flight":
        if args.manifest is None:
            sys.stderr.write("--manifest required for pre-flight mode\n")
            return 2
        manifest = load_manifest(args.manifest)
        evaluation = evaluate(manifest, counts)
        result = {
            "mode": "pre-flight",
            "rocprofv3_trace_path": trace_path,
            "dispatch_pre_flight_pass": evaluation["dispatch_verified"],
            **evaluation,
        }
        json.dump(result, sys.stdout, indent=2)
        return 0 if evaluation["dispatch_verified"] else 3

    # post-rebuild
    if args.manifest is None:
        sys.stderr.write("--manifest required for post-rebuild mode\n")
        return 2
    manifest = load_manifest(args.manifest)
    evaluation = evaluate(manifest, counts)
    result = {
        "mode": "post-rebuild",
        "rocprofv3_trace_path": trace_path,
        **evaluation,
    }
    out_path = os.path.join(results_dir, "dispatch_verification.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    json.dump(result, sys.stdout, indent=2)
    return 0 if evaluation["dispatch_verified"] else 3


if __name__ == "__main__":
    sys.exit(main())
