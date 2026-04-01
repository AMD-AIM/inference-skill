#!/usr/bin/env python3
"""
E2E validator for inferencex-optimize pipeline outputs.

Validates all phase artifacts, scans logs for errors, and generates
test_report.json + test_report.md in the output directory.

Usage:
    python3 e2e_optimize_test.py                           # interactive
    python3 e2e_optimize_test.py --target vllm             # vLLM test
    python3 e2e_optimize_test.py --target sglang           # SGLang test
    python3 e2e_optimize_test.py --output-dir <path>       # validate existing
"""

import argparse
import datetime
import glob
import gzip
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

TARGETS = {
    "vllm": {
        "config_key": "gptoss-fp4-mi355x-vllm",
        "model": "amd/gpt-oss-120b-w-mxfp4-a-fp8",
        "size": "120B dense",
        "framework": "vllm",
        "command": (
            'claude "Use inferencex-optimize for gptoss-fp4-mi355x-vllm '
            'with optimize workflow, TP=8, 1k1k, conc=4"'
        ),
    },
    "sglang": {
        "config_key": "dsr1-fp4-mi355x-sglang",
        "model": "amd/DeepSeek-R1-0528-MXFP4-Preview",
        "size": "671B MoE (37B active)",
        "framework": "sglang",
        "command": (
            'claude "Use inferencex-optimize for dsr1-fp4-mi355x-sglang '
            'with optimize workflow, TP=8, 1k1k, conc=4"'
        ),
    },
}

# Phase lists by mode
OPTIMIZE_PHASES = [
    "env", "config", "benchmark", "benchmark-analyze",
    "profile", "profile-analyze",
    "problem-generate", "kernel-optimize", "integration", "report-generate",
]

# Error patterns to scan in logs
ERROR_PATTERNS = [
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bFAIL(?:ED|URE)?\b", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"RuntimeError:"),
    re.compile(r"\bOOM\b|Out[Oo]f[Mm]emory|CUDA out of memory"),
    re.compile(r"CUDA error|HIP error"),
    re.compile(r"\b404\b.*(?:Not Found|not found)"),
    re.compile(r"exit code:\s*[1-9]"),
    re.compile(r"Killed|SIGKILL|SIGTERM"),
]

# False-positive patterns to skip
SKIP_PATTERNS = [
    re.compile(r"error_rate"),
    re.compile(r"error_count"),
    re.compile(r"--ignore.error"),
    re.compile(r"log.*error.*level", re.IGNORECASE),
    re.compile(r"ErrorIfNotLoaded"),
    re.compile(r"suppress.*error", re.IGNORECASE),
    # Optimization history logs: FAILED attempts are expected (iterative optimization)
    re.compile(r"FAILED \(accuracy\)"),
    re.compile(r"FAILED \(slower\)"),
    # SGLang/vLLM benign import warnings (model modules not available in this image)
    re.compile(r"Ignore import error"),
    # Plugin import warnings during optimized benchmark (expected when plugin is optional)
    re.compile(r"sglang_plugin import failed"),
    re.compile(r"vllm_plugin import failed"),
]


# ---------------------------------------------------------------------------
# Check result helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, phase, name, status, detail=""):
        self.phase = phase
        self.name = name
        self.status = status  # pass, fail, skip, warning
        self.detail = detail

    def to_dict(self):
        return {
            "phase": self.phase,
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
        }


class Issue:
    def __init__(self, source, severity, pattern, context, suggested_phase, analysis):
        self.source = source
        self.severity = severity
        self.pattern = pattern
        self.context = context
        self.suggested_phase = suggested_phase
        self.analysis = analysis

    def to_dict(self):
        return {
            "source": self.source,
            "severity": self.severity,
            "pattern": self.pattern,
            "context": self.context[:200],
            "suggested_phase": self.suggested_phase,
            "analysis": self.analysis,
        }


# ---------------------------------------------------------------------------
# Interactive target selection
# ---------------------------------------------------------------------------

def select_target():
    print("\nWhich test case do you want to run?")
    print("  1. vLLM   (gptoss-fp4-mi355x-vllm, 120B dense)")
    print("  2. SGLang (dsr1-fp4-mi355x-sglang, 671B MoE)")
    print("  3. Both   (run sequentially)")
    print("  4. Validate existing output directory")

    while True:
        choice = input("\n> ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        print("Please enter 1, 2, 3, or 4.")

    if choice == "4":
        path = input("Enter the output directory path: ").strip()
        return [("validate", path)]

    targets = []
    if choice in ("1", "3"):
        targets.append("vllm")
    if choice in ("2", "3"):
        targets.append("sglang")
    return targets


def guide_pipeline_run(target_key):
    target = TARGETS[target_key]
    print(f"\n{'='*60}")
    print(f"Test target: {target_key.upper()}")
    print(f"  Config:    {target['config_key']}")
    print(f"  Model:     {target['model']} ({target['size']})")
    print(f"  Framework: {target['framework']}")
    print(f"{'='*60}")
    print(f"\nRun this command in a separate terminal:\n")
    print(f"  {target['command']}")
    print(f"\nAfter the pipeline completes, enter the output directory path.")
    print("(It will be something like ~/inferencex_{config_key}_{timestamp}/)")

    while True:
        path = input("\nOutput directory: ").strip()
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            return path
        print(f"Directory not found: {path}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(output_dir):
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_entries(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("results", payload.get("entries", [payload]))
    return []


def get_throughput(data):
    return data.get("total_token_throughput", data.get("output_throughput", 0))


# ---------------------------------------------------------------------------
# Phase validation checks
# ---------------------------------------------------------------------------

def check_env(output_dir, config):
    checks = []
    progress_path = os.path.join(output_dir, "progress.json")
    if not os.path.isfile(progress_path):
        checks.append(CheckResult("env", "progress.json exists", "fail", "File not found"))
        return checks

    with open(progress_path) as f:
        progress = json.load(f)

    completed = progress.get("phases_completed", [])
    if "env" in completed:
        checks.append(CheckResult("env", "progress.json has env phase", "pass"))
    else:
        checks.append(CheckResult("env", "progress.json has env phase", "fail",
                                  f"phases_completed: {completed}"))
    return checks


def check_config(output_dir, config):
    checks = []
    sweep_path = os.path.join(output_dir, "results", "sweep_configs.json")
    if not os.path.isfile(sweep_path):
        checks.append(CheckResult("config", "sweep_configs.json exists", "skip", "File not found"))
        return checks

    with open(sweep_path) as f:
        sweep = json.load(f)

    if not isinstance(sweep, list) or len(sweep) == 0:
        checks.append(CheckResult("config", "sweep_configs.json valid", "fail",
                                  f"Expected non-empty array, got {type(sweep).__name__} len={len(sweep) if isinstance(sweep, list) else 'N/A'}"))
        return checks

    checks.append(CheckResult("config", "sweep_configs.json valid", "pass",
                              f"{len(sweep)} config(s)"))

    required_fields = {"image", "model", "tp", "conc", "isl", "osl"}
    entry = sweep[0]
    missing = required_fields - set(entry.keys())
    if missing:
        checks.append(CheckResult("config", "sweep config has required fields", "fail",
                                  f"Missing: {missing}"))
    else:
        checks.append(CheckResult("config", "sweep config has required fields", "pass"))

    return checks


def check_benchmark(output_dir, config):
    checks = []
    results_dir = os.path.join(output_dir, "results")
    result_files = [f for f in glob.glob(os.path.join(results_dir, "*.json"))
                    if not any(skip in os.path.basename(f) for skip in
                               ["benchmark_summary", "sweep_configs", "optimization",
                                "gap_analysis", "docker_run", "bottlenecks",
                                "profile_analysis", "gpu_arch"])]

    if not result_files:
        checks.append(CheckResult("benchmark", "benchmark result exists", "skip",
                                  "No result JSON found"))
        return checks

    checks.append(CheckResult("benchmark", "benchmark result exists", "pass",
                              f"{len(result_files)} file(s)"))

    for rf in result_files:
        data = load_json(rf)
        tps = get_throughput(data)
        basename = os.path.basename(rf)
        if tps and tps > 0:
            checks.append(CheckResult("benchmark", f"{basename} throughput > 0", "pass",
                                      f"{tps:.1f} tok/s"))
        else:
            checks.append(CheckResult("benchmark", f"{basename} throughput > 0", "fail",
                                      f"throughput={tps}"))
    return checks


def check_benchmark_analyze(output_dir, config):
    checks = []
    summary_path = os.path.join(output_dir, "results", "benchmark_summary.json")
    if not os.path.isfile(summary_path):
        checks.append(CheckResult("benchmark-analyze", "benchmark_summary.json exists", "skip"))
        return checks

    with open(summary_path) as f:
        summary = json.load(f)

    if "configs" in summary or "results" in summary:
        checks.append(CheckResult("benchmark-analyze", "benchmark_summary.json valid", "pass"))
    else:
        checks.append(CheckResult("benchmark-analyze", "benchmark_summary.json valid", "fail",
                                  f"Missing configs/results key. Keys: {list(summary.keys())}"))
    return checks


def check_profile(output_dir, config):
    checks = []
    profile_dir = os.path.join(output_dir, "profiles")
    traces = glob.glob(os.path.join(profile_dir, "*.pt.trace.json*"))
    if not traces:
        # SGLang uses different naming: *-TP-N-DECODE.trace.json.gz, merged-*.trace.json.gz
        traces = [f for f in glob.glob(os.path.join(profile_dir, "*.trace.json*"))
                  if not os.path.basename(f).endswith("_docker_run.log")]
    if not traces:
        # Fall back to any .json.gz files that contain trace data
        traces = [f for f in glob.glob(os.path.join(profile_dir, "*.json.gz"))
                  if "docker" not in os.path.basename(f).lower()]

    if not traces:
        checks.append(CheckResult("profile", "trace files exist", "skip",
                                  "No trace files found"))
        return checks

    checks.append(CheckResult("profile", "trace files exist", "pass",
                              f"{len(traces)} file(s)"))

    for trace in traces:
        size = os.path.getsize(trace)
        basename = os.path.basename(trace)
        if size > 1_000_000:
            checks.append(CheckResult("profile", f"{basename} size > 1MB", "pass",
                                      f"{size / 1e6:.1f} MB"))
        else:
            checks.append(CheckResult("profile", f"{basename} size > 1MB", "warning",
                                      f"{size / 1e6:.1f} MB (may be incomplete)"))

        if trace.endswith(".gz"):
            try:
                with gzip.open(trace, "rb") as gz:
                    while True:
                        chunk = gz.read(1024 * 1024)
                        if not chunk:
                            break
                checks.append(CheckResult("profile", f"{basename} gzip integrity", "pass"))
            except (gzip.BadGzipFile, EOFError, OSError) as e:
                checks.append(CheckResult("profile", f"{basename} gzip integrity", "fail",
                                          str(e)))
    return checks


def check_profile_analyze(output_dir, config):
    checks = []
    gap_path = os.path.join(output_dir, "results", "gap_analysis", "gap_analysis.json")
    if not os.path.isfile(gap_path):
        checks.append(CheckResult("profile-analyze", "gap_analysis.json exists", "skip"))
        return checks

    with open(gap_path) as f:
        gap = json.load(f)

    kernels = gap.get("top_kernels", [])
    if len(kernels) > 0:
        checks.append(CheckResult("profile-analyze", "gap_analysis has top_kernels", "pass",
                                  f"{len(kernels)} kernel(s)"))
    else:
        checks.append(CheckResult("profile-analyze", "gap_analysis has top_kernels", "fail",
                                  "Empty top_kernels"))

    if "category_breakdown" in gap:
        checks.append(CheckResult("profile-analyze", "gap_analysis has category_breakdown", "pass"))
    else:
        checks.append(CheckResult("profile-analyze", "gap_analysis has category_breakdown", "fail"))

    return checks


def check_problem_generate(output_dir, config):
    checks = []
    problems_dir = os.path.join(output_dir, "problems")
    problem_files = glob.glob(os.path.join(problems_dir, "problem_*.py"))
    problem_files = [f for f in problem_files if "_opt" not in os.path.basename(f)]

    if not problem_files:
        checks.append(CheckResult("problem-generate", "problem files exist", "skip"))
        return checks

    checks.append(CheckResult("problem-generate", "problem files exist", "pass",
                              f"{len(problem_files)} file(s)"))

    for pf in problem_files:
        basename = os.path.basename(pf)
        with open(pf) as f:
            content = f.read()
        # Base problem files have: class Model, get_inputs, get_init_inputs
        # class ModelNew is only in the *_opt.py files (generated during Phase 7)
        missing = []
        for pattern in ["class Model", "def get_inputs", "def get_init_inputs"]:
            if pattern not in content:
                missing.append(pattern)
        if missing:
            checks.append(CheckResult("problem-generate", f"{basename} contract", "fail",
                                      f"Missing: {missing}"))
        else:
            checks.append(CheckResult("problem-generate", f"{basename} contract", "pass"))

    manifest_path = os.path.join(problems_dir, "optimization_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = manifest.get("optimizations", manifest.get("entries", []))
        checks.append(CheckResult("problem-generate", "optimization_manifest.json valid", "pass",
                                  f"{len(entries)} entries"))
    else:
        checks.append(CheckResult("problem-generate", "optimization_manifest.json exists", "skip"))

    return checks


def check_kernel_optimize(output_dir, config):
    checks = []
    problems_dir = os.path.join(output_dir, "problems")

    geak_path = os.path.join(problems_dir, "geak_results.json")
    if os.path.isfile(geak_path):
        entries = extract_entries(load_json(geak_path))

        checks.append(CheckResult("kernel-optimize", "geak_results.json exists", "pass",
                                  f"{len(entries)} entries"))

        # List of *_opt.py files in optimized/ (staged for integration)
        optimized_dir = os.path.join(output_dir, "optimized")
        staged_opt_files = {os.path.basename(f)
                           for f in glob.glob(os.path.join(optimized_dir, "*_opt.py"))}

        for entry in entries:
            name = entry.get("name", entry.get("kernel", "unknown"))
            speedup = entry.get("speedup")
            correct = entry.get("correctness", entry.get("correct", True))
            opt_filename = f"{name}_opt.py"
            in_optimized = opt_filename in staged_opt_files

            if speedup is not None:
                status = "pass" if correct else "fail"
                checks.append(CheckResult("kernel-optimize", f"{name} result", status,
                                          f"speedup={speedup}x, correct={correct}"))

                # Cross-check: kernel staging correctness
                if speedup < 1.0:
                    if in_optimized:
                        checks.append(CheckResult("kernel-optimize",
                                                  f"{name} regression not staged", "fail",
                                                  f"speedup={speedup}x < 1.0 but {opt_filename} found in optimized/"))
                    else:
                        checks.append(CheckResult("kernel-optimize",
                                                  f"{name} regression correctly skipped", "pass",
                                                  f"speedup={speedup}x < 1.0, not in optimized/"))
                else:
                    if in_optimized:
                        checks.append(CheckResult("kernel-optimize",
                                                  f"{name} winner correctly staged", "pass",
                                                  f"speedup={speedup}x, {opt_filename} in optimized/"))
                    else:
                        checks.append(CheckResult("kernel-optimize",
                                                  f"{name} winner staged", "warning",
                                                  f"speedup={speedup}x but {opt_filename} not in optimized/"))
            else:
                checks.append(CheckResult("kernel-optimize", f"{name} result", "warning",
                                          "speedup=null"))
    else:
        checks.append(CheckResult("kernel-optimize", "geak_results.json exists", "skip"))

    opt_files = glob.glob(os.path.join(problems_dir, "*_opt.py"))
    if opt_files:
        checks.append(CheckResult("kernel-optimize", "optimized kernel files exist", "pass",
                                  f"{len(opt_files)} file(s)"))
    else:
        opt_files_optimized = glob.glob(os.path.join(output_dir, "optimized", "*_opt.py"))
        if opt_files_optimized:
            checks.append(CheckResult("kernel-optimize", "optimized kernel files exist", "pass",
                                      f"{len(opt_files_optimized)} file(s) in optimized/"))
        else:
            checks.append(CheckResult("kernel-optimize", "optimized kernel files exist", "skip"))

    return checks


def check_integration(output_dir, config):
    checks = []
    framework = config.get("FRAMEWORK", "vllm") if config else "vllm"

    plugin_dir_name = f"{framework}_plugin"
    manifest_path = os.path.join(output_dir, "optimized", plugin_dir_name, "manifest.json")

    # Load geak_results for cross-check
    geak_path = os.path.join(output_dir, "problems", "geak_results.json")
    geak_speedups = {}
    if os.path.isfile(geak_path):
        geak_entries = extract_entries(load_json(geak_path))
        for entry in geak_entries:
            name = entry.get("name", entry.get("kernel", ""))
            geak_speedups[name] = entry.get("speedup")

    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

        registered = manifest.get("registered", [])
        ops_key = "registered_ops" if framework == "vllm" else "patched_modules"
        ops = manifest.get(ops_key, manifest.get("ops", registered))
        if ops:
            checks.append(CheckResult("integration", f"{plugin_dir_name} manifest valid", "pass",
                                      f"{len(ops)} {ops_key}"))
        else:
            # Check if there are winning kernels that should have been registered
            winners = [n for n, s in geak_speedups.items() if s is not None and s > 1.0]
            detail = f"Empty {ops_key}"
            if winners:
                detail += f" (winning kernels exist: {', '.join(winners)})"
            checks.append(CheckResult("integration", f"{plugin_dir_name} manifest valid", "warning",
                                      detail))

        # Cross-check: no regressed kernel in manifest
        for reg in registered:
            kernel_file = reg.get("kernel", "")
            # Extract kernel name from filename like "problem_fused_rmsnorm_opt.py"
            kernel_name = kernel_file.replace("problem_", "").replace("_opt.py", "").replace("_opt", "")
            # Find matching geak entry
            matched_speedup = None
            for geak_name, spd in geak_speedups.items():
                if kernel_name in geak_name or geak_name in kernel_name:
                    matched_speedup = spd
                    break
            if matched_speedup is not None and matched_speedup < 1.0:
                checks.append(CheckResult("integration",
                                          f"registered kernel {kernel_file} not regressed", "fail",
                                          f"speedup={matched_speedup}x < 1.0 — regressed kernel in plugin"))
            elif matched_speedup is not None:
                checks.append(CheckResult("integration",
                                          f"registered kernel {kernel_file} not regressed", "pass",
                                          f"speedup={matched_speedup}x"))
    else:
        checks.append(CheckResult("integration", f"{plugin_dir_name} manifest exists", "skip"))

    comparison_path = os.path.join(output_dir, "results", "optimization_comparison.json")
    if os.path.isfile(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)

        validated = comparison.get("validated", False)
        speedup = comparison.get("speedup")

        if validated and speedup is not None:
            checks.append(CheckResult("integration", "optimization_comparison validated", "pass",
                                      f"speedup={speedup:.3f}x"))
            if speedup < 1.0:
                bl = comparison.get("baseline", {}).get("total_token_throughput", "?")
                opt = comparison.get("optimized", {}).get("total_token_throughput", "?")
                checks.append(CheckResult("integration", "E2E performance not degraded", "fail",
                                          f"speedup={speedup:.3f}x < 1.0 (baseline={bl}, optimized={opt})"))
            else:
                checks.append(CheckResult("integration", "E2E performance not degraded", "pass",
                                          f"speedup={speedup:.3f}x"))
        else:
            detail = f"validated={validated}, speedup={speedup}"
            checks.append(CheckResult("integration", "optimization_comparison validated", "fail",
                                      detail))
    else:
        checks.append(CheckResult("integration", "optimization_comparison.json exists", "skip"))

    return checks


def check_report(output_dir, config):
    checks = []
    report_md = os.path.join(output_dir, "report", "optimization_report.md")
    if os.path.isfile(report_md):
        size = os.path.getsize(report_md)
        if size > 500:
            checks.append(CheckResult("report", "optimization_report.md valid", "pass",
                                      f"{size} bytes"))
        else:
            checks.append(CheckResult("report", "optimization_report.md valid", "warning",
                                      f"Only {size} bytes (may be incomplete)"))
    else:
        checks.append(CheckResult("report", "optimization_report.md exists", "skip"))

    summary_path = os.path.join(output_dir, "report", "optimization_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if "phases" in summary or "kernel_results" in summary:
            checks.append(CheckResult("report", "optimization_summary.json valid", "pass"))
        else:
            checks.append(CheckResult("report", "optimization_summary.json valid", "fail",
                                      f"Missing phases key. Keys: {list(summary.keys())}"))
    else:
        checks.append(CheckResult("report", "optimization_summary.json exists", "skip"))

    return checks


def check_progress_consistency(output_dir, config):
    checks = []
    progress_path = os.path.join(output_dir, "progress.json")
    if not os.path.isfile(progress_path):
        return checks

    with open(progress_path) as f:
        progress = json.load(f)

    completed = progress.get("phases_completed", [])
    mode = config.get("MODE", "optimize") if config else "optimize"

    if mode == "optimize":
        expected = OPTIMIZE_PHASES
    elif mode == "optimize-only":
        expected = ["env", "config", "problem-generate", "kernel-optimize",
                    "integration", "report-generate"]
    elif mode == "full":
        expected = OPTIMIZE_PHASES[:6]
    elif mode == "benchmark":
        expected = OPTIMIZE_PHASES[:4]
    else:
        expected = OPTIMIZE_PHASES

    missing = [p for p in expected if p not in completed]
    if missing:
        checks.append(CheckResult("progress", "all expected phases completed", "fail",
                                  f"Missing: {missing}"))
    else:
        checks.append(CheckResult("progress", "all expected phases completed", "pass",
                                  f"{len(completed)}/{len(expected)} phases"))

    for i, phase in enumerate(completed):
        if phase in expected:
            expected_idx = expected.index(phase)
            if i > 0 and completed[i - 1] in expected:
                prev_idx = expected.index(completed[i - 1])
                if expected_idx < prev_idx:
                    checks.append(CheckResult("progress", "phase order consistent", "warning",
                                              f"{completed[i-1]} before {phase} is out of order"))
                    break

    return checks


# ---------------------------------------------------------------------------
# Issue detection - generic log/artifact scanning
# ---------------------------------------------------------------------------

def _infer_phase(filepath):
    """Infer which phase doc a file relates to based on its path/name."""
    basename = os.path.basename(filepath).lower()
    dirpart = os.path.dirname(filepath).lower()

    if "profile" in dirpart or "profile" in basename:
        return "04-profile.md"
    if "benchmark" in basename or "docker_run" in basename:
        if "profile" in basename:
            return "04-profile.md"
        return "02-benchmark.md"
    if "problem" in basename or "fusion" in basename or "bottleneck" in basename:
        return "06-problem-generate.md"
    if "kernel" in basename or "geak" in basename or "_opt" in basename:
        return "07-kernel-optimize.md"
    if "plugin" in dirpart or "optimized" in dirpart:
        return "08-integration.md"
    if "report" in dirpart:
        return "09-report-generate.md"
    return "unknown"


def scan_for_issues(output_dir):
    issues = []

    # Find all log files recursively
    log_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

    for log_path in log_files:
        rel_path = os.path.relpath(log_path, output_dir)
        try:
            with open(log_path, errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    # Skip false positives
                    if any(sp.search(line_stripped) for sp in SKIP_PATTERNS):
                        continue

                    for pattern in ERROR_PATTERNS:
                        if pattern.search(line_stripped):
                            issues.append(Issue(
                                source=f"{rel_path}:{lineno}",
                                severity="error",
                                pattern=pattern.pattern,
                                context=line_stripped,
                                suggested_phase=_infer_phase(log_path),
                                analysis=_analyze_error(pattern.pattern, line_stripped),
                            ))
                            break  # one issue per line
        except (IOError, OSError):
            pass

    # Check gzip trace integrity
    for trace in glob.glob(os.path.join(output_dir, "profiles", "*.json.gz")):
        rel_path = os.path.relpath(trace, output_dir)
        try:
            with gzip.open(trace, "rb") as gz:
                while True:
                    chunk = gz.read(1024 * 1024)
                    if not chunk:
                        break
        except (gzip.BadGzipFile, EOFError, OSError) as e:
            issues.append(Issue(
                source=rel_path,
                severity="error",
                pattern="gzip truncation",
                context=str(e),
                suggested_phase="04-profile.md",
                analysis="Trace file is truncated. The profiler may not have finished "
                         "flushing before the container was stopped. Check Phase 04 "
                         "flush-wait logic.",
            ))

    # Check E2E performance
    comparison_path = os.path.join(output_dir, "results", "optimization_comparison.json")
    if os.path.isfile(comparison_path):
        with open(comparison_path) as f:
            comp = json.load(f)
        if comp.get("validated") and comp.get("speedup") is not None:
            if comp["speedup"] < 1.0:
                bl = comp.get("baseline", {}).get("total_token_throughput", "?")
                opt = comp.get("optimized", {}).get("total_token_throughput", "?")
                issues.append(Issue(
                    source="results/optimization_comparison.json",
                    severity="error",
                    pattern="E2E performance degradation",
                    context=f"speedup={comp['speedup']:.3f}x, baseline={bl}, optimized={opt}",
                    suggested_phase="08-integration.md",
                    analysis="Optimized E2E throughput is lower than baseline. Possible causes: "
                             "torch.compile/CUDAGraph masking kernel-level gains, plugin import "
                             "overhead, incorrect kernel dispatch, or regression in fused ops.",
                ))

    # Check kernel→integration leakage: regressed kernels must not be in optimized/ or plugin
    geak_path = os.path.join(output_dir, "problems", "geak_results.json")
    if os.path.isfile(geak_path):
        geak_entries = extract_entries(load_json(geak_path))
        optimized_dir = os.path.join(output_dir, "optimized")
        staged_files = {os.path.basename(f) for f in glob.glob(os.path.join(optimized_dir, "*_opt.py"))}

        # Load plugin manifest
        manifest_registered = []
        for plugin_name in ["vllm_plugin", "sglang_plugin"]:
            mp = os.path.join(optimized_dir, plugin_name, "manifest.json")
            if os.path.isfile(mp):
                with open(mp) as f:
                    mdata = json.load(f)
                manifest_registered = mdata.get("registered", [])
                break
        manifest_kernels = {r.get("kernel", "") for r in manifest_registered}

        for entry in geak_entries:
            name = entry.get("name", entry.get("kernel", ""))
            speedup = entry.get("speedup")
            if speedup is not None and speedup < 1.0:
                opt_filename = f"{name}_opt.py"
                if opt_filename in staged_files:
                    issues.append(Issue(
                        source=f"optimized/{opt_filename}",
                        severity="error",
                        pattern="regressed kernel integrated",
                        context=f"{name}: speedup={speedup:.3f}x < 1.0 but found in optimized/",
                        suggested_phase="07-kernel-optimize.md",
                        analysis=f"Kernel '{name}' has speedup {speedup:.3f}x (slower than baseline) "
                                 f"but was copied to optimized/. Phase 07 should skip kernels with "
                                 f"speedup < 1.0.",
                    ))
                if opt_filename in manifest_kernels:
                    issues.append(Issue(
                        source=f"optimized/plugin/manifest.json",
                        severity="error",
                        pattern="regressed kernel in plugin",
                        context=f"{name}: speedup={speedup:.3f}x < 1.0 but registered in plugin",
                        suggested_phase="08-integration.md",
                        analysis=f"Kernel '{name}' has speedup {speedup:.3f}x (slower than baseline) "
                                 f"but was registered in the plugin manifest. This will degrade E2E "
                                 f"performance.",
                    ))

    return issues


def _analyze_error(pattern, context):
    """Generate a brief analysis based on the error pattern."""
    ctx_lower = context.lower()
    if "oom" in ctx_lower or "out of memory" in ctx_lower:
        return "GPU out of memory. Check if stale server processes are consuming GPU memory."
    if "hip error" in ctx_lower or "cuda error" in ctx_lower:
        return "GPU runtime error. Check device visibility and driver compatibility."
    if "404" in context:
        return "HTTP 404. API route may not be mounted (e.g., vLLM profiler routes)."
    if "traceback" in ctx_lower:
        return "Python exception. Check the full traceback in the log file."
    if "killed" in ctx_lower or "sigkill" in ctx_lower:
        return "Process was killed (likely OOM killer or timeout)."
    if "runtime" in ctx_lower and "error" in ctx_lower:
        return "Runtime error in the inference framework."
    return "Error detected in log output. Review the surrounding context."


def unique_issues_by_source_pattern(issues):
    seen = set()
    unique = []
    for issue in issues:
        key = (issue.source, issue.pattern)
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(output_dir, config, checks, issues):
    config_key = config.get("CONFIG_KEY", "unknown") if config else "unknown"
    now = datetime.datetime.now().isoformat()

    passed = sum(1 for c in checks if c.status == "pass")
    failed = sum(1 for c in checks if c.status == "fail")
    skipped = sum(1 for c in checks if c.status == "skip")
    warnings = sum(1 for c in checks if c.status == "warning")

    report = {
        "config_key": config_key,
        "timestamp": now,
        "output_dir": output_dir,
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": warnings,
        "checks": [c.to_dict() for c in checks],
        "issues_count": len(issues),
        "issues": [i.to_dict() for i in issues],
    }

    # Write JSON report
    json_path = os.path.join(output_dir, "test_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write markdown report
    md_path = os.path.join(output_dir, "test_report.md")
    with open(md_path, "w") as f:
        f.write(f"# E2E Test Report: {config_key}\n\n")
        f.write(f"Generated: {now}\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Passed | {passed} |\n")
        f.write(f"| Failed | {failed} |\n")
        f.write(f"| Skipped | {skipped} |\n")
        f.write(f"| Warnings | {warnings} |\n")
        f.write(f"| Log issues | {len(issues)} |\n\n")

        f.write("## Phase Checks\n\n")
        f.write("| Phase | Check | Status | Detail |\n")
        f.write("|-------|-------|--------|--------|\n")
        for c in checks:
            status_icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP",
                           "warning": "WARN"}[c.status]
            detail = c.detail[:80] if c.detail else ""
            f.write(f"| {c.phase} | {c.name} | {status_icon} | {detail} |\n")

        if issues:
            f.write("\n## Detected Issues\n\n")
            unique_issues = unique_issues_by_source_pattern(issues)

            for i, issue in enumerate(unique_issues, 1):
                f.write(f"### Issue {i}: {issue.pattern}\n\n")
                f.write(f"- **Source**: `{issue.source}`\n")
                f.write(f"- **Severity**: {issue.severity}\n")
                f.write(f"- **Context**: `{issue.context[:150]}`\n")
                f.write(f"- **Suggested phase**: `{issue.suggested_phase}`\n")
                f.write(f"- **Analysis**: {issue.analysis}\n\n")

            f.write("## Recommendations\n\n")
            phase_issues = {}
            for issue in unique_issues:
                if issue.severity == "error":
                    phase_issues.setdefault(issue.suggested_phase, []).append(issue)

            if phase_issues:
                for phase, phase_issue_list in sorted(phase_issues.items()):
                    f.write(f"- **{phase}**: {len(phase_issue_list)} error(s) detected. "
                            f"Review and update phase doc.\n")
            else:
                f.write("No phase doc updates recommended.\n")

    return json_path, md_path, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_output_dir(output_dir):
    """Run all validations on a single output directory."""
    print(f"\nValidating: {output_dir}")
    print("-" * 60)

    config = load_config(output_dir)
    if config:
        print(f"  Config key: {config.get('CONFIG_KEY', '?')}")
        print(f"  Framework:  {config.get('FRAMEWORK', '?')}")
        print(f"  Mode:       {config.get('MODE', '?')}")
    else:
        print("  WARNING: No config.json found, using defaults")

    # Run all phase checks
    checks = []
    checks.extend(check_env(output_dir, config))
    checks.extend(check_config(output_dir, config))
    checks.extend(check_benchmark(output_dir, config))
    checks.extend(check_benchmark_analyze(output_dir, config))
    checks.extend(check_profile(output_dir, config))
    checks.extend(check_profile_analyze(output_dir, config))
    checks.extend(check_problem_generate(output_dir, config))
    checks.extend(check_kernel_optimize(output_dir, config))
    checks.extend(check_integration(output_dir, config))
    checks.extend(check_report(output_dir, config))
    checks.extend(check_progress_consistency(output_dir, config))

    # Scan for issues
    print("\nScanning logs and artifacts for issues...")
    issues = scan_for_issues(output_dir)

    # Generate reports
    json_path, md_path, report = generate_report(output_dir, config, checks, issues)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {report['passed']} passed, {report['failed']} failed, "
          f"{report['skipped']} skipped, {report['warnings']} warnings")
    if issues:
        error_issues = [i for i in issues if i.severity == "error"]
        warn_issues = [i for i in issues if i.severity == "warning"]
        unique_errors = unique_issues_by_source_pattern(error_issues)
        print(f"Log issues: {len(unique_errors)} error(s), {len(warn_issues)} warning(s)")
    print(f"{'='*60}")

    # Print per-phase status
    phases_seen = []
    for c in checks:
        if c.phase not in phases_seen:
            phases_seen.append(c.phase)

    for phase in phases_seen:
        phase_checks = [c for c in checks if c.phase == phase]
        statuses = [c.status for c in phase_checks]
        if "fail" in statuses:
            icon = "FAIL"
        elif all(s == "skip" for s in statuses):
            icon = "SKIP"
        elif "warning" in statuses:
            icon = "WARN"
        else:
            icon = "PASS"
        print(f"  [{icon}] {phase}")

    print(f"\nReports written to:")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return report["failed"] == 0


def main():
    parser = argparse.ArgumentParser(description="E2E validator for inferencex-optimize")
    parser.add_argument("--output-dir", help="Validate an existing output directory")
    parser.add_argument("--target", choices=["vllm", "sglang"],
                        help="Specify test target directly")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive target selection (default when no args)")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
        if not os.path.isdir(output_dir):
            print(f"Error: directory not found: {output_dir}")
            sys.exit(1)
        success = validate_output_dir(output_dir)
        sys.exit(0 if success else 1)

    if args.target:
        targets = [args.target]
    else:
        raw = select_target()
        if isinstance(raw, list) and raw and isinstance(raw[0], tuple):
            _, path = raw[0]
            success = validate_output_dir(os.path.expanduser(path))
            sys.exit(0 if success else 1)
        targets = raw

    all_passed = True
    for target_key in targets:
        output_dir = guide_pipeline_run(target_key)
        if not validate_output_dir(output_dir):
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
