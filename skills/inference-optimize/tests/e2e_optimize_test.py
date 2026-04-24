#!/usr/bin/env python3
"""
E2E validator for inference-optimize pipeline outputs.

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
            'claude "Use inference-optimize for gptoss-fp4-mi355x-vllm '
            'with optimize workflow, TP=8, 1k1k, conc=4"'
        ),
    },
    "sglang": {
        "config_key": "dsr1-fp4-mi355x-sglang",
        "model": "amd/DeepSeek-R1-0528-MXFP4-Preview",
        "size": "671B MoE (37B active)",
        "framework": "sglang",
        "command": (
            'claude "Use inference-optimize for dsr1-fp4-mi355x-sglang '
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
    re.compile(r"TimeoutError:"),
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
    print("(It will be something like ~/inference_{config_key}_{timestamp}/)")

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


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "report"))
from integration_outcome import derive_fields, performance_gate as _performance_gate, SEVERE_TTFT_REGRESSION_PCT


def expected_performance_gate(speedup, ttft_regression_pct=None):
    gate, _ = _performance_gate(speedup, ttft_regression_pct)
    return gate


def check_skill_layout():
    checks = []
    skill_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tarball_path = os.path.join(skill_root, "resources", "TraceLens-internal.tar.gz")
    if os.path.isfile(tarball_path):
        checks.append(CheckResult("packaging", "TraceLens fallback tarball present", "pass",
                                  tarball_path))
    else:
        checks.append(CheckResult("packaging", "TraceLens fallback tarball present", "fail",
                                  f"Missing: {tarball_path}"))

    # Multi-agent orchestration files
    orch_files = [
        "orchestrator/ORCHESTRATOR.md",
        "orchestrator/phase-registry.json",
        "orchestrator/monitor.md",
    ]
    for f in orch_files:
        path = os.path.join(skill_root, f)
        if os.path.isfile(path):
            checks.append(CheckResult("packaging", f"{f} present", "pass", path))
        else:
            checks.append(CheckResult("packaging", f"{f} present", "fail", f"Missing: {path}"))

    # Agent files
    expected_agents = [
        f"agents/phase-{i:02d}-{name}.md"
        for i, name in enumerate([
            "env-setup", "config-parse", "benchmark", "benchmark-analyze",
            "profile", "profile-analyze", "problem-generate",
            "kernel-optimize", "integration", "report-generate",
        ])
    ] + ["agents/coding-agent.md", "agents/analysis-agent.md"]
    for f in expected_agents:
        path = os.path.join(skill_root, f)
        if os.path.isfile(path):
            checks.append(CheckResult("packaging", f"{f} present", "pass"))
        else:
            checks.append(CheckResult("packaging", f"{f} present", "fail", f"Missing: {path}"))

    # Protocol files
    protocol_files = [
        "protocols/phase-result.schema.md",
        "protocols/monitor-feedback.schema.md",
        "protocols/handoff-format.md",
        "protocols/rerun-protocol.md",
        "protocols/analyzer-manifest.schema.md",
    ]
    for f in protocol_files:
        path = os.path.join(skill_root, f)
        if os.path.isfile(path):
            checks.append(CheckResult("packaging", f"{f} present", "pass"))
        else:
            checks.append(CheckResult("packaging", f"{f} present", "fail", f"Missing: {path}"))

    return checks


def check_multi_agent_workspace(output_dir, config):
    """Validate multi-agent communication workspace directories and files."""
    checks = []

    # Check workspace directories exist
    for dirname in ["handoff", "agent-results", "monitor"]:
        dirpath = os.path.join(output_dir, dirname)
        if os.path.isdir(dirpath):
            checks.append(CheckResult("multi-agent", f"{dirname}/ directory exists", "pass"))
        else:
            checks.append(CheckResult("multi-agent", f"{dirname}/ directory exists", "skip",
                                      "Not present (single-agent mode or not yet created)"))

    # Check agent-results files exist for completed phases
    progress_path = os.path.join(output_dir, "progress.json")
    results_dir = os.path.join(output_dir, "agent-results")
    phase_index_map = {
        "env": 0, "config": 1, "benchmark": 2, "benchmark-analyze": 3,
        "profile": 4, "profile-analyze": 5, "problem-generate": 6,
        "kernel-optimize": 7, "integration": 8, "report-generate": 9,
    }
    completed = []

    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        completed = progress.get("phases_completed", [])

        if os.path.isdir(results_dir):
            for phase_key in completed:
                idx = phase_index_map.get(phase_key)
                if idx is not None:
                    result_file = os.path.join(results_dir, f"phase-{idx:02d}-result.md")
                    if os.path.isfile(result_file):
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-result.md exists", "pass"))
                    else:
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-result.md exists", "skip",
                                                   "Result file not found"))

    # Check monitor running-summary exists
    summary_path = os.path.join(output_dir, "monitor", "running-summary.md")
    if os.path.isfile(summary_path):
        checks.append(CheckResult("multi-agent", "running-summary.md exists", "pass"))
    else:
        checks.append(CheckResult("multi-agent", "running-summary.md exists", "skip",
                                  "Not present (single-agent mode)"))

    # Check monitor reviews for critical phases
    monitor_dir = os.path.join(output_dir, "monitor")
    if os.path.isdir(monitor_dir):
        critical_phases = {2: "benchmark", 5: "profile-analyze",
                           7: "kernel-optimize", 8: "integration"}
        for idx, name in critical_phases.items():
            review_file = os.path.join(monitor_dir, f"phase-{idx:02d}-review.md")
            if os.path.isfile(review_file):
                checks.append(CheckResult("multi-agent",
                                           f"phase-{idx:02d}-review.md (critical) exists", "pass"))

    # Validate progress.json uses canonical phase keys
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        skill_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        registry_path = os.path.join(skill_root, "orchestrator", "phase-registry.json")
        rerun_limits = {"max_per_phase": 0, "max_total": 0}
        if os.path.isfile(registry_path):
            with open(registry_path) as f:
                rerun_limits = json.load(f).get("rerun", rerun_limits)
        max_per_phase = rerun_limits.get("max_per_phase", 0)
        max_total = rerun_limits.get("max_total", 0)
        completed = progress.get("phases_completed", [])
        valid_keys = set(OPTIMIZE_PHASES)
        invalid = [k for k in completed if k not in valid_keys]
        if invalid:
            checks.append(CheckResult("multi-agent", "progress.json uses canonical phase keys",
                                       "fail", f"Invalid keys: {invalid}"))
        elif completed:
            checks.append(CheckResult("multi-agent", "progress.json uses canonical phase keys",
                                       "pass", f"{len(completed)} phases completed"))

        # Validate retry_counts structure
        retry_counts = progress.get("retry_counts", {})
        if isinstance(retry_counts, dict):
            bad_keys = [k for k in retry_counts if k not in valid_keys]
            if bad_keys:
                checks.append(CheckResult("multi-agent", "retry_counts uses canonical keys",
                                           "fail", f"Invalid keys: {bad_keys}"))
            else:
                checks.append(CheckResult("multi-agent", "retry_counts uses canonical keys",
                                           "pass"))
            if max_per_phase > 0:
                for k, v in retry_counts.items():
                    if isinstance(v, int) and v > max_per_phase:
                        checks.append(CheckResult(
                            "multi-agent",
                            f"retry_counts[{k}] within limit",
                            "fail",
                            f"Retried {v} times (max_per_phase={max_per_phase})",
                        ))

        total_reruns = progress.get("total_reruns", 0)
        if max_total > 0 and isinstance(total_reruns, int) and total_reruns > max_total:
            checks.append(CheckResult("multi-agent", "total_reruns within limit",
                                       "fail", f"Total reruns={total_reruns} (max={max_total})"))

        # Validate fallbacks_used structure
        fallbacks_used = progress.get("fallbacks_used", [])
        if isinstance(fallbacks_used, list):
            for fb in fallbacks_used:
                if not isinstance(fb, dict) or "phase_key" not in fb or "fallback_target" not in fb:
                    checks.append(CheckResult("multi-agent", "fallbacks_used entry valid",
                                               "fail", f"Invalid entry: {fb}"))
                    break
            else:
                checks.append(CheckResult("multi-agent", "fallbacks_used structure valid", "pass",
                                           f"{len(fallbacks_used)} entries"))

        # Validate current_phase
        current_phase = progress.get("current_phase")
        if current_phase is not None:
            if current_phase in valid_keys:
                checks.append(CheckResult("multi-agent", "current_phase is canonical key", "pass",
                                           current_phase))
            else:
                checks.append(CheckResult("multi-agent", "current_phase is canonical key", "fail",
                                           f"Invalid: {current_phase}"))

        # Validate status enum
        status = progress.get("status")
        if status is not None:
            if status in ("running", "completed", "failed"):
                checks.append(CheckResult("multi-agent", "progress.json status valid", "pass",
                                           status))
            else:
                checks.append(CheckResult("multi-agent", "progress.json status valid", "fail",
                                           f"Invalid status: {status}"))

    # Validate handoff files match completed phases
    handoff_dir = os.path.join(output_dir, "handoff")
    rca_phase_map = {
        "benchmark": "results/benchmark_rca.json",
        "profile-analyze": "results/profile_rca.json",
        "problem-generate": "results/problem_gen_rca.json",
        "kernel-optimize": "results/kernel_opt_rca.json",
        "integration": "results/integration_rca.json",
    }
    if os.path.isdir(handoff_dir) and os.path.isfile(progress_path):
        for phase_key in completed:
            idx = phase_index_map.get(phase_key)
            if idx is not None:
                handoff_file = os.path.join(handoff_dir, f"to-phase-{idx:02d}.md")
                if os.path.isfile(handoff_file):
                    checks.append(CheckResult("multi-agent",
                                               f"handoff/to-phase-{idx:02d}.md exists", "pass"))
                    with open(handoff_file) as f:
                        content = f.read()
                    if "## Resolved Variables" not in content:
                        checks.append(CheckResult("multi-agent",
                                                   f"handoff phase-{idx:02d} has Resolved Variables",
                                                   "warning", "Missing ## Resolved Variables section"))
                    # Check for ## Root Cause Analysis when RCA artifact exists
                    rca_file = rca_phase_map.get(phase_key)
                    if rca_file:
                        rca_path = os.path.join(output_dir, rca_file)
                        if os.path.isfile(rca_path) and "## Root Cause Analysis" in content:
                            checks.append(CheckResult("multi-agent",
                                                       f"handoff phase-{idx:02d} has RCA section", "pass"))
                        elif os.path.isfile(rca_path) and "## Root Cause Analysis" not in content:
                            checks.append(CheckResult("multi-agent",
                                                       f"handoff phase-{idx:02d} has RCA section", "warning",
                                                       "RCA artifact exists but handoff missing ## Root Cause Analysis"))
                else:
                    checks.append(CheckResult("multi-agent",
                                               f"handoff/to-phase-{idx:02d}.md exists", "skip",
                                               "Handoff not found"))

    # Validate result doc format (YAML frontmatter with required fields)
    if os.path.isdir(results_dir):
        for phase_key in completed:
            idx = phase_index_map.get(phase_key)
            if idx is None:
                continue
            result_file = os.path.join(results_dir, f"phase-{idx:02d}-result.md")
            if not os.path.isfile(result_file):
                continue
            with open(result_file) as f:
                content = f.read()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    has_phase = "phase:" in frontmatter
                    has_status = "status:" in frontmatter
                    has_phase_index = "phase_index:" in frontmatter
                    has_timestamp = "timestamp:" in frontmatter
                    required_fields = {"phase": has_phase, "status": has_status,
                                       "phase_index": has_phase_index, "timestamp": has_timestamp}
                    missing = [k for k, v in required_fields.items() if not v]
                    if not missing:
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-result.md has valid frontmatter",
                                                   "pass"))
                    else:
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-result.md has valid frontmatter",
                                                   "warning", f"Missing fields: {missing}"))

                    # Validate status enum
                    status_match = re.search(r"status:\s*(\S+)", frontmatter)
                    if status_match:
                        status_val = status_match.group(1)
                        if status_val not in ("completed", "failed", "partial"):
                            checks.append(CheckResult("multi-agent",
                                                       f"phase-{idx:02d}-result.md status enum",
                                                       "warning",
                                                       f"Invalid status '{status_val}', expected completed|failed|partial"))

    # Validate monitor review format for critical phases
    if os.path.isdir(monitor_dir):
        for idx, name in critical_phases.items():
            review_file = os.path.join(monitor_dir, f"phase-{idx:02d}-review.md")
            if not os.path.isfile(review_file):
                continue
            with open(review_file) as f:
                content = f.read()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    has_verdict = "verdict:" in frontmatter
                    if has_verdict:
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-review.md has verdict",
                                                   "pass"))
                        # Non-PASS verdicts must include failure_type
                        verdict_match = re.search(r"verdict:\s*(\S+)", frontmatter)
                        if verdict_match:
                            verdict_value = verdict_match.group(1).upper()
                            if verdict_value == "WARN":
                                checks.append(CheckResult(
                                    "multi-agent",
                                    f"phase-{idx:02d}-review.md verdict enum",
                                    "fail",
                                    "WARN verdict is no longer allowed; use FAIL",
                                ))
                            if verdict_value == "FAIL" and "failure_type:" not in frontmatter:
                                checks.append(CheckResult("multi-agent",
                                                           f"phase-{idx:02d}-review.md FAIL has failure_type",
                                                           "warning",
                                                           "FAIL verdict missing failure_type field"))
                    else:
                        checks.append(CheckResult("multi-agent",
                                                   f"phase-{idx:02d}-review.md has verdict",
                                                   "warning", "Missing verdict in frontmatter"))

    return checks


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


def _check_trace_manifest(output_dir):
    """Validate trace_manifest.json independently of gap_analysis.json."""
    checks = []
    manifest_path = os.path.join(output_dir, "results", "trace_manifest.json")
    if not os.path.isfile(manifest_path):
        checks.append(CheckResult("profile-analyze", "trace_manifest.json exists", "fail",
                                  "Missing structured trace manifest"))
        return checks

    try:
        manifest = load_json(manifest_path)
    except (json.JSONDecodeError, IOError) as exc:
        checks.append(CheckResult("profile-analyze", "trace_manifest.json readable", "fail",
                                  str(exc)))
        return checks

    required_fields = {
        "trace_count", "world_size", "traces",
        "tracelens_primary_trace", "phase_split_inputs_ready",
    }
    missing = required_fields - set(manifest.keys())
    if missing:
        checks.append(CheckResult("profile-analyze", "trace_manifest.json schema valid", "fail",
                                  f"Missing: {sorted(missing)}"))
    else:
        checks.append(CheckResult("profile-analyze", "trace_manifest.json schema valid", "pass"))

    if "schema_version" in manifest:
        checks.append(CheckResult("profile-analyze", "trace_manifest has schema_version", "pass",
                                  manifest["schema_version"]))

    traces = manifest.get("traces")
    if isinstance(traces, list):
        checks.append(CheckResult("profile-analyze", "trace_manifest traces list", "pass",
                                  f"{len(traces)} trace(s)"))
        expected_trace_count = manifest.get("trace_count")
        valid_trace_count = sum(
            1
            for trace in traces
            if isinstance(trace, dict) and trace.get("integrity") == "valid"
        )
        if expected_trace_count == valid_trace_count:
            checks.append(CheckResult("profile-analyze", "trace_manifest trace_count matches valid traces", "pass",
                                      f"trace_count={expected_trace_count}, valid_traces={valid_trace_count}"))
        else:
            checks.append(CheckResult("profile-analyze", "trace_manifest trace_count matches valid traces", "fail",
                                      f"trace_count={expected_trace_count}, valid_traces={valid_trace_count}, len(traces)={len(traces)}"))

        invalid_entries = []
        invalid_integrity = []
        invalid_roles = []
        for idx, trace in enumerate(traces):
            if not isinstance(trace, dict):
                invalid_entries.append(idx)
                continue
            missing_keys = {"path", "size_bytes", "integrity", "role"} - set(trace.keys())
            if missing_keys:
                invalid_entries.append(f"{idx} missing {sorted(missing_keys)}")
            if trace.get("integrity") not in {"valid", "corrupt", "missing"}:
                invalid_integrity.append(f"{idx}:{trace.get('integrity')}")
            if trace.get("role") not in {"primary", "secondary", "unknown"}:
                invalid_roles.append(f"{idx}:{trace.get('role')}")

        if invalid_entries:
            checks.append(CheckResult("profile-analyze", "trace_manifest entries valid", "fail",
                                      f"Invalid entries: {invalid_entries}"))
        else:
            checks.append(CheckResult("profile-analyze", "trace_manifest entries valid", "pass"))

        if invalid_integrity:
            checks.append(CheckResult("profile-analyze", "trace_manifest integrity enum valid", "fail",
                                      ", ".join(invalid_integrity)))
        else:
            checks.append(CheckResult("profile-analyze", "trace_manifest integrity enum valid", "pass"))

        if invalid_roles:
            checks.append(CheckResult("profile-analyze", "trace_manifest role enum valid", "fail",
                                      ", ".join(invalid_roles)))
        else:
            checks.append(CheckResult("profile-analyze", "trace_manifest role enum valid", "pass"))
    else:
        checks.append(CheckResult("profile-analyze", "trace_manifest traces list", "fail",
                                  f"Expected list, got {type(traces).__name__}"))

    if isinstance(manifest.get("phase_split_inputs_ready"), bool):
        checks.append(CheckResult("profile-analyze", "trace_manifest phase_split_inputs_ready bool", "pass",
                                  str(manifest["phase_split_inputs_ready"])))
    else:
        checks.append(CheckResult("profile-analyze", "trace_manifest phase_split_inputs_ready bool", "fail",
                                  f"Got {type(manifest.get('phase_split_inputs_ready')).__name__}"))

    return checks


def check_profile_analyze(output_dir, config):
    checks = []
    mode = config.get("MODE", "optimize") if config else "optimize"
    profile_expected = mode in {"profile", "full", "optimize"}
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.isfile(progress_path):
        try:
            with open(progress_path) as f:
                completed = json.load(f).get("phases_completed", [])
            profile_expected = profile_expected or ("profile-analyze" in completed)
        except (json.JSONDecodeError, IOError):
            pass

    if profile_expected:
        checks.extend(_check_trace_manifest(output_dir))
    else:
        checks.append(CheckResult("profile-analyze", "trace_manifest.json exists", "skip",
                                  f"profile-analyze not expected for mode={mode}"))

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
    """Under the library-rebuild contract, Phase 6 (`problem-generate` /
    `upstream-resolve`) emits `problems/optimization_manifest.json`
    (schema 2.0 — kernels keyed by upstream symbol with library /
    geak_strategy / fork_path), `forks/manifest.json` (per-library fork
    state), and `results/baseline_dispatch_trace.json` (rocprofv3 capture).
    No more `problem_*.py` files (the synthetic-harness contract is gone).
    """
    checks = []
    problems_dir = os.path.join(output_dir, "problems")
    forks_dir = os.path.join(output_dir, "forks")
    results_dir = os.path.join(output_dir, "results")

    manifest_path = os.path.join(problems_dir, "optimization_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = manifest.get("optimizations", manifest.get("entries", manifest.get("kernels", [])))
        checks.append(CheckResult("problem-generate", "optimization_manifest.json valid", "pass",
                                  f"{len(entries)} entries"))
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("kernel") or f"entry[{i}]"
            missing_fields = [
                f for f in ("library", "geak_strategy")
                if f not in entry
            ]
            if missing_fields:
                checks.append(CheckResult("problem-generate",
                                          f"{name} library-rebuild contract", "warning",
                                          f"Missing fields: {missing_fields}"))
            else:
                checks.append(CheckResult("problem-generate",
                                          f"{name} library-rebuild contract", "pass",
                                          f"library={entry.get('library')}, "
                                          f"strategy={entry.get('geak_strategy')}"))
    else:
        checks.append(CheckResult("problem-generate", "optimization_manifest.json exists", "skip"))

    forks_manifest_path = os.path.join(forks_dir, "manifest.json")
    if os.path.isfile(forks_manifest_path):
        with open(forks_manifest_path) as f:
            forks_manifest = json.load(f)
        fork_entries = forks_manifest.get("forks", forks_manifest)
        if isinstance(fork_entries, dict) and not isinstance(fork_entries, list):
            fork_count = len([k for k in fork_entries.keys() if k != "ck_branch_merged_status"])
        else:
            fork_count = len(fork_entries) if isinstance(fork_entries, list) else 0
        checks.append(CheckResult("problem-generate", "forks/manifest.json valid", "pass",
                                  f"{fork_count} fork(s) pinned"))
    else:
        checks.append(CheckResult("problem-generate", "forks/manifest.json exists", "skip"))

    baseline_trace = os.path.join(results_dir, "baseline_dispatch_trace.json")
    if os.path.isfile(baseline_trace):
        checks.append(CheckResult("problem-generate", "baseline_dispatch_trace.json captured",
                                  "pass"))
    else:
        checks.append(CheckResult("problem-generate", "baseline_dispatch_trace.json exists",
                                  "skip"))

    return checks


def check_kernel_optimize(output_dir, config):
    """Under the library-rebuild contract, Phase 7 commits GEAK winners
    onto the per-library `geak/` branch in `forks/<lib>/` and writes
    `geak_results.json` with `{geak_strategy, fork_commit_after_winner,
    library_test_pass_count, allocator_test_pass, dispatch_pre_flight_pass,
    geak_speedup_lib_bench}` per kernel. There are no `*_opt.py` files
    and no `optimized/` staging directory."""
    checks = []
    problems_dir = os.path.join(output_dir, "problems")
    forks_dir = os.path.join(output_dir, "forks")
    results_dir = os.path.join(output_dir, "results")

    geak_path = os.path.join(problems_dir, "geak_results.json")
    if os.path.isfile(geak_path):
        entries = extract_entries(load_json(geak_path))

        checks.append(CheckResult("kernel-optimize", "geak_results.json exists", "pass",
                                  f"{len(entries)} entries"))

        for entry in entries:
            name = entry.get("name", entry.get("kernel", "unknown"))
            strategy = entry.get("geak_strategy", "")
            speedup = entry.get("geak_speedup_lib_bench", entry.get("speedup"))
            preflight_pass = entry.get("dispatch_pre_flight_pass")
            allocator_pass = entry.get("allocator_test_pass")
            unverified = entry.get("optimization_unverified_per_kernel")

            # Speedup reporting (when present)
            if speedup is not None:
                status = "pass" if speedup >= 1.0 else "warning"
                detail = f"strategy={strategy}, speedup={speedup}x"
                if unverified:
                    detail += " [unverified-per-kernel: Bucket B]"
                checks.append(CheckResult("kernel-optimize", f"{name} result", status, detail))
            else:
                checks.append(CheckResult("kernel-optimize", f"{name} result", "warning",
                                          f"strategy={strategy}, speedup=null"))

            # Dispatch pre-flight gate is mandatory for all buckets
            if preflight_pass is False:
                checks.append(CheckResult("kernel-optimize",
                                          f"{name} dispatch pre-flight", "fail",
                                          "dispatch_pre_flight_pass=false"))
            elif preflight_pass is True:
                checks.append(CheckResult("kernel-optimize",
                                          f"{name} dispatch pre-flight", "pass"))

            # Allocator test only required for Bucket A (in_place_optimize, redirects)
            if strategy == "in_place_optimize" and allocator_pass is False:
                checks.append(CheckResult("kernel-optimize",
                                          f"{name} allocator integration", "fail",
                                          "allocator_test_pass=false"))
    else:
        checks.append(CheckResult("kernel-optimize", "geak_results.json exists", "skip"))

    # forks/ tree must exist when problem-generate ran
    if os.path.isdir(forks_dir):
        fork_subdirs = [d for d in os.listdir(forks_dir)
                        if os.path.isdir(os.path.join(forks_dir, d))]
        checks.append(CheckResult("kernel-optimize", "forks/ directory populated",
                                  "pass" if fork_subdirs else "warning",
                                  f"{len(fork_subdirs)} fork(s)"))
    else:
        checks.append(CheckResult("kernel-optimize", "forks/ directory exists", "skip"))

    preflight_trace = os.path.join(results_dir, "preflight_dispatch_trace.json")
    if os.path.isfile(preflight_trace):
        checks.append(CheckResult("kernel-optimize", "preflight_dispatch_trace.json captured",
                                  "pass"))
    else:
        checks.append(CheckResult("kernel-optimize", "preflight_dispatch_trace.json exists",
                                  "skip"))

    return checks


def check_integration(output_dir, config):
    """Under the library-rebuild contract, Phase 8 emits
    `results/dispatch_verification.json` (rocprofv3 expected/vendor symbol
    counts), `results/integration_manifest.json` (schema 2.0 with
    `libraries_rebuilt`, `dispatch_verified`, `e2e_ran`,
    `artifacts_valid`), and the per-library `results/rebuild_<lib>.log`.
    No plugin manifests, no `optimized/<framework>_plugin/`."""
    checks = []
    results_dir = os.path.join(output_dir, "results")

    # dispatch_verification.json
    disp_path = os.path.join(results_dir, "dispatch_verification.json")
    if os.path.isfile(disp_path):
        with open(disp_path) as f:
            disp = json.load(f)
        verified = bool(disp.get("dispatch_verified"))
        checks.append(CheckResult("integration", "dispatch_verified",
                                  "pass" if verified else "fail",
                                  f"dispatch_verified={verified}"))
        leaked = disp.get("vendor_symbol_leaked_count")
        if leaked is not None:
            checks.append(CheckResult("integration", "no vendor symbol leakage",
                                      "pass" if leaked == 0 else "fail",
                                      f"vendor_symbol_leaked_count={leaked}"))
        rh = disp.get("redirect_honored_count")
        rr = disp.get("redirect_required_count")
        if rh is not None and rr is not None:
            checks.append(CheckResult("integration", "redirects honored",
                                      "pass" if rh >= rr else "fail",
                                      f"redirect_honored={rh}/{rr}"))
    else:
        checks.append(CheckResult("integration", "dispatch_verification.json exists", "skip"))

    # integration_manifest.json (schema 2.0 — libraries_rebuilt)
    manifest_path = os.path.join(results_dir, "integration_manifest.json")
    libraries_rebuilt = []
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        libraries_rebuilt = manifest.get("libraries_rebuilt", []) or []
        checks.append(CheckResult("integration", "libraries_rebuilt list valid",
                                  "pass" if isinstance(libraries_rebuilt, list) else "fail",
                                  f"{len(libraries_rebuilt)} library/libraries rebuilt"))
        for lib_entry in libraries_rebuilt:
            if not isinstance(lib_entry, dict):
                continue
            lib = lib_entry.get("lib", "?")
            log_rel = lib_entry.get("install_log_path", f"results/rebuild_{lib}.log")
            log_abs = os.path.join(output_dir, log_rel)
            if not os.path.isabs(log_abs):
                log_abs = os.path.join(output_dir, log_rel)
            present = os.path.isfile(log_abs)
            checks.append(CheckResult("integration", f"rebuild log for {lib}",
                                      "pass" if present else "fail",
                                      f"{log_rel} {'found' if present else 'missing'}"))
    else:
        checks.append(CheckResult("integration", "integration_manifest.json exists", "skip"))

    comparison_path = os.path.join(output_dir, "results", "optimization_comparison.json")
    if os.path.isfile(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)

        validated = comparison.get("validated", False)
        speedup = comparison.get("speedup")
        artifacts_valid = comparison.get("artifacts_valid")
        performance_valid = comparison.get("performance_valid")
        performance_gate = comparison.get("performance_gate")
        e2e_speedup = comparison.get("e2e_speedup")
        ttft_regression_pct = comparison.get("ttft_regression_pct")
        expected_gate = expected_performance_gate(speedup, ttft_regression_pct)

        # Check tri-state validation fields
        if "artifacts_valid" not in comparison:
            checks.append(CheckResult("integration", "artifacts_valid field present", "fail"))
        elif artifacts_valid:
            checks.append(CheckResult("integration", "artifacts_valid is true", "pass"))
        else:
            checks.append(CheckResult("integration", "artifacts_valid is true", "fail",
                                      "Baseline or optimized JSON failed to load"))

        if "performance_gate" not in comparison:
            checks.append(CheckResult("integration", "performance_gate field present", "fail"))
        elif speedup is None:
            checks.append(CheckResult("integration", "performance_gate matches speedup band", "fail",
                                      "speedup missing"))
        elif performance_gate == expected_gate:
            checks.append(CheckResult("integration", "performance_gate matches speedup band", "pass",
                                      f"speedup={speedup:.3f}x, gate={performance_gate}"))
        else:
            checks.append(CheckResult("integration", "performance_gate matches speedup band", "fail",
                                      f"speedup={speedup}, expected={expected_gate}, got={performance_gate}"))

        if "e2e_speedup" not in comparison:
            checks.append(CheckResult("integration", "e2e_speedup field present", "fail"))
        elif speedup is None:
            checks.append(CheckResult("integration", "e2e_speedup matches speedup", "fail",
                                      f"e2e_speedup={e2e_speedup}, speedup missing"))
        elif abs(e2e_speedup - speedup) < 1e-9:
            checks.append(CheckResult("integration", "e2e_speedup matches speedup", "pass",
                                      f"{e2e_speedup:.3f}x"))
        else:
            checks.append(CheckResult("integration", "e2e_speedup matches speedup", "fail",
                                      f"e2e_speedup={e2e_speedup}, speedup={speedup}"))

        if "performance_valid" not in comparison:
            checks.append(CheckResult("integration", "performance_valid field present", "fail"))
        elif expected_gate is None:
            checks.append(CheckResult("integration", "performance_valid matches gate", "fail",
                                      f"performance_valid={performance_valid}, speedup={speedup}"))
        else:
            expected_performance_valid = expected_gate == "pass"
            if performance_valid == expected_performance_valid:
                checks.append(CheckResult("integration", "performance_valid matches gate", "pass",
                                          f"performance_valid={performance_valid}, gate={expected_gate}"))
            else:
                checks.append(CheckResult("integration", "performance_valid matches gate", "fail",
                                          f"performance_valid={performance_valid}, expected={expected_performance_valid}"))

        if "validated" not in comparison:
            checks.append(CheckResult("integration", "validated field present", "fail"))
        elif expected_gate is None or artifacts_valid is None:
            checks.append(CheckResult("integration", "validated matches outcome", "fail",
                                      f"validated={validated}, speedup={speedup}, artifacts_valid={artifacts_valid}"))
        else:
            expected_validated = bool(artifacts_valid and expected_gate == "pass")
            if validated == expected_validated:
                checks.append(CheckResult("integration", "validated matches outcome", "pass",
                                          f"validated={validated}, gate={expected_gate}"))
            else:
                checks.append(CheckResult("integration", "validated matches outcome", "fail",
                                          f"validated={validated}, expected={expected_validated}"))

        gate_for_outcome = performance_gate or expected_gate
        if gate_for_outcome == "pass" and speedup is not None:
            checks.append(CheckResult("integration", "optimization_comparison outcome", "pass",
                                      f"speedup={speedup:.3f}x"))
        elif gate_for_outcome == "warn" and speedup is not None:
            checks.append(CheckResult("integration", "optimization_comparison outcome", "warning",
                                      f"speedup={speedup:.3f}x (accepted 0.97-1.0 warn band)"))
        elif speedup is not None:
            bl = comparison.get("baseline", {}).get("total_token_throughput", "?")
            opt = comparison.get("optimized", {}).get("total_token_throughput", "?")
            checks.append(CheckResult("integration", "optimization_comparison outcome", "fail",
                                      f"speedup={speedup:.3f}x (baseline={bl}, optimized={opt})"))

        # Check TTFT regression using shared threshold
        ttft_pct = comparison.get("ttft_regression_pct")
        if ttft_pct is not None and ttft_pct > SEVERE_TTFT_REGRESSION_PCT:
            checks.append(CheckResult("integration", "TTFT regression acceptable", "warning",
                                      f"ttft_regression_pct={ttft_pct:.1f}% (threshold={SEVERE_TTFT_REGRESSION_PCT}%)"))
    else:
        checks.append(CheckResult("integration", "optimization_comparison.json exists", "skip"))

    return checks


def check_integration_manifest(output_dir, config):
    """Validate results/integration_manifest.json — required when integration completes."""
    checks = []
    manifest_path = os.path.join(output_dir, "results", "integration_manifest.json")
    if not os.path.isfile(manifest_path):
        progress_path = os.path.join(output_dir, "progress.json")
        completed = []
        if os.path.isfile(progress_path):
            with open(progress_path) as f:
                completed = json.load(f).get("phases_completed", [])
        if "integration" in completed:
            checks.append(CheckResult("integration", "integration_manifest.json exists", "fail",
                                      "Integration completed but manifest missing — hard contract"))
        return checks

    try:
        manifest = load_json(manifest_path)
    except (json.JSONDecodeError, IOError) as exc:
        checks.append(CheckResult("integration", "integration_manifest.json readable", "fail",
                                  str(exc)))
        return checks

    # Schema 2.0 (library-rebuild): libraries_rebuilt + dispatch_verified
    # supersede plugin_type. Older schema with `targets`/`summary` is
    # accepted as a transitional shape but warned.
    schema_v2_required = {"libraries_rebuilt", "dispatch_verified", "e2e_ran",
                          "artifacts_valid"}
    missing_v2 = schema_v2_required - set(manifest.keys())
    legacy_required = {"schema_version", "targets", "comparison_file", "summary"}
    missing_legacy = legacy_required - set(manifest.keys())
    if not missing_v2:
        checks.append(CheckResult("integration", "integration_manifest.json schema valid", "pass",
                                  "schema 2.0 (library-rebuild)"))
    elif not missing_legacy:
        checks.append(CheckResult("integration", "integration_manifest.json schema valid", "warning",
                                  "legacy schema (pre library-rebuild contract)"))
    else:
        checks.append(CheckResult("integration", "integration_manifest.json schema valid", "fail",
                                  f"Missing v2 fields: {sorted(missing_v2)}"))

    # Schema 2.0: validate libraries_rebuilt entries (per-library
    # rebuild status with commit + install log path).
    libs = manifest.get("libraries_rebuilt")
    if isinstance(libs, list):
        checks.append(CheckResult("integration", "integration_manifest libraries_rebuilt list",
                                  "pass", f"{len(libs)} library/libraries"))
        for i, entry in enumerate(libs):
            if not isinstance(entry, dict):
                checks.append(CheckResult("integration",
                                          f"integration_manifest libraries_rebuilt[{i}] valid",
                                          "fail", f"Expected dict, got {type(entry).__name__}"))
                continue
            required = {"lib", "commit", "install_log_path"}
            missing = required - set(entry.keys())
            if missing:
                checks.append(CheckResult("integration",
                                          f"integration_manifest libraries_rebuilt[{i}] schema",
                                          "fail", f"Missing: {sorted(missing)}"))
            else:
                checks.append(CheckResult("integration",
                                          f"integration_manifest libraries_rebuilt[{i}] schema",
                                          "pass",
                                          f"lib={entry['lib']} commit={entry['commit'][:7] if isinstance(entry.get('commit'), str) else '?'}"))
    elif libs is not None:
        # Legacy schema may not have this list; surface as warning rather than fail
        checks.append(CheckResult("integration", "integration_manifest libraries_rebuilt list",
                                  "fail", f"Expected list, got {type(libs).__name__}"))

    return checks


def check_phase08_result_scalars(output_dir, config):
    """Validate Phase 08 result-doc scalar fields used by monitoring and reporting."""
    checks = []
    result_path = os.path.join(output_dir, "agent-results", "phase-08-result.md")
    if not os.path.isfile(result_path):
        progress_path = os.path.join(output_dir, "progress.json")
        completed = []
        if os.path.isfile(progress_path):
            with open(progress_path) as f:
                completed = json.load(f).get("phases_completed", [])
        if "integration" in completed:
            checks.append(CheckResult("integration", "phase-08-result.md exists", "fail",
                                      "Integration completed but result doc missing"))
        return checks

    with open(result_path) as f:
        content = f.read()

    key_findings_match = re.search(r"## Key Findings\s*\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
    if not key_findings_match:
        checks.append(CheckResult("integration", "phase-08-result.md has Key Findings section",
                                  "fail", "Missing ## Key Findings"))
        return checks

    findings_text = key_findings_match.group(1)

    required_scalars = {
        "baseline_file": r"baseline_file:\s*(\S+)",
        "optimized_file": r"optimized_file:\s*(\S+)",
        "validation_status": r"validation_status:\s*(pass|warn|fail)",
        "coverage_pct": r"coverage_pct:\s*([\d.]+)",
        "blocked_target_count": r"blocked_target_count:\s*(\d+)",
        "critical_blocker_count": r"critical_blocker_count:\s*(\d+)",
    }

    for field_name, pattern in required_scalars.items():
        match = re.search(pattern, findings_text)
        if match:
            checks.append(CheckResult("integration",
                                      f"phase-08-result scalar: {field_name}", "pass",
                                      match.group(1)))
        else:
            checks.append(CheckResult("integration",
                                      f"phase-08-result scalar: {field_name}", "fail",
                                      f"Missing or malformed {field_name} in Key Findings"))

    validation_status_match = re.search(r"validation_status:\s*(\S+)", findings_text)
    comparison_path = os.path.join(output_dir, "results", "optimization_comparison.json")
    if validation_status_match and os.path.isfile(comparison_path):
        doc_status = validation_status_match.group(1)
        try:
            with open(comparison_path) as f:
                comp = json.load(f)
            comp_gate = comp.get("performance_gate")
            if doc_status == comp_gate:
                checks.append(CheckResult("integration",
                                          "phase-08 validation_status matches comparison gate",
                                          "pass", f"{doc_status} == {comp_gate}"))
            else:
                checks.append(CheckResult("integration",
                                          "phase-08 validation_status matches comparison gate",
                                          "fail", f"doc={doc_status}, comparison={comp_gate}"))
        except (json.JSONDecodeError, IOError):
            pass

    return checks


def check_rca_artifacts(output_dir, config):
    """Verify *_rca.json files exist when the corresponding phase had reruns."""
    checks = []
    progress_path = os.path.join(output_dir, "progress.json")
    if not os.path.isfile(progress_path):
        return checks

    with open(progress_path) as f:
        progress = json.load(f)

    retry_counts = progress.get("retry_counts", {})
    results_dir = os.path.join(output_dir, "results")

    rca_map = {
        "benchmark": "benchmark_rca.json",
        "profile-analyze": "profile_rca.json",
        "problem-generate": "problem_gen_rca.json",
        "kernel-optimize": "kernel_opt_rca.json",
        "integration": "integration_rca.json",
    }

    for phase_key, rca_file in rca_map.items():
        retries = retry_counts.get(phase_key, 0)
        rca_path = os.path.join(results_dir, rca_file)
        if retries > 0:
            if os.path.isfile(rca_path):
                try:
                    with open(rca_path) as f:
                        rca = json.load(f)
                    required_fields = {"phase", "failure_type", "summary", "terminal_action"}
                    missing = required_fields - set(rca.keys())
                    if missing:
                        checks.append(CheckResult("rca", f"{rca_file} schema valid", "warning",
                                                  f"Missing fields: {missing}"))
                    else:
                        checks.append(CheckResult("rca", f"{rca_file} exists and valid", "pass",
                                                  f"terminal_action={rca.get('terminal_action')}"))
                except (json.JSONDecodeError, IOError) as e:
                    checks.append(CheckResult("rca", f"{rca_file} readable", "fail", str(e)))
            else:
                checks.append(CheckResult("rca", f"{rca_file} exists for retried phase", "warning",
                                          f"{phase_key} retried {retries} time(s) but no RCA artifact"))
        elif os.path.isfile(rca_path):
            checks.append(CheckResult("rca", f"{rca_file} present (no retries)", "pass",
                                      "RCA artifact exists even without retries (preemptive)"))

    return checks


def check_pipeline_blockers(output_dir, config):
    """Verify results/pipeline_blockers.json schema when present."""
    checks = []
    blockers_path = os.path.join(output_dir, "results", "pipeline_blockers.json")
    if not os.path.isfile(blockers_path):
        return checks

    try:
        with open(blockers_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        checks.append(CheckResult("blockers", "pipeline_blockers.json readable", "fail", str(e)))
        return checks

    blockers = data.get("blockers", [])
    checks.append(CheckResult("blockers", "pipeline_blockers.json valid", "pass",
                              f"{len(blockers)} blocker(s)"))

    valid_phases = {"benchmark", "profile-analyze", "kernel-optimize", "integration"}
    for i, blocker in enumerate(blockers):
        phase = blocker.get("phase", "")
        if phase not in valid_phases:
            checks.append(CheckResult("blockers", f"blocker[{i}] phase valid", "fail",
                                      f"Invalid phase: {phase}"))
        required = {"phase", "summary", "terminal_action"}
        missing = required - set(blocker.keys())
        if missing:
            checks.append(CheckResult("blockers", f"blocker[{i}] schema valid", "warning",
                                      f"Missing fields: {missing}"))

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

        with open(report_md) as f:
            content = f.read()

        # Check for new template sections
        blockers_path = os.path.join(output_dir, "results", "pipeline_blockers.json")
        has_blockers = os.path.isfile(blockers_path)

        if "## Pipeline Status" in content:
            checks.append(CheckResult("report", "report has Pipeline Status section", "pass"))
        else:
            checks.append(CheckResult("report", "report has Pipeline Status section", "warning",
                                      "Missing ## Pipeline Status section"))

        if has_blockers and "## Blockers" in content:
            checks.append(CheckResult("report", "report has Blockers section (blockers exist)", "pass"))
        elif has_blockers:
            checks.append(CheckResult("report", "report has Blockers section (blockers exist)", "warning",
                                      "pipeline_blockers.json exists but ## Blockers not in report"))

        if "## Deferred Work" in content:
            checks.append(CheckResult("report", "report has Deferred Work section", "pass"))
        elif not has_blockers:
            checks.append(CheckResult("report", "report has Deferred Work section", "warning",
                                      "No blockers but ## Deferred Work missing"))

        # Check that old-style ## Recommendations is gone when new sections exist
        if "## Recommendations" in content and "## Pipeline Status" in content:
            checks.append(CheckResult("report", "report uses new template (no old Recommendations)", "warning",
                                      "Both ## Recommendations and ## Pipeline Status present"))
    else:
        checks.append(CheckResult("report", "optimization_report.md exists", "skip"))

    summary_path = os.path.join(output_dir, "report", "optimization_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if "config_key" in summary or "kernel_results" in summary:
            checks.append(CheckResult("report", "optimization_summary.json valid", "pass"))
        else:
            checks.append(CheckResult("report", "optimization_summary.json valid", "fail",
                                      f"Missing config_key. Keys: {list(summary.keys())}"))

        # Check new summary fields
        if "pipeline_status" in summary:
            checks.append(CheckResult("report", "summary has pipeline_status", "pass",
                                      summary["pipeline_status"]))

            # Cross-check pipeline_status against blockers and report
            blockers_path = os.path.join(output_dir, "results", "pipeline_blockers.json")
            has_blockers = os.path.isfile(blockers_path)
            ps = summary["pipeline_status"]

            if has_blockers and ps == "completed":
                checks.append(CheckResult("report", "pipeline_status consistent with blockers",
                                          "fail",
                                          "pipeline_blockers.json exists but status is 'completed'"))
            elif not has_blockers and ps == "completed with blockers":
                comp_path = os.path.join(output_dir, "results", "optimization_comparison.json")
                has_fail_gate = False
                if os.path.isfile(comp_path):
                    try:
                        comp = load_json(comp_path)
                        has_fail_gate = comp.get("performance_gate") == "fail"
                    except (json.JSONDecodeError, IOError):
                        pass
                if not has_fail_gate:
                    checks.append(CheckResult("report", "pipeline_status consistent with blockers",
                                              "fail",
                                              "No blockers file and no fail gate but status is 'completed with blockers'"))
                else:
                    checks.append(CheckResult("report", "pipeline_status consistent with blockers",
                                              "pass", "fail gate justifies 'completed with blockers'"))
            else:
                checks.append(CheckResult("report", "pipeline_status consistent with blockers",
                                          "pass", ps))

            # Cross-check against report completion line
            report_md = os.path.join(output_dir, "report", "optimization_report.md")
            if os.path.isfile(report_md):
                with open(report_md) as f:
                    report_content = f.read()
                completion_match = re.search(
                    r"\*\*Completion\*\*:\s*(.+)",
                    report_content)
                if completion_match:
                    report_status = completion_match.group(1).strip()
                    if report_status == ps:
                        checks.append(CheckResult("report",
                                                  "summary pipeline_status matches report completion",
                                                  "pass", ps))
                    else:
                        checks.append(CheckResult("report",
                                                  "summary pipeline_status matches report completion",
                                                  "warning",
                                                  f"summary='{ps}', report='{report_status}'"))

        if "blocker_count" in summary:
            checks.append(CheckResult("report", "summary has blocker_count", "pass",
                                      f"blocker_count={summary['blocker_count']}"))
        if "all_phases_completed" in summary:
            checks.append(CheckResult("report", "summary has all_phases_completed", "pass",
                                      f"all_phases_completed={summary['all_phases_completed']}"))
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
    elif mode == "profile":
        expected = ["env", "config", "profile", "profile-analyze"]
    else:
        expected = OPTIMIZE_PHASES

    # Check for pipeline blockers that explain missing phases
    blockers_path = os.path.join(output_dir, "results", "pipeline_blockers.json")
    blocker_phases = set()
    if os.path.isfile(blockers_path):
        try:
            with open(blockers_path) as f:
                blockers_data = json.load(f)
            for b in blockers_data.get("blockers", []):
                blocker_phases.add(b.get("phase", ""))
        except (json.JSONDecodeError, IOError):
            pass

    # terminal_policy: stop phases (benchmark, profile-analyze) block all downstream
    stop_phases = {"benchmark", "profile-analyze"}
    allow_partial_phases = {"kernel-optimize", "integration"}

    missing = [p for p in expected if p not in completed]
    if missing:
        # Check if missing phases are explained by blockers + terminal_policy
        explained = []
        unexplained = []
        for m in missing:
            if m in blocker_phases:
                explained.append(m)
            elif any(bp in stop_phases and bp in blocker_phases for bp in stop_phases):
                explained.append(m)
            elif any(bp in allow_partial_phases and bp in blocker_phases for bp in allow_partial_phases):
                if m not in ("report-generate",):
                    explained.append(m)
                else:
                    unexplained.append(m)
            else:
                unexplained.append(m)

        if unexplained:
            checks.append(CheckResult("progress", "all expected phases completed", "fail",
                                      f"Missing: {unexplained}"))
        elif explained:
            checks.append(CheckResult("progress", "all expected phases completed", "warning",
                                      f"Missing but explained by blockers: {explained}"))
        else:
            checks.append(CheckResult("progress", "all expected phases completed", "pass",
                                      f"{len(completed)}/{len(expected)} phases"))
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
        return "phase-04-profile.md"
    if "benchmark" in basename or "docker_run" in basename:
        if "profile" in basename:
            return "phase-04-profile.md"
        return "phase-02-benchmark.md"
    if "problem" in basename or "upstream" in basename or "fork" in basename or "baseline_dispatch" in basename:
        return "phase-06-problem-generate.md"
    if "kernel" in basename or "geak" in basename or "library_test" in basename or "preflight_dispatch" in basename:
        return "phase-07-kernel-optimize.md"
    if "rebuild" in basename or "dispatch_verification" in basename or "integration_manifest" in basename:
        return "phase-08-integration.md"
    if "report" in dirpart:
        return "phase-09-report-generate.md"
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
        speedup = comp.get("speedup")
        ttft_pct = comp.get("ttft_regression_pct")
        performance_gate = comp.get("performance_gate", expected_performance_gate(speedup, ttft_pct))
        if speedup is not None and performance_gate in {"warn", "fail"}:
            bl = comp.get("baseline", {}).get("total_token_throughput", "?")
            opt = comp.get("optimized", {}).get("total_token_throughput", "?")
            issues.append(Issue(
                source="results/optimization_comparison.json",
                severity="warning" if performance_gate == "warn" else "error",
                pattern="E2E performance warning band" if performance_gate == "warn" else "E2E performance degradation",
                context=f"speedup={speedup:.3f}x, baseline={bl}, optimized={opt}, gate={performance_gate}",
                suggested_phase="08-integration.md",
                analysis="Optimized E2E throughput did not land in the clean pass band. Possible causes: "
                         "torch.compile/CUDAGraph masking kernel-level gains, fork rebuild not active, "
                         "dispatch swap not honored (vendor symbol still firing), or regression in "
                         "fused ops. Warn-band results are allowed but should be reported honestly; "
                         "fail-band results need recovery.",
            ))

    # Library-rebuild contract: dispatch swap leakage check.
    # If rocprofv3 (Phase 8) shows the vendor symbol still firing or a
    # required redirect was not honored, the rebuild is integrated but
    # not actually active — surface as an integration-phase issue.
    disp_path = os.path.join(output_dir, "results", "dispatch_verification.json")
    if os.path.isfile(disp_path):
        try:
            with open(disp_path) as f:
                disp = json.load(f)
        except (json.JSONDecodeError, OSError):
            disp = {}
        if disp.get("dispatch_verified") is False:
            issues.append(Issue(
                source="results/dispatch_verification.json",
                severity="error",
                pattern="dispatch not verified",
                context=f"dispatch_verified=false; expected_symbol_total_count="
                        f"{disp.get('expected_symbol_total_count')}, "
                        f"vendor_symbol_leaked_count="
                        f"{disp.get('vendor_symbol_leaked_count')}",
                suggested_phase="08-integration.md",
                analysis="Phase 8 rebuild completed but rocprofv3 did not confirm the "
                         "expected GEAK-optimized symbols are firing. Either the fork "
                         "rebuild did not shadow the vendor install (check Python import "
                         "path), the kernel was never imported, or the dispatch site "
                         "still resolves to the vendor symbol.",
            ))
        leaked = disp.get("vendor_symbol_leaked_count")
        if isinstance(leaked, int) and leaked > 0:
            issues.append(Issue(
                source="results/dispatch_verification.json",
                severity="error",
                pattern="vendor symbol leaked",
                context=f"vendor_symbol_leaked_count={leaked}",
                suggested_phase="08-integration.md",
                analysis="The vendor baseline symbol(s) still appear in the post-rebuild "
                         "rocprofv3 trace. Dispatch swap is incomplete — both the "
                         "GEAK-optimized symbol and the vendor symbol are firing, which "
                         "means the rebuild path and the legacy dispatch path coexist.",
            ))
        rh = disp.get("redirect_honored_count")
        rr = disp.get("redirect_required_count")
        if isinstance(rh, int) and isinstance(rr, int) and rh < rr:
            issues.append(Issue(
                source="results/dispatch_verification.json",
                severity="error",
                pattern="redirect not honored",
                context=f"redirect_honored={rh}/{rr}",
                suggested_phase="08-integration.md",
                analysis="A dispatch_redirect_* strategy was planned in Phase 6 but the "
                         "post-rebuild trace does not show the redirect target firing. "
                         "Check the dispatch-site patch in the host fork.",
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
    checks.extend(check_skill_layout())
    checks.extend(check_env(output_dir, config))
    checks.extend(check_config(output_dir, config))
    checks.extend(check_benchmark(output_dir, config))
    checks.extend(check_benchmark_analyze(output_dir, config))
    checks.extend(check_profile(output_dir, config))
    checks.extend(check_profile_analyze(output_dir, config))
    checks.extend(check_problem_generate(output_dir, config))
    checks.extend(check_kernel_optimize(output_dir, config))
    checks.extend(check_integration(output_dir, config))
    checks.extend(check_integration_manifest(output_dir, config))
    checks.extend(check_phase08_result_scalars(output_dir, config))
    checks.extend(check_report(output_dir, config))
    checks.extend(check_rca_artifacts(output_dir, config))
    checks.extend(check_pipeline_blockers(output_dir, config))
    checks.extend(check_progress_consistency(output_dir, config))
    checks.extend(check_multi_agent_workspace(output_dir, config))

    # Scan for issues
    print("\nScanning logs and artifacts for issues...")
    issues = scan_for_issues(output_dir)

    # Fail if all checks were skipped — no artifacts to validate
    passed = sum(1 for c in checks if c.status == "pass")
    failed = sum(1 for c in checks if c.status == "fail")
    if passed == 0 and failed == 0:
        print("ERROR: All checks were skipped — no artifacts to validate")
        return False

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
    parser = argparse.ArgumentParser(description="E2E validator for inference-optimize")
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
