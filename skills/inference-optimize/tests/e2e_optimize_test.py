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
            for k, v in retry_counts.items():
                if isinstance(v, int) and v > 2:
                    checks.append(CheckResult("multi-agent",
                                               f"retry_counts[{k}] within limit",
                                               "fail", f"Retried {v} times (max_per_phase=2)"))

        total_reruns = progress.get("total_reruns", 0)
        if isinstance(total_reruns, int) and total_reruns > 5:
            checks.append(CheckResult("multi-agent", "total_reruns within limit",
                                       "fail", f"Total reruns={total_reruns} (max=5)"))

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
                        if verdict_match and verdict_match.group(1).upper() in ("FAIL", "WARN"):
                            if "failure_type:" not in frontmatter:
                                checks.append(CheckResult("multi-agent",
                                                           f"phase-{idx:02d}-review.md {verdict_match.group(1).upper()} has failure_type",
                                                           "warning",
                                                           f"{verdict_match.group(1).upper()} verdict missing failure_type field"))
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

        registered = manifest.get("registered", manifest.get("patched", []))
        ops_key = "registered_ops" if framework == "vllm" else "patched_modules"
        if framework == "vllm":
            ops = manifest.get(ops_key, manifest.get("ops", registered))
        else:
            ops = manifest.get(ops_key, manifest.get("patched", manifest.get("ops", registered)))
        if ops:
            checks.append(CheckResult("integration", f"{plugin_dir_name} manifest valid", "pass",
                                      f"{len(ops)} {ops_key}"))
        else:
            winners = [n for n, s in geak_speedups.items() if s is not None and s > 1.0]
            detail = f"Empty {ops_key}"
            if winners:
                detail += f" (winning kernels exist: {', '.join(winners)})"
            checks.append(CheckResult("integration", f"{plugin_dir_name} manifest valid", "warning",
                                      detail))

        for reg in registered:
            kernel_file = reg.get("kernel", "")
            kernel_name = kernel_file.replace("problem_", "").replace("_opt.py", "").replace("_opt", "")
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

    required = {"schema_version", "targets", "plugin_type", "comparison_file", "summary"}
    missing = required - set(manifest.keys())
    if missing:
        checks.append(CheckResult("integration", "integration_manifest.json schema valid", "fail",
                                  f"Missing: {sorted(missing)}"))
    else:
        checks.append(CheckResult("integration", "integration_manifest.json schema valid", "pass"))

    summary = manifest.get("summary", {})
    if isinstance(summary, dict):
        summary_required = {"total_targets", "integrated", "blocked", "coverage_pct"}
        summary_missing = summary_required - set(summary.keys())
        if summary_missing:
            checks.append(CheckResult("integration", "integration_manifest summary fields", "fail",
                                      f"Missing: {sorted(summary_missing)}"))
        else:
            checks.append(CheckResult("integration", "integration_manifest summary fields", "pass",
                                      f"coverage_pct={summary.get('coverage_pct')}"))
            cov = summary.get("coverage_pct")
            if not isinstance(cov, (int, float)) or cov < 0 or cov > 1:
                checks.append(CheckResult("integration", "integration_manifest coverage_pct range",
                                          "fail", f"Expected 0..1 float, got {cov}"))
            else:
                checks.append(CheckResult("integration", "integration_manifest coverage_pct range",
                                          "pass", f"{cov}"))
    else:
        checks.append(CheckResult("integration", "integration_manifest summary is dict", "fail",
                                  f"Expected dict, got {type(summary).__name__}"))

    targets = manifest.get("targets", [])
    if isinstance(targets, list):
        checks.append(CheckResult("integration", "integration_manifest targets list", "pass",
                                  f"{len(targets)} target(s)"))
        valid_statuses = {"integrated", "blocked", "skipped"}
        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                checks.append(CheckResult("integration", f"integration_manifest target[{i}] valid",
                                          "fail", f"Expected dict, got {type(t).__name__}"))
                continue
            t_required = {"name", "status", "strategy"}
            t_missing = t_required - set(t.keys())
            if t_missing:
                checks.append(CheckResult("integration", f"integration_manifest target[{i}] schema",
                                          "fail", f"Missing: {sorted(t_missing)}"))
            t_status = t.get("status")
            if t_status and t_status not in valid_statuses:
                checks.append(CheckResult("integration",
                                          f"integration_manifest target[{i}] status enum",
                                          "fail", f"Invalid status '{t_status}', expected {valid_statuses}"))
            if t_status == "blocked" and not t.get("blocker_classification"):
                checks.append(CheckResult("integration",
                                          f"integration_manifest target[{i}] blocked has classification",
                                          "warning", "blocked target missing blocker_classification"))
    else:
        checks.append(CheckResult("integration", "integration_manifest targets list", "fail",
                                  f"Expected list, got {type(targets).__name__}"))

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
    if "problem" in basename or "fusion" in basename or "bottleneck" in basename:
        return "phase-06-problem-generate.md"
    if "kernel" in basename or "geak" in basename or "_opt" in basename:
        return "phase-07-kernel-optimize.md"
    if "plugin" in dirpart or "optimized" in dirpart:
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
                         "torch.compile/CUDAGraph masking kernel-level gains, plugin import overhead, "
                         "incorrect kernel dispatch, or regression in fused ops. Warn-band results are "
                         "allowed but should be reported honestly; fail-band results need recovery.",
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
