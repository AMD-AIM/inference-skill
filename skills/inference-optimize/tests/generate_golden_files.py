"""Generate golden files for V1 regression testing.

Run once to capture current runner.py behavior.
Golden files are committed alongside this script.
"""
import json
import os
import pathlib
import sys
import tempfile

# Add runner directory to path for imports
SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))

from runner import DeterministicRunner, RunnerState, compute_parity_hash

SCENARIOS = {
    "optimize_all_pass": {
        "description": "All phases PASS, no retries",
        "mode": "optimize",
        "dispatch_verdicts": {},  # all default to PASS
        "monitor_verdicts": {},   # all default to PASS
    },
    "optimize_with_retry": {
        "description": "Phase-07 FAIL -> retry -> PASS",
        "mode": "optimize",
        "dispatch_verdicts": {"kernel-optimize": ["FAIL", "PASS"]},
        "monitor_verdicts": {"kernel-optimize": [{"verdict": "FAIL", "failure_type": "logic"}, {"verdict": "PASS"}]},
    },
    "optimize_with_fallback": {
        "description": "Phase-07 budget exhausted -> fallback to phase-06",
        "mode": "optimize",
        "dispatch_verdicts": {"kernel-optimize": ["FAIL", "FAIL", "FAIL"]},
        "monitor_verdicts": {"kernel-optimize": [
            {"verdict": "FAIL", "failure_type": "logic"},
            {"verdict": "FAIL", "failure_type": "logic"},
            {"verdict": "FAIL", "failure_type": "logic"},
        ]},
    },
}


def make_mock_fns(scenario):
    """Create mock dispatch_fn, monitor_fn, rca_fn from scenario spec.

    Signatures match runner.py expectations:
      dispatch_fn(phase_key, handoff_path) -> verdict_dict
      monitor_fn(phase_key, result_path, summary_path, checks) -> verdict_dict
      rca_fn(phase_key, manifest_dict) -> rca_dict
    """
    call_counts = {}

    def dispatch_fn(phase_key, handoff_path):
        key = f"dispatch_{phase_key}"
        call_counts[key] = call_counts.get(key, 0) + 1
        idx = call_counts[key] - 1
        verdicts = scenario.get("dispatch_verdicts", {}).get(phase_key, [])
        if idx < len(verdicts):
            v = verdicts[idx]
            return {"verdict": v, "result_path": f"agent-results/phase-{phase_key}-result.md"}
        return {"verdict": "PASS", "result_path": f"agent-results/phase-{phase_key}-result.md"}

    def monitor_fn(phase_key, result_path, summary_path, checks):
        key = f"monitor_{phase_key}"
        call_counts[key] = call_counts.get(key, 0) + 1
        idx = call_counts[key] - 1
        verdicts = scenario.get("monitor_verdicts", {}).get(phase_key, [])
        if idx < len(verdicts):
            return verdicts[idx]
        return {"verdict": "PASS"}

    def rca_fn(phase_key, manifest_dict):
        return {"terminal_action": "retry", "analysis": "mock RCA"}

    return dispatch_fn, monitor_fn, rca_fn


def _build_config(tmpdir, mode):
    """Build a minimal config dict for testing."""
    return {
        "CONFIG_KEY": "test-config",
        "OUTPUT_DIR": tmpdir,
        "MODE": mode,
        "REPO_DIR": os.path.join(tmpdir, "benchmark_repo"),
        "REPO_URL": "https://github.com/SemiAnalysisAI/InferenceX.git",
        "HF_CACHE": "/tmp/hf",
        "SCRIPTS_DIR": os.path.join(tmpdir, "scripts"),
        "PROFILE_DIR": os.path.join(tmpdir, "profiles"),
        "RESULTS_DIR": os.path.join(tmpdir, "results"),
        "PROBLEMS_DIR": os.path.join(tmpdir, "problems"),
        "OPTIMIZED_DIR": os.path.join(tmpdir, "optimized"),
        "GEAK_DIR": "/tmp/geak",
        "GEAK_OE_DIR": "/tmp/geak_oe",
        "GEAK_MODE": "auto",
        "OPTIMIZE_SCOPE": "all",
        "OPTIMIZE_PRIORITY_THRESHOLD": "0.1",
        "ENV_INFO_FILE": os.path.join(tmpdir, "env_info.json"),
        "RESOURCES_DIR": os.path.join(tmpdir, "resources"),
        "TEMPLATES_DIR": os.path.join(tmpdir, "templates"),
        "REPORT_DIR": os.path.join(tmpdir, "reports"),
        "GPUS": "0,1",
        "DRY_RUN_NOTE": "",
        "ENFORCE_EAGER_FLAG": "",
        "FILTER_EP": "",
        "FILTER_TP": "",
        "FILTER_CONC_START": "",
        "FILTER_CONC_END": "",
        "FILTER_SEQ": "",
    }


def _create_artifacts(tmpdir):
    """Create required artifact files and directories for optimize mode."""
    for d in ["handoff", "agent-results", "monitor", "results", "results/parity",
              "problems", "optimized", "profiles", "scripts", "templates",
              "reports", "resources", "results/gap_analysis"]:
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    with open(os.path.join(tmpdir, "env_info.json"), "w") as f:
        json.dump({"gpu_arch": "mi300x", "gpu_count": 8}, f)
    with open(os.path.join(tmpdir, "results/sweep_configs.json"), "w") as f:
        json.dump({"framework": "sglang", "precision": "bf16"}, f)
    with open(os.path.join(tmpdir, "results/gap_analysis/gap_analysis.json"), "w") as f:
        json.dump({"top_kernels": ["kernel_a", "kernel_b"]}, f)
    with open(os.path.join(tmpdir, "results/profile_analysis.json"), "w") as f:
        json.dump({"status": "complete"}, f)


def generate_golden(scenario_name, scenario_spec, output_dir):
    """Run a scenario and capture golden output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _build_config(tmpdir, scenario_spec["mode"])
        _create_artifacts(tmpdir)

        registry = json.loads(REGISTRY_PATH.read_text())
        dispatch_fn, monitor_fn, rca_fn = make_mock_fns(scenario_spec)

        runner = DeterministicRunner(config, registry, tmpdir)
        state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)

        # Capture parity snapshot
        golden = state.parity_snapshot()
        golden_path = os.path.join(output_dir, f"{scenario_name}.json")
        with open(golden_path, "w") as f:
            json.dump(golden, f, indent=2, sort_keys=True)
        print(f"  Written: {golden_path}")
        return golden


def main():
    golden_dir = os.path.join(os.path.dirname(__file__), "golden")
    os.makedirs(golden_dir, exist_ok=True)

    for name, spec in SCENARIOS.items():
        print(f"Generating golden file for: {name}")
        try:
            generate_golden(name, spec, golden_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
