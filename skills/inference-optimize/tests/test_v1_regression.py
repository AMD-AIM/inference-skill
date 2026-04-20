"""V1 regression tests -- compare refactored runner output against golden files."""
import json
import os
import pathlib
import sys
import pytest
import tempfile

SKILL_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_DIR = SKILL_ROOT / "scripts" / "orchestrate"
REGISTRY_PATH = SKILL_ROOT / "orchestrator" / "phase-registry.json"

sys.path.insert(0, str(RUNNER_DIR))

from runner import DeterministicRunner, RunnerState, compute_parity_hash

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
SCENARIOS = ["optimize_all_pass", "optimize_with_retry", "optimize_with_fallback"]


def load_golden(scenario_name):
    path = os.path.join(GOLDEN_DIR, f"{scenario_name}.json")
    if not os.path.isfile(path):
        pytest.skip(f"Golden file not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR


def _build_config(tmpdir, mode):
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


def _registry_for_scenario(scenario):
    registry = json.loads(REGISTRY_PATH.read_text())
    # Fallback parity scenario needs finite retry budgets.
    if scenario == "optimize_with_fallback":
        registry["rerun"]["max_per_phase"] = 2
        registry["rerun"]["max_total"] = 3
    return registry


class TestV1Regression:
    """Ensure refactored runner produces identical output to golden files."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_parity_snapshot_matches_golden(self, scenario):
        """Parity snapshot must match golden file exactly."""
        golden = load_golden(scenario)
        # Re-run the scenario
        from tests.generate_golden_files import SCENARIOS as SPECS, make_mock_fns

        spec = SPECS[scenario]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, spec["mode"])
            _create_artifacts(tmpdir)
            registry = _registry_for_scenario(scenario)
            dispatch_fn, monitor_fn, rca_fn = make_mock_fns(spec)
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)
            current = state.parity_snapshot()

        assert current == golden, f"Parity mismatch for {scenario}"

    def test_progress_schema_unchanged_v1(self):
        """progress.json schema_version is 1.0 in the runner state."""
        from tests.generate_golden_files import SCENARIOS as SPECS, make_mock_fns

        spec = SPECS["optimize_all_pass"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, spec["mode"])
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            dispatch_fn, monitor_fn, rca_fn = make_mock_fns(spec)
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)

            # schema_version lives in progress.json (to_progress), not parity_snapshot
            progress = state.to_progress()
            assert progress.get("schema_version") == "1.0"

    def test_v2_false_produces_identical_parity_hash(self):
        """With V2_MONITOR=false, parity hash must be identical to golden."""
        golden = load_golden("optimize_all_pass")
        golden_hash = compute_parity_hash(golden)

        from tests.generate_golden_files import SCENARIOS as SPECS, make_mock_fns

        spec = SPECS["optimize_all_pass"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, spec["mode"])
            config["V2_MONITOR"] = False
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            dispatch_fn, monitor_fn, rca_fn = make_mock_fns(spec)
            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)
            current_hash = compute_parity_hash(state.parity_snapshot())

        assert current_hash == golden_hash

    def test_v1_warn_path_remains_non_rca(self):
        """V1 WARN verdict should not spawn RCA and should continue execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_config(tmpdir, "benchmark")
            config["V2_MONITOR"] = False
            _create_artifacts(tmpdir)
            registry = json.loads(REGISTRY_PATH.read_text())
            rca_calls = []

            def dispatch_fn(phase_key, handoff_path):
                return {"verdict": "PASS"}

            def monitor_fn(phase_key, result_path, summary_path, checks):
                if phase_key == "benchmark":
                    return {"verdict": "WARN", "failure_type": "logic"}
                return {"verdict": "PASS"}

            def rca_fn(phase_key, manifest_dict):
                rca_calls.append((phase_key, manifest_dict))
                return {"terminal_action": "retry", "analysis": "unexpected"}

            runner = DeterministicRunner(config, registry, tmpdir)
            state = runner.run(dispatch_fn=dispatch_fn, monitor_fn=monitor_fn, rca_fn=rca_fn)

            assert state.status == "completed"
            assert rca_calls == []
