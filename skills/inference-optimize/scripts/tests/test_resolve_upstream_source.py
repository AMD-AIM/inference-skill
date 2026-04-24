"""Tests for resolve_upstream_source.py."""

import json
import os
import subprocess
import sys

import pytest
import yaml

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "optimize", "resolve_upstream_source.py"
)


def _make_map(tmp_path):
    path = tmp_path / "kernel_source_map.yaml"
    yaml.safe_dump(
        {
            "entries": [
                {
                    "symbol_pattern": "fused_recurrent_*",
                    "library": "fla",
                    "source_form": "triton",
                    "bucket": "A",
                    "geak_strategy": "in_place_optimize",
                    "upstream_repo": "sustcsonglin/flash-linear-attention",
                    "source_file": "fla/ops/x.py",
                    "library_test_path": "tests/ops/test_x.py",
                    "library_test_command": "pytest -x -q {library_test_path}",
                    "rebuild_command": "pip install -e .",
                    "expected_dispatch_symbols": ["fused_recurrent_*"],
                    "vendor_baseline_symbols": [],
                    "geak_feasibility": "easy",
                    "cost_profile": {"per_attempt_minutes": 1, "rebuild_minutes": 1},
                    "geak_task_hint": "Optimize.",
                    "redirect_target": None,
                },
                {
                    "symbol_pattern": "Cijk_*",
                    "library": "hipblaslt_tensile",
                    "source_form": "tensile_asm",
                    "bucket": "B",
                    "geak_strategy": "dispatch_redirect_to_open_lib",
                    "gating_reason": "no_test_harness",
                    "upstream_repo": "ROCm/hipBLASLt",
                    "source_file": "library/src/.../x.s",
                    "library_test_path": None,
                    "library_test_command": None,
                    "rebuild_command": "cmake --build build -j",
                    "expected_dispatch_symbols": ["Cijk_*"],
                    "vendor_baseline_symbols": ["Cijk_*"],
                    "geak_feasibility": "hard_experimental",
                    "cost_profile": {"per_attempt_minutes": 5, "rebuild_minutes": 5},
                    "geak_task_hint": "Tensile asm.",
                    "redirect_target": {"library": "aiter_triton", "symbol": "aiter_gemm_triton"},
                },
            ]
        },
        path.open("w"),
    )
    return path


def _make_pins(tmp_path):
    path = tmp_path / "library_pins.yaml"
    yaml.safe_dump(
        {
            "pins": [
                {
                    "vllm_version": "v0.19.1",
                    "pins": {"fla": "deadbeef", "hipblaslt": "cafef00d"},
                }
            ]
        },
        path.open("w"),
    )
    return path


def _run(symbol, map_path, pins_path):
    return subprocess.run(
        [
            sys.executable, SCRIPT,
            "--symbol", symbol,
            "--map", str(map_path),
            "--pins", str(pins_path),
            "--vllm-version", "v0.19.1",
        ],
        capture_output=True,
        text=True,
    )


class TestResolveUpstreamSource:
    def test_match_first_entry(self, tmp_path):
        result = _run("fused_recurrent_gated_delta_rule_fwd_kernel", _make_map(tmp_path), _make_pins(tmp_path))
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["library"] == "fla"
        assert payload["source_form"] == "triton"
        assert payload["bucket"] == "A"
        assert payload["geak_strategy"] == "in_place_optimize"
        assert payload["pinned_commit"] == "deadbeef"
        assert payload["matched_symbol"] == "fused_recurrent_gated_delta_rule_fwd_kernel"

    def test_match_glob_pattern(self, tmp_path):
        result = _run("Cijk_Ailk_Bljk_BBH_MT128x128", _make_map(tmp_path), _make_pins(tmp_path))
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["library"] == "hipblaslt_tensile"
        assert payload["bucket"] == "B"
        assert payload["redirect_target"] == {"library": "aiter_triton", "symbol": "aiter_gemm_triton"}

    def test_unresolved_symbol(self, tmp_path):
        result = _run("totally_unknown_kernel", _make_map(tmp_path), _make_pins(tmp_path))
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["library"] == "unknown"
        assert payload["geak_strategy"] == "unfeasible_record_only"
        assert payload["skip_reason"] == "unresolved_unknown_symbol"

    def test_first_match_wins(self, tmp_path):
        # The Cijk_ entry should not match a fused_recurrent symbol.
        result = _run("fused_recurrent_xyz", _make_map(tmp_path), _make_pins(tmp_path))
        payload = json.loads(result.stdout)
        assert payload["library"] == "fla"
