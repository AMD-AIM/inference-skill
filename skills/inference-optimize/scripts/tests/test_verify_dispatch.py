"""Tests for verify_dispatch.py.

Exercises the rocprofv3 JSON parser and the manifest-evaluation logic
against frozen fixture traces (no real rocprofv3 invocation).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "integrate"))
import verify_dispatch  # noqa: E402


def _write_trace(tmp_path, kernels):
    """Build a rocprofv3-shaped JSON: list of dispatch records under
    'kernel_dispatches', each with a `kernel_name` field. parse_kernel_counts
    walks any nested structure and counts `kernel_name` occurrences."""
    doc = {
        "kernel_dispatches": [
            {"kernel_name": k, "kind": "kernel_dispatch", "duration_ns": 100}
            for k in kernels
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(doc))
    return path


class TestParseKernelCounts:
    def test_counts_repeated_kernels(self, tmp_path):
        path = _write_trace(tmp_path, ["a_kernel", "b_kernel", "a_kernel", "a_kernel"])
        counts = verify_dispatch.parse_kernel_counts(str(path))
        assert counts["a_kernel"] == 3
        assert counts["b_kernel"] == 1

    def test_missing_file_returns_empty(self, tmp_path):
        assert verify_dispatch.parse_kernel_counts(str(tmp_path / "nope.json")) == {}

    def test_legacy_buffer_records_schema(self, tmp_path):
        """rocprofv3 0.4.x nests dispatches under
        rocprofiler-sdk-tool.buffer_records.kernel_dispatch."""
        doc = {
            "rocprofiler-sdk-tool": {
                "buffer_records": {
                    "kernel_dispatch": [
                        {"kernel_name": "x_kernel"},
                        {"kernel_name": "y_kernel"},
                        {"kernel_name": "x_kernel"},
                    ]
                }
            }
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(doc))
        counts = verify_dispatch.parse_kernel_counts(str(path))
        assert counts == {"x_kernel": 2, "y_kernel": 1}

    def test_unrecognized_schema_warns_and_returns_empty(self, tmp_path):
        """A trace with no recognized dispatch path must warn (not silently
        zero) so dispatch verification fails closed loudly."""
        doc = {"some_unrelated_top_level": [{"kernel_name": "z_kernel"}]}
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(doc))
        with pytest.warns(UserWarning, match="did not match any known"):
            counts = verify_dispatch.parse_kernel_counts(str(path))
        assert counts == {}

    def test_no_double_count_across_nesting(self, tmp_path):
        """Same kernel name appearing under an unrelated nested key must
        NOT be summed twice -- only the canonical dispatch list counts."""
        doc = {
            "kernel_dispatches": [
                {"kernel_name": "same_kernel"},
                {"kernel_name": "same_kernel"},
            ],
            # Unrelated nested data containing the same kernel name --
            # the recursive walker would have summed these. The direct-
            # lookup parser must ignore them.
            "extra": {"nested": {"kernel_name": "same_kernel"}},
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(doc))
        counts = verify_dispatch.parse_kernel_counts(str(path))
        assert counts == {"same_kernel": 2}


class TestEvaluate:
    def test_dispatch_verified_when_expected_present_and_no_leak(self):
        manifest = {
            "optimizations": [
                {
                    "name": "k_alpha",
                    "optimize": True,
                    "expected_dispatch_symbols": ["alpha_kernel"],
                    "vendor_baseline_symbols": [],
                    "geak_strategy": "in_place_optimize",
                }
            ]
        }
        counts = {"alpha_kernel": 5}
        e = verify_dispatch.evaluate(manifest, counts)
        assert e["dispatch_verified"] is True
        assert e["expected_symbol_total_count"] == 5
        assert e["vendor_symbol_leaked_count"] == 0
        assert e["redirect_required_count"] == 0

    def test_vendor_leak_fails_verification(self):
        manifest = {
            "optimizations": [
                {
                    "name": "k_redirect",
                    "optimize": True,
                    "expected_dispatch_symbols": ["target_kernel"],
                    "vendor_baseline_symbols": ["vendor_kernel"],
                    "geak_strategy": "dispatch_redirect_to_open_lib",
                }
            ]
        }
        counts = {"target_kernel": 3, "vendor_kernel": 1}
        e = verify_dispatch.evaluate(manifest, counts)
        assert e["dispatch_verified"] is False
        assert e["vendor_symbol_leaked_count"] == 1
        assert e["redirect_required_count"] == 1
        assert e["redirect_honored_count"] == 0

    def test_redirect_honored_when_target_only(self):
        manifest = {
            "optimizations": [
                {
                    "name": "k_redirect",
                    "optimize": True,
                    "expected_dispatch_symbols": ["target_kernel"],
                    "vendor_baseline_symbols": ["vendor_kernel"],
                    "geak_strategy": "dispatch_redirect_to_triton",
                }
            ]
        }
        counts = {"target_kernel": 7}
        e = verify_dispatch.evaluate(manifest, counts)
        assert e["dispatch_verified"] is True
        assert e["redirect_honored_count"] == 1

    def test_glob_patterns_match(self):
        manifest = {
            "optimizations": [
                {
                    "name": "k_glob",
                    "optimize": True,
                    "expected_dispatch_symbols": ["paged_attention_v2*"],
                    "vendor_baseline_symbols": [],
                    "geak_strategy": "in_place_optimize",
                }
            ]
        }
        counts = {"paged_attention_v2_kernel": 2, "paged_attention_v2_reduce_kernel": 4}
        e = verify_dispatch.evaluate(manifest, counts)
        assert e["expected_symbol_total_count"] == 6
        assert e["dispatch_verified"] is True

    def test_missing_expected_symbol_fails(self):
        manifest = {
            "optimizations": [
                {
                    "name": "k_alpha",
                    "optimize": True,
                    "expected_dispatch_symbols": ["alpha_kernel"],
                    "vendor_baseline_symbols": [],
                    "geak_strategy": "in_place_optimize",
                }
            ]
        }
        counts = {"unrelated_kernel": 1}
        e = verify_dispatch.evaluate(manifest, counts)
        assert e["dispatch_verified"] is False
        assert any(r["status"] == "missing" for r in e["expected_symbols"])
