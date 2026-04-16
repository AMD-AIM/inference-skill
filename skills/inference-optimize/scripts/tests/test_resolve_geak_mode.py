"""Tests for resolve_geak_mode.py -- 6 mode resolution paths."""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "optimize", "resolve_geak_mode.py"
)


def run(user_mode, env_info_path):
    result = subprocess.run(
        [sys.executable, SCRIPT, "--user-mode", user_mode, "--env-info", env_info_path],
        capture_output=True, text=True,
    )
    return result


def write_env_info(tmpdir, geak_available=True, llm_api_key_set=True):
    path = os.path.join(tmpdir, "env_info.json")
    with open(path, "w") as f:
        json.dump({"geak_available": geak_available, "llm_api_key_set": llm_api_key_set}, f)
    return path


class TestResolveGeakMode:
    def test_manual_always_manual(self, tmp_path):
        """User mode 'manual' always resolves to 'manual' regardless of env."""
        env_path = write_env_info(str(tmp_path), geak_available=True, llm_api_key_set=True)
        result = run("manual", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=manual" in result.stdout

    def test_auto_with_geak_and_key_resolves_full(self, tmp_path):
        """Auto mode with GEAK + API key -> full."""
        env_path = write_env_info(str(tmp_path))
        result = run("auto", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=full" in result.stdout

    def test_full_with_geak_and_key_resolves_full(self, tmp_path):
        """Full mode with GEAK + API key -> full."""
        env_path = write_env_info(str(tmp_path))
        result = run("full", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=full" in result.stdout

    def test_triton_only_resolves_triton_only(self, tmp_path):
        """Triton-only mode with GEAK + API key -> triton_only."""
        env_path = write_env_info(str(tmp_path))
        result = run("triton_only", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=triton_only" in result.stdout

    def test_no_geak_falls_back_to_manual(self, tmp_path):
        """When GEAK unavailable, falls back to manual with warning."""
        env_path = write_env_info(str(tmp_path), geak_available=False)
        result = run("auto", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=manual" in result.stdout
        assert "WARNING" in result.stdout

    def test_no_api_key_falls_back_to_manual(self, tmp_path):
        """When API key missing, falls back to manual with warning."""
        env_path = write_env_info(str(tmp_path), geak_available=True, llm_api_key_set=False)
        result = run("auto", env_path)
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=manual" in result.stdout
        assert "WARNING" in result.stdout

    def test_missing_env_info_file_falls_back(self, tmp_path):
        """Non-existent env_info path falls back to manual."""
        result = run("full", os.path.join(str(tmp_path), "nonexistent.json"))
        assert result.returncode == 0
        assert "EFFECTIVE_GEAK_MODE=manual" in result.stdout
