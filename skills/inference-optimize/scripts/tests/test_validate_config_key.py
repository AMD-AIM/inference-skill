"""Tests for validate_config_key.py -- key found, missing with suggestions, malformed YAML."""

import os
import subprocess
import sys

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "env", "validate_config_key.py"
)


def run(config_file, config_key, max_suggestions=None):
    cmd = [sys.executable, SCRIPT, "--config-file", config_file, "--config-key", config_key]
    if max_suggestions is not None:
        cmd.extend(["--max-suggestions", str(max_suggestions)])
    return subprocess.run(cmd, capture_output=True, text=True)


def write_yaml(tmpdir, content):
    path = os.path.join(tmpdir, "master.yaml")
    with open(path, "w") as f:
        f.write(content)
    return path


class TestValidateConfigKey:
    def test_key_found(self, tmp_path):
        """Valid key returns exit 0 and prints CONFIG_KEY_OK."""
        yaml_path = write_yaml(str(tmp_path), "gptoss-fp4-mi355x-vllm:\n  model: foo\ndsr1-fp4-mi355x-sglang:\n  model: bar\n")
        result = run(yaml_path, "gptoss-fp4-mi355x-vllm")
        assert result.returncode == 0
        assert "CONFIG_KEY_OK=gptoss-fp4-mi355x-vllm" in result.stdout

    def test_key_missing_with_suggestions(self, tmp_path):
        """Missing key prints suggestions and returns exit 1."""
        yaml_path = write_yaml(str(tmp_path), "gptoss-fp4-mi355x-vllm:\n  model: foo\ngptoss-fp8-mi355x-vllm:\n  model: bar\n")
        result = run(yaml_path, "gptoss-fp4-mi355x-sglang")
        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "ERROR" in result.stdout

    def test_key_missing_no_close_matches(self, tmp_path):
        """Totally unrelated key shows no-matches message."""
        yaml_path = write_yaml(str(tmp_path), "alpha:\n  x: 1\nbeta:\n  x: 2\n")
        result = run(yaml_path, "zzz-completely-unrelated-key-xyz")
        assert result.returncode == 1

    def test_file_not_found(self, tmp_path):
        """Non-existent config file returns exit 1."""
        result = run(os.path.join(str(tmp_path), "nonexistent.yaml"), "anykey")
        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "ERROR" in result.stdout

    def test_malformed_yaml(self, tmp_path):
        """Invalid YAML returns exit 1."""
        yaml_path = write_yaml(str(tmp_path), ":\n  - [invalid\n  yaml: {broken")
        result = run(yaml_path, "anykey")
        assert result.returncode == 1

    def test_yaml_not_dict(self, tmp_path):
        """YAML that parses to a list instead of dict returns exit 1."""
        yaml_path = write_yaml(str(tmp_path), "- item1\n- item2\n")
        result = run(yaml_path, "item1")
        assert result.returncode == 1
