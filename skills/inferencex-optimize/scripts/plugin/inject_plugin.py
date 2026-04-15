#!/usr/bin/env python3
"""Inject optimization plugin import into a benchmark script.

Run INSIDE the Docker container:
  docker exec $CONTAINER python3 /tmp/inject_plugin.py --framework <vllm|sglang> --target <script>
"""

import argparse
import re
import sys


def inject_vllm(target):
    with open(target) as f:
        content = f.read()

    is_python = target.endswith('.py') or (content.lstrip().startswith('#!') and 'python' in content.split('\n')[0])

    if is_python:
        plugin_import = (
            '\n# Optimized kernel plugin (auto-injected by inferencex-optimize Phase 8)\n'
            'import os, sys\n'
            'os.environ["PYTHONPATH"] = "/workspace/optimized:" + os.environ.get("PYTHONPATH", "")\n'
            'sys.path.insert(0, "/workspace/optimized")\n'
            'try:\n    import vllm_plugin\nexcept Exception:\n    print("[WARNING] vllm_plugin import failed")\n'
        )
    else:
        plugin_import = (
            '\n# Optimized kernel plugin (auto-injected by inferencex-optimize Phase 8)\n'
            'export PYTHONPATH=/workspace/optimized:$PYTHONPATH\n'
            'python3 -c "import sys; sys.path.insert(0, \'/workspace/optimized\'); '
            'import vllm_plugin" 2>/dev/null || echo "[WARNING] vllm_plugin import failed"\n'
        )
    new_content = content.replace("vllm serve ", plugin_import + "vllm serve ", 1)
    if new_content != content:
        with open(target, "w") as f:
            f.write(new_content)
        print("Patched benchmark script with vllm_plugin import")
    else:
        print("ERROR: Could not find 'vllm serve' in script")
        sys.exit(1)


def inject_sglang(target):
    with open(target) as f:
        content = f.read()

    plugin_import = (
        '# Optimized kernel plugin (auto-injected by inferencex-optimize Phase 8)\n'
        'export PYTHONPATH=/workspace/optimized:$PYTHONPATH\n'
        'python3 -c "import sys; sys.path.insert(0, \'/workspace/optimized\'); '
        'import sglang_plugin" || echo "[WARNING] sglang_plugin import failed"\n\n'
    )
    new_content = content.replace(
        "python3 -m sglang.launch_server",
        plugin_import + "python3 -m sglang.launch_server",
        1,
    )
    if new_content != content:
        with open(target, "w") as f:
            f.write(new_content)
        print("Patched benchmark script with sglang_plugin import")
    else:
        print("ERROR: Could not find 'python3 -m sglang.launch_server' in script")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Inject plugin import into benchmark script")
    parser.add_argument("--framework", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--target", required=True, help="Path to benchmark script")
    args = parser.parse_args()

    if args.framework == "vllm":
        inject_vllm(args.target)
    else:
        inject_sglang(args.target)


if __name__ == "__main__":
    main()
