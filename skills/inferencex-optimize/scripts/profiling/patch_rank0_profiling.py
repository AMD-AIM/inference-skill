#!/usr/bin/env python3
"""Patch inference framework to restrict trace export to rank 0 only.

Prevents I/O contention when all TP workers write multi-GB traces simultaneously.
Run INSIDE the Docker container: docker exec $CONTAINER python3 /tmp/patch_rank0_profiling.py --framework <vllm|sglang>
"""

import argparse
import glob
import re
import sys


def patch_vllm():
    """Skip profiler creation on non-rank-0 workers; force-mount profiler routes."""
    path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"WARNING: {path} not found, skipping vLLM rank-0 patch")
        return

    content = content.replace(
        'if profiler_config.profiler == "torch":',
        'if profiler_config.profiler == "torch" and self.local_rank == 0:',
    )

    content = re.sub(
        r'(def profile\(self, is_start: bool = True\):\s*\n\s*)if self\.profiler is None:\s*\n\s*raise RuntimeError\(',
        r'\1if self.profiler is None:\n            return  # rank > 0: no profiler, silently skip\n        if False:\n            raise RuntimeError(',
        content,
    )

    with open(path, "w") as f:
        f.write(content)
    print("Patched gpu_worker.py: profiler enabled for rank 0 only")

    # Force-mount profiler routes (vLLM v0.18+ may skip if profiler_config is None at import)
    paths = glob.glob(
        "/usr/local/lib/python3.*/dist-packages/vllm/entrypoints/serve/profile/api_router.py"
    )
    if not paths:
        print("INFO: vLLM profiler api_router.py not found — routes may be mounted differently")
        return

    path = paths[0]
    with open(path) as f:
        content = f.read()
    old = "if app.state.args.profiler_config"
    if old in content:
        content = content.replace(
            old,
            "if True  # force-mount profiler routes (patched by inferencex-optimize)",
            1,
        )
        with open(path, "w") as f:
            f.write(content)
        print(f"Patched {path}: profiler routes unconditionally mounted")
    else:
        print("INFO: profiler_config guard not found — routes may already be unconditional")


def patch_sglang():
    """Skip trace export on non-rank-0 workers (all ranks still profile + barrier)."""
    paths = glob.glob(
        "/usr/local/lib/python3.*/dist-packages/sglang/srt/utils/profile_utils.py"
    )
    if not paths:
        print("WARNING: SGLang profile_utils.py not found, skipping rank-0 patch")
        return

    path = paths[0]
    with open(path) as f:
        content = f.read()

    old = '            self.torch_profiler.export_chrome_trace(\n                os.path.join(self.output_dir, filename)\n            )'
    new = '            if self.tp_rank == 0:\n                self.torch_profiler.export_chrome_trace(\n                    os.path.join(self.output_dir, filename)\n                )'
    if old in content:
        content = content.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(content)
        print("Patched profile_utils.py: trace export for rank 0 only")
    else:
        print("WARNING: export_chrome_trace pattern not found in profile_utils.py, skipping")


def main():
    parser = argparse.ArgumentParser(description="Patch framework for rank-0-only profiling")
    parser.add_argument("--framework", required=True, choices=["vllm", "sglang"])
    args = parser.parse_args()

    if args.framework == "vllm":
        patch_vllm()
    else:
        patch_sglang()


if __name__ == "__main__":
    main()
