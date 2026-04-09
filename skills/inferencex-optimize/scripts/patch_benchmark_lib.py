#!/usr/bin/env python3
"""Patch benchmark_lib.sh for profiling: disable relay staging and num_prompts capping.

Run INSIDE the Docker container:
  docker exec $CONTAINER python3 /tmp/patch_benchmark_lib.py
"""

import re


def main():
    path = "/workspace/benchmarks/benchmark_lib.sh"
    with open(path) as f:
        content = f.read()

    # Disable move_profile_trace_for_relay function call (keep definition intact)
    content = re.sub(
        r"^(\s*)move_profile_trace_for_relay\s*$",
        r"\1: # move_profile_trace_for_relay (disabled)",
        content,
        flags=re.MULTILINE,
    )
    print("Disabled move_profile_trace_for_relay")

    # Disable num_prompts capping to max_concurrency (for steady-state profiling)
    content = re.sub(
        r'^(\s*)num_prompts="\$max_concurrency"',
        r'\1: # num_prompts="$max_concurrency" (disabled for steady-state profiling)',
        content,
        flags=re.MULTILINE,
    )
    print("Disabled num_prompts capping — benchmark will use original num_prompts (conc * 10)")

    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
