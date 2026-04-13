#!/usr/bin/env python3
"""Inject profiler configuration into benchmark scripts.

For vLLM: injects --profiler-config.* CLI args into the 'vllm serve' command.
For SGLang: injects --disable-cuda-graph for eager-mode profiling.

Run INSIDE the Docker container:
  docker exec -e OSL=... -e CONC=... $CONTAINER python3 /tmp/inject_profiler_config.py \
      --framework vllm --target /workspace/benchmark_script.sh [--enforce-eager]
"""

import argparse
import math
import os
import re
import sys


def inject_vllm(target, enforce_eager):
    prof_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "/workspace/profiles")
    osl = int(os.environ.get("OSL", "512"))
    conc = int(os.environ.get("CONC", "32"))
    rrr = float(os.environ.get("RANDOM_RANGE_RATIO", "0.5"))

    num_prompts = conc * 10
    avg_osl = osl * (1 + rrr) / 2 if rrr < 1 else osl
    total_iters = int(num_prompts * avg_osl / conc)
    transition = int(0.9 * total_iters)

    max_iters = 256
    delay_iters = max(0, transition - max_iters // 2)

    if delay_iters + max_iters > 0.8 * total_iters:
        delay_iters = max(0, int(0.5 * total_iters) - max_iters // 2)
        print(
            f"Safety cap applied: delay_iters reduced to {delay_iters} "
            f"(total_iters={total_iters} too small for original delay)"
        )

    print(
        f"Computed profiler iterations: delay={delay_iters}, max={max_iters}  "
        f"(OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO={rrr}, "
        f"avg_osl={avg_osl:.0f}, total_est={total_iters}, transition_est={transition})"
    )

    eager_arg = "--enforce-eager " if enforce_eager else ""
    profiler_args = (
        f"{eager_arg}"
        f"--profiler-config.profiler torch "
        f"--profiler-config.torch_profiler_dir {prof_dir} "
        f"--profiler-config.torch_profiler_record_shapes True "
        f"--profiler-config.torch_profiler_with_stack True "
        f"--profiler-config.torch_profiler_with_memory False "
        f"--profiler-config.torch_profiler_with_flops False "
        f"--profiler-config.torch_profiler_use_gzip True "
        f"--profiler-config.ignore_frontend True "
        f"--profiler-config.delay_iterations {delay_iters} "
        f"--profiler-config.max_iterations {max_iters}"
    )

    with open(target) as fh:
        content = fh.read()
    content = re.sub(r"--enforce-eager\s+", "", content)
    content = re.sub(r"--profiler-config\.\S+\s+\S+\s*", "", content)
    content = re.sub(r"--ignore_frontend\s+\S+\s*", "", content)
    new_content = content.replace("vllm serve ", "vllm serve " + profiler_args + " ", 1)
    if new_content == content:
        new_content = re.sub(
            r"(vllm\s+serve\s)", r"\1" + profiler_args + " ", content, count=1
        )
    if new_content != content:
        with open(target, "w") as fh:
            fh.write(new_content)
        print(f"Patched {target} with --profiler-config.* args")
    else:
        print(f"ERROR: Could not inject profiler args into {target}")
        print('Expected "vllm serve" command not found. Manual patching required.')
        sys.exit(1)


def inject_sglang(target, enforce_eager):
    with open(target) as fh:
        content = fh.read()
    original = content

    if enforce_eager:
        content = re.sub(r"--disable-cuda-graph\s*", "", content)
        content = re.sub(r"--cuda-graph-max-bs\s+\S+\s*", "", content)
        for pattern in [r"(sglang\.launch_server\s)", r"(sglang\.launch_server$)"]:
            new_content = re.sub(
                pattern, r"\1--disable-cuda-graph ", content, count=1, flags=re.MULTILINE
            )
            if new_content != content:
                content = new_content
                break

    if content != original:
        with open(target, "w") as fh:
            fh.write(content)
        print(f"Patched {target} for SGLang profiling (eager={enforce_eager})")
    else:
        print(f"No changes needed for {target}")


def main():
    parser = argparse.ArgumentParser(description="Inject profiler config into benchmark script")
    parser.add_argument("--framework", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--target", required=True, help="Path to benchmark script")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.framework == "vllm":
        inject_vllm(args.target, args.enforce_eager)
    else:
        inject_sglang(args.target, args.enforce_eager)


if __name__ == "__main__":
    main()
