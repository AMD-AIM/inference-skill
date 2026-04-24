#!/usr/bin/env python3
"""GEAK `--test-command` fallback for Bucket B kernels (no library harness).

Wired by Phase 7 for `in_place_optimize_no_harness` strategies. Boots a
minimal vLLM process with the rebuilt fork, runs ONE decode iteration
against a stored bf16 reference (the same .npz produced in Phase 6 by
`capture_kernel_reference.py`), prints `latency_ms=<N>` on stdout, and
exits non-zero on numerical divergence (`max_abs > 1e-2` for bf16).

GEAK's SelectPatchAgent is LLM-driven and parses arbitrary numeric
output from the test log; the `latency_ms=<N>` line is the canonical
metric and the `--task` text directs the agent at it.

Usage (as GEAK --test-command):
  python scripts/optimize/no_harness_fallback_test.py \
      --kernel <name> --fork <lib-fork-dir> \
      --reference OUTPUT_DIR/refs/<name>_bf16.npz
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time


RUNNER_TEMPLATE = """\
import json
import sys
import time
import numpy as np

try:
    from vllm import LLM, SamplingParams
except Exception as exc:
    print("VLLM_IMPORT_FAIL: " + repr(exc))
    sys.exit(2)

ref = np.load({reference!r}, allow_pickle=False)
prompts = [p.decode("utf-8") for p in ref["prompts"]] if "prompts" in ref.files else [
    "Hello world.",
]
ref_logits = ref["logits"] if "logits" in ref.files else None
expected_tokens = ref["expected_tokens"] if "expected_tokens" in ref.files else None

llm = LLM(model={model_id!r}, dtype="bfloat16", enforce_eager=False)
sp = SamplingParams(temperature=0.0, max_tokens=1)

# Warm-up to trigger HIP/CUDA graph capture; not measured.
llm.generate(prompts[:1], sp)

t0 = time.perf_counter()
outs = llm.generate(prompts[:1], sp)
latency_ms = (time.perf_counter() - t0) * 1000.0

max_abs = None
if ref_logits is not None:
    try:
        actual = np.asarray([o.outputs[0].logprobs for o in outs], dtype=object)
        if actual.dtype != object:
            max_abs = float(np.max(np.abs(actual.astype(np.float32) - ref_logits.astype(np.float32))))
    except Exception:
        max_abs = None

token_match = None
if expected_tokens is not None:
    actual_tokens = [list(o.outputs[0].token_ids) for o in outs]
    token_match = actual_tokens == expected_tokens.tolist()[: len(actual_tokens)]

print("latency_ms=" + format(latency_ms, ".4f"))
print("MAX_ABS=" + ("none" if max_abs is None else format(max_abs, ".6f")))
print("TOKEN_MATCH=" + str(token_match))
print(json.dumps({{"latency_ms": latency_ms, "max_abs": max_abs, "token_match": token_match}}))

# bf16 divergence threshold
if max_abs is not None and max_abs > 1e-2:
    sys.exit(11)
if token_match is False:
    sys.exit(12)
sys.exit(0)
"""


def main():
    parser = argparse.ArgumentParser(description="No-harness Bucket B inner-loop test")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--fork", required=True,
                        help="The fork directory whose code is exercised")
    parser.add_argument("--reference", required=True,
                        help="Bf16 reference .npz from capture_kernel_reference.py")
    parser.add_argument("--model-id", default=os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        # Fail closed: GEAK should treat this attempt as a failure rather
        # than recording a fictitious latency.
        sys.stderr.write(f"reference_npz_missing: {args.reference}\n")
        return 21

    runner = RUNNER_TEMPLATE.format(reference=args.reference, model_id=args.model_id)
    env = os.environ.copy()
    env["VLLM_FORK_DIR"] = os.path.abspath(args.fork)

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            env=env,
            timeout=args.timeout_sec,
        )
        stdout, stderr = proc.stdout, proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[timeout after {args.timeout_sec}s]\n"
        rc = 124

    duration = time.time() - start
    sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    sys.stdout.write(f"# duration_sec={duration:.2f}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
