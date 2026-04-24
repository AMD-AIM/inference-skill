#!/usr/bin/env python3
"""Capture a bf16 reference for one Bucket B kernel.

Phase 6 invokes this exactly once per kernel whose final
`geak_strategy == in_place_optimize_no_harness`. Boots the BASELINE vLLM
(unmodified upstream wheel), runs a representative decode + prefill batch,
and writes:

  OUTPUT_DIR/refs/<kernel>_bf16.npz
    prompts:          (N,) array of utf-8 bytes (the prompts used)
    expected_tokens:  (N, T) integer token-id arrays (decoded baseline)
    logits:           (T_first, V) bf16 logits for the first decode step,
                      saved when vLLM exposes raw logits in the build.

The npz is consumed by `no_harness_fallback_test.py` and
`allocator_integration_test.py`.

Failures here demote the kernel to `unfeasible_record_only` with
`skip_reason: reference_capture_failed` (handled in Phase 6).

Usage:
  capture_kernel_reference.py --kernel <name> \
      --output OUTPUT_DIR/refs/<name>_bf16.npz \
      [--model-id Qwen/Qwen3.5-9B] [--n-prompts 4] [--decode-tokens 32]
"""

import argparse
import os
import sys
import time

import numpy as np


DEFAULT_PROMPTS = [
    "Explain how a transformer attention head works in two sentences.",
    "List three properties of bf16 floating point.",
    "Write a haiku about GPU kernels.",
    "Briefly compare paged attention to flash attention.",
]


def main():
    parser = argparse.ArgumentParser(description="Capture bf16 reference for Bucket B kernel")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=32)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vllm import failed: {exc!r}\n")
        return 2

    prompts = (DEFAULT_PROMPTS * ((args.n_prompts // len(DEFAULT_PROMPTS)) + 1))[: args.n_prompts]

    try:
        llm = LLM(model=args.model_id, dtype="bfloat16", enforce_eager=False)
        sp = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"vllm generate failed: {exc!r}\n")
        return 3

    expected_tokens = np.asarray(
        [list(o.outputs[0].token_ids) for o in outs], dtype=object
    )

    payload = {
        "prompts": np.asarray([p.encode("utf-8") for p in prompts]),
        "expected_tokens": expected_tokens,
    }

    # Best-effort logits capture; skipped silently when not exposed.
    try:
        logits = []
        for o in outs:
            logprobs = o.outputs[0].logprobs
            if logprobs:
                first = logprobs[0]
                arr = np.asarray(
                    [v.logprob for v in first.values()], dtype=np.float32
                )
                logits.append(arr)
        if logits:
            payload["logits"] = np.stack(logits, axis=0)
    except Exception:
        pass

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **payload)
    sys.stdout.write(
        f"captured {args.kernel}: {len(prompts)} prompts, decode_tokens={args.decode_tokens}, elapsed={elapsed:.2f}s -> {args.output}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
