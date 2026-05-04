# workshop_LLM_for_LLM Speaker Notes

Conversational 10-minute talk track based on `workshop_LLM_for_LLM.ipynb`.

Principle: lead with the thesis, not a slide-by-slide summary.

## What the audience should remember

- Agentic AI is becoming practical for inference optimization on ROCm
- `58%`: the end-to-end throughput gain
- `91%`: the share of GPU active time spent in GEMM in the decode phase
- `measure -> analyze -> tune -> verify`: the method

## Suggested pacing

- Spend most of the time on Slides 1, 3, 5, 7 to 9, 11, and 14.
- Keep Slides 2, 4, 6, 10, and 12 short.
- Use Slide 13 as a quick handoff into the live demo.

## Slide 1. Title

"Agentic AI for model systems is quickly becoming practical. Strong models already have useful knowledge of GPU kernels, runtime behavior, profilers, and tuning workflows. And with ROCm and the AMD GPU stack exposing the right hooks, that knowledge can now be turned into measurable inference gains. That is the topic today: using an agent to optimize a real vLLM workload end to end."

## Slide 2. Workshop Roadmap

"The story is straightforward. We start from a real serving workload, let the agent locate the bottleneck, optimize that bottleneck, and verify the gain. The rest of the deck just unpacks that loop."

## Slide 3. Why Agentic Optimization Is Becoming Practical

"In this workshop, the agent lifts Qwen3-8B from 356 to 565 tokens per second on the same Radeon 7900x"

"What makes this interesting is not only the gain. It is that the workflow is now practical: the models are strong enough at systems reasoning, and the ROCm plus vLLM stack is open enough to let that reasoning act on the real bottleneck."

## Slide 4. vLLM + ROCm Inference Architecture

"Very briefly, vLLM is the serving layer and ROCm is the path down to AMD libraries and hardware. The optimization happens inside that stack; we are not changing the model architecture."

## Slide 5. Approach: The `vllm-optimize` Skill

"`vllm-optimize` packages the workflow: benchmark, profile, analyze, tune, verify, and report. The key idea is evidence-gated automation: each phase produces artifacts and constrains the next step."

"In this example, the full loop takes about 35 minutes from baseline to optimized serving configuration."

## Slide 6. Find the Bottleneck

"After the baseline, the first real question is simple: where does the GPU spend time? That question sets up everything that follows."

## Slide 7. GPU Kernel Breakdown Chart

"For Qwen3-8B at concurrency 16, the answer is clear: decode is dominated by GEMM. Once that shows up in the trace, the optimization space becomes much smaller."

## Slide 8. Diagnosis

"The number behind that chart is 91 percent of GPU active time in GEMM. That is strong enough to justify a targeted kernel path."

"This is the point where the workflow earns trust: the optimization target comes from the profile, not from intuition."

## Slide 9. Fix the Bottleneck

"The agent then uses TunableOps for offline GEMM kernel selection: record the real GEMM calls, benchmark candidates offline, and inject the best choices at serving time."

"And TunableOps is not the only path. For more complex operator fusion, we have a Triton Kernel Agent, and for more complex HIP kernel generation, users can also enable GEAK."

"The important point is that all of these optimizations are designed to be pluggable. They stay local to the measured bottleneck, without changing vLLM itself or the model structure."

## Slide 10. Kernel Optimization Result

"I would treat this as intermediate evidence: the kernel path improves. Useful, but the real question is still end-to-end throughput."

## Slide 11. Measure the Gain

"And the end-to-end result is the headline: 356 TPS to 565 TPS, a 58.4 percent gain, under the same model, hardware, and workload shape."

## Slide 12. End-to-End Throughput Chart

"This slide grounds the result in a concrete benchmark configuration. The comparison is fair and reproducible."

## Slide 13. Live Demo

"The last part is intentionally hands-on. The goal is not just to watch one demo, but to let everyone run the workflow themselves. We can launch an instance from this URL, open OpenCode, run `/vllm-optimize Qwen3-8B`, and inspect the evidence as each phase completes."

"Once you are comfortable with the flow, the next step is to repeat it on your own model or configuration. That is where this becomes useful beyond the workshop itself."
