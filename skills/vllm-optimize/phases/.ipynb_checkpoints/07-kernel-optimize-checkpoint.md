# Phase 7: Kernel Optimization {{SKIP_LABEL}}

## Objective
Optimize GPU kernels identified as bottlenecks in the problem generation phase, using GEAK or manual kernel writing.

## GEAK Mode Selection

The optimization mode is determined by `GEAK_MODE` (from INTAKE optimization extras):
- `auto`: Auto-detect GEAK availability. Use GEAK if installed and API key available, otherwise manual.
- `full`: Use GEAK for both Triton (simple mode) and HIP/CK (kernel-url mode) kernels.
- `triton_only`: Use GEAK simple mode for Triton/ATen kernels only; skip HIP/CK optimization.
- `manual`: Write optimized kernels manually without GEAK.

## Steps

### 1. Check Prerequisites
```bash
if [ ! -f "{{PROBLEMS_DIR}}/problem_list.json" ]; then
    echo "ERROR: problem_list.json not found. Run problem-generate phase first."
    exit 1
fi
```

### 2. Detect GEAK Availability
```bash
GEAK_AVAILABLE=false
GEAK_DIR="{{GEAK_DIR}}"

if [ -d "$GEAK_DIR" ] && python3 -c "
  import sys; sys.path.insert(0, '$GEAK_DIR/src')
  from minisweagent.run.mini import app; print('GEAK available')
" 2>/dev/null; then
    GEAK_AVAILABLE=true
    echo "GEAK found at $GEAK_DIR"
else
    echo "GEAK not available"
fi

# Determine actual mode
ACTUAL_MODE="{{GEAK_MODE}}"
if [ "$ACTUAL_MODE" = "auto" ]; then
    if [ "$GEAK_AVAILABLE" = "true" ]; then
        ACTUAL_MODE="triton_only"
        echo "GEAK available: using triton_only mode"
    else
        ACTUAL_MODE="manual"
        echo "GEAK not available: using manual mode"
    fi
fi
echo "Optimization mode: $ACTUAL_MODE"
```

### 3. Create Optimized Kernels Directory
```bash
mkdir -p "{{OPTIMIZED_DIR}}"
```

### 4. Process Each Problem

For each problem in `problem_list.json`:

**If GEAK mode (triton_only or full):**
```bash
for problem_file in {{PROBLEMS_DIR}}/problem_*.json; do
    [ -f "$problem_file" ] || continue
    echo "Optimizing: $problem_file"

    python3 -c "
import json, sys, os
sys.path.insert(0, '{{GEAK_DIR}}/src')
from minisweagent.run.mini import app

with open('$problem_file') as f:
    problem = json.load(f)

result = app.run(problem)
print(f'GEAK result: {result}')
" 2>&1 || echo "GEAK optimization failed for $problem_file, skipping"
done
```

**If manual mode:**
Analyze each kernel bottleneck and create optimized implementations:

```python
import json, os

with open("{{PROBLEMS_DIR}}/problem_list.json") as f:
    problems = json.load(f)

for problem in problems.get("problems", []):
    kernel_name = problem["kernel_name"]
    priority = problem["priority"]
    pct = problem["pct_total"]

    print(f"\n[{priority.upper()}] {kernel_name} ({pct:.1f}% of total GPU time)")
    print(f"  Calls: {problem['calls']}")
    print(f"  Total time: {problem['total_us']/1000:.2f}ms")
    print(f"  Avg time per call: {problem['total_us']/max(problem['calls'],1)/1000:.2f}us")

    # Manual kernel optimization involves:
    # 1. Understanding the kernel's role in the model
    # 2. Identifying optimization opportunities (fusion, tiling, memory access)
    # 3. Writing optimized Triton or HIP kernels
    # 4. Integrating via vLLM plugin mechanism

    # Create a stub for the optimized kernel
    optimized_dir = "{{OPTIMIZED_DIR}}"
    kernel_id = kernel_name.replace(' ', '_').replace('::', '_')[:50]
    kernel_dir = os.path.join(optimized_dir, kernel_id)
    os.makedirs(kernel_dir, exist_ok=True)

    # Write optimization notes
    with open(os.path.join(kernel_dir, "optimization_notes.md"), 'w') as f:
        f.write(f"# Optimization Notes: {kernel_name}\n\n")
        f.write(f"**Priority**: {priority}\n")
        f.write(f"**% of total GPU time**: {pct:.1f}%\n")
        f.write(f"**Calls**: {problem['calls']}\n")
        f.write(f"**Total time**: {problem['total_us']/1000:.2f}ms\n\n")
        f.write("## Analysis\n\n")
        f.write("(Manual optimization analysis goes here)\n\n")
        f.write("## Optimized Implementation\n\n")
        f.write("(Implemented optimized kernel code goes here)\n")
```

### 5. Test Optimized Kernels
Use `kernel_test_runner.py` to validate each optimized kernel:
```bash
for kernel_dir in {{OPTIMIZED_DIR}}/*/; do
    [ -d "$kernel_dir" ] || continue
    echo "Testing: $kernel_dir"

    python3 {{SCRIPTS_DIR}}/kernel_test_runner.py \
        --kernel-dir "$kernel_dir" \
        --output "$kernel_dir/test_results.json" \
        2>&1 || echo "Kernel test failed for $kernel_dir"
done
```

### 6. Finalize Optimized Kernels
Use `kernel_finalize.py` to prepare kernels for integration:
```bash
python3 {{SCRIPTS_DIR}}/kernel_finalize.py \
    --optimized-dir "{{OPTIMIZED_DIR}}" \
    --output-dir "{{OPTIMIZED_DIR}}/finalized"
```

## Completion
Update progress.json:
```json
{
  "phase": "kernel-optimize",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze", "profiling", "profile-analyze", "problem-generate", "kernel-optimize"],
  "current_step": "kernel optimization complete",
  "details": {
    "problems_processed": "<N>",
    "kernels_optimized": "<M>",
    "geak_mode": "<actual_mode>"
  }
}
```