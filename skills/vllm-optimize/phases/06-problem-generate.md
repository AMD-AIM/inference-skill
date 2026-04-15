# Phase 6: Problem Generation {{SKIP_LABEL}}

## Objective
Identify bottleneck kernels from profile analysis and prepare them for optimization. **Vendor library kernels (Tensile GEMM, cuBLAS, etc.) ARE valid targets** — if they dominate GPU time, we generate problems for them too.

## Steps

### 1. Check Prerequisites
```bash
if [ ! -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ]; then
    echo "ERROR: gap_analysis.json not found. Run profile-analyze phase first."
    exit 1
fi
```

### 2. Set Up Directories
```bash
mkdir -p "{{PROBLEMS_DIR}}"
mkdir -p "{{OPTIMIZED_DIR}}"
```

### 3. Identify Optimization Targets

Read gap analysis and select kernels above the optimization threshold. **Include vendor kernels** (GEMM, Attention) — they are valid optimization targets if they dominate GPU time.

```bash
python3 << 'PYEOF'
import json, os

gap_path = "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json"
threshold = float("{{OPTIMIZE_PRIORITY_THRESHOLD}}")

with open(gap_path) as f:
    gap = json.load(f)

targets = []
for k in gap.get("top_kernels", []):
    pct = k.get("pct_total", 0)
    if pct >= threshold:
        targets.append({
            "name": k["name"],
            "pct_total": pct,
            "calls": k.get("calls", 0),
            "total_us": k.get("total_us", 0),
            "avg_us": k.get("avg_us", 0),
        })

# Save problem list
output = {
    "problems": targets,
    "threshold": threshold,
    "total_kernel_time_us": gap.get("total_kernel_time_us", 0),
    "note": "Vendor kernels (GEMM, Attention) are valid optimization targets"
}

os.makedirs("{{PROBLEMS_DIR}}", exist_ok=True)
with open("{{PROBLEMS_DIR}}/problem_list.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(targets)} optimization targets (threshold: {threshold}%)")
for t in targets:
    print(f"  [{t['pct_total']:.1f}%] {t['name'][:70]}")
    print(f"         {t['calls']} calls, total {t['total_us']/1000:.1f}ms, avg {t['avg_us']:.1f}us")
PYEOF
```

### 4. Copy Agent Script to Scripts Dir

Ensure the kernel optimization agent is available:
```bash
if [ ! -f "{{SCRIPTS_DIR}}/kernel_optimize_agent.py" ]; then
    SKILL_SCRIPTS="$(dirname "$(dirname "$(readlink -f "{{SCRIPTS_DIR}}/select_gpus.py" 2>/dev/null || echo "")")")/scripts"
    if [ -f "$SKILL_SCRIPTS/kernel_optimize_agent.py" ]; then
        cp "$SKILL_SCRIPTS/kernel_optimize_agent.py" "{{SCRIPTS_DIR}}/"
        echo "Copied kernel_optimize_agent.py to SCRIPTS_DIR"
    else
        echo "WARNING: kernel_optimize_agent.py not found in skill bundle"
    fi
fi
```

## Completion
Update progress.json:
```json
{
  "phase": "problem-generate",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze", "profiling", "profile-analyze", "problem-generate"],
  "current_step": "problems generated",
  "details": {
    "problems_generated": "<N>",
    "includes_vendor_kernels": true
  }
}
```
