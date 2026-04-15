# Phase 6: Problem Generation {{SKIP_LABEL}}

## Objective
Generate optimization problem files from profile analysis results, identifying kernel-level bottlenecks and creating structured problem descriptions for kernel optimization.

## Steps

### 1. Check Prerequisites
Verify that gap analysis and fusion analysis artifacts exist:
```bash
if [ ! -f "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" ]; then
    echo "ERROR: gap_analysis.json not found. Run profile-analyze phase first."
    exit 1
fi

if [ ! -f "{{OUTPUT_DIR}}/results/gpu_arch.json" ]; then
    echo "ERROR: gpu_arch.json not found. Run profile-analyze phase first."
    exit 1
fi
```

### 2. Set Up Problems Directory
```bash
mkdir -p "{{PROBLEMS_DIR}}"
mkdir -p "{{OUTPUT_DIR}}/results/model_shapes"
```

### 3. Extract Model Shapes
Analyze the model to determine hidden_size, intermediate_size, and other relevant shapes for kernel optimization:
```bash
python3 << 'PYEOF'
import json, os

model_name = os.environ.get("MODEL", "")
hf_cache = os.environ.get("HF_HOME", os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface")))

try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=hf_cache)
    model_shapes = {
        "model_name": model_name,
        "hidden_size": getattr(config, "hidden_size", 4096),
        "intermediate_size": getattr(config, "intermediate_size", 11008),
        "num_attention_heads": getattr(config, "num_attention_heads", 32),
        "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
        "num_key_value_heads": getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)),
        "vocab_size": getattr(config, "vocab_size", 32000),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 4096),
    }
except Exception as e:
    print(f"WARNING: Could not extract model shapes: {e}")
    model_shapes = {
        "model_name": model_name,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
    }

with open("{{OUTPUT_DIR}}/results/model_shapes/model_shapes.json", 'w') as f:
    json.dump(model_shapes, f, indent=2)
print(json.dumps(model_shapes, indent=2))
PYEOF
```

### 4. Generate Problem Files
Use `generate_problems_vllm.py` to create structured optimization problems:
```bash
python3 {{SCRIPTS_DIR}}/generate_problems_vllm.py \
    --gap-analysis "{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json" \
    --fusion-opportunities "{{OUTPUT_DIR}}/results/gap_analysis/fusion_opportunities.json" \
    --model-shapes "{{OUTPUT_DIR}}/results/model_shapes/model_shapes.json" \
    --gpu-arch "{{OUTPUT_DIR}}/results/gpu_arch.json" \
    --framework vllm \
    --output-dir "{{PROBLEMS_DIR}}" \
    --priority-threshold "{{OPTIMIZE_PRIORITY_THRESHOLD}}"
```

If `generate_problems_vllm.py` is not available or fails, generate problems manually from gap analysis:
```python
import json, os

with open("{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json") as f:
    gap = json.load(f)

threshold = float(os.environ.get("OPTIMIZE_PRIORITY_THRESHOLD", "5.0"))
problems = []

for k in gap.get("top_kernels", []):
    if k["pct_total"] >= threshold:
        problems.append({
            "kernel_name": k["name"],
            "total_us": k["total_us"],
            "calls": k["calls"],
            "pct_total": k["pct_total"],
            "priority": "high" if k["pct_total"] >= 15 else "medium",
            "framework": "vllm"
        })

with open("{{PROBLEMS_DIR}}/problem_list.json", 'w') as f:
    json.dump({"problems": problems, "threshold": threshold}, f, indent=2)

print(f"Generated {len(problems)} problem files above {threshold}% threshold")
for p in problems:
    print(f"  [{p['priority']}] {p['kernel_name']}: {p['pct_total']:.1f}%")
```

### 5. Problem Summary
List generated problems and their priorities.

## Completion
Update progress.json:
```json
{
  "phase": "problem-generate",
  "phases_completed": ["env", "vllm-setup", "benchmark", "benchmark-analyze", "profiling", "profile-analyze", "problem-generate"],
  "current_step": "problems generated",
  "details": {
    "problems_generated": "<N>",
    "high_priority": "<H>",
    "medium_priority": "<M>"
  }
}
```