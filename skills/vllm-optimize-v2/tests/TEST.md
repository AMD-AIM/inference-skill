# Tests: vllm-optimize-v2 Skill Validation

## Structural Validation (no GPU required)

Validates all required files exist, have correct structure, and satisfy the 7 hard constraints from the spec.

```bash
python3 tests/validate.py
```

Verbose mode (shows all checks):
```bash
python3 tests/validate.py --verbose
```

Exit code: 0 = all checks pass, 1 = one or more failures.

### What gets checked

| Category | Checks |
|----------|--------|
| Files | All 16 required files exist and are non-empty |
| SKILL.md | name, description, phases, AMD target, constraints, modes, first-turn rule |
| INTAKE.md | Round 1 questions, modes, status messages, confirmation |
| RUNTIME.md | All 17 required variables, bootstrap, phase map, no-system-files rule |
| Phase docs | Objective, required content patterns, completion block (7 phases × N checks) |
| Scripts | Python syntax, API contract for all 5 scripts |
| Knowledge base | AMD patterns, autotune warning, debugging guide |
| Consistency | Phase names consistent, cross-file references, serving-test in Phase 4+5 |
| Spec constraints | All 7 hard constraints addressed in correct files |

---

## Script Unit Tests (GPU required for full test)

### kernel_breakdown.py — no GPU needed

```bash
# Create a minimal fake trace for testing
python3 -c "
import gzip, json, os
os.makedirs('test_out/test_trace', exist_ok=True)
events = [
  {'name': 'hipblaslt_gemm_kernel', 'cat': 'kernel', 'ph': 'X', 'dur': 500, 'ts': 0},
  {'name': 'hipblaslt_gemm_kernel', 'cat': 'kernel', 'ph': 'X', 'dur': 300, 'ts': 600},
  {'name': 'rmsnorm_kernel',        'cat': 'kernel', 'ph': 'X', 'dur': 100, 'ts': 1000},
]
data = {'traceEvents': events}
with gzip.open('test_out/test_trace/test.json.gz', 'wt') as f:
    json.dump(data, f)
print('Test trace created.')
"

python3 scripts/kernel_breakdown.py \
    --trace-dir test_out/test_trace \
    --output test_out/test_gap.json \
    --top-n 10

python3 -c "
import json
with open('test_out/test_gap.json') as f:
    d = json.load(f)
assert len(d['top_kernels']) > 0, 'no kernels found'
assert d['total_kernel_time_us'] > 0, 'total time is 0'
assert 'category_breakdown' in d, 'missing category_breakdown'
print('kernel_breakdown.py: OK')
"
```

### extract_shapes.py — no GPU needed

```bash
python3 -c "
import gzip, json, os
os.makedirs('test_out/test_trace2', exist_ok=True)
events = [
  {'name': 'aten::mm', 'cat': 'cpu_op', 'ph': 'X', 'dur': 10, 'ts': 0,
   'args': {'Input Dims': [[64, 4096], [4096, 4096]]}},
  {'name': 'aten::addmm', 'cat': 'cpu_op', 'ph': 'X', 'dur': 10, 'ts': 20,
   'args': {'Input Dims': [[64], [64, 4096], [4096, 12288]]}},
]
data = {'traceEvents': events}
with gzip.open('test_out/test_trace2/test.json.gz', 'wt') as f:
    json.dump(data, f)
"

python3 scripts/extract_shapes.py \
    --trace-dir test_out/test_trace2 \
    --output test_out/test_shapes.json

python3 -c "
import json
with open('test_out/test_shapes.json') as f:
    d = json.load(f)
assert 'benchmark_shapes' in d, 'missing benchmark_shapes'
assert 'unique_m_values' in d, 'missing unique_m_values'
print(f'extract_shapes.py: OK  ({d[\"total_shapes\"]} shapes, M values: {d[\"unique_m_values\"]})')
"
```

### select_gpus.py — no GPU needed (falls back gracefully)

```bash
python3 scripts/select_gpus.py 1
python3 scripts/select_gpus.py 2
echo "select_gpus.py: OK (above output is GPU IDs)"
```

---

## Kernel Agent Unit Tests (GPU required)

### Correctness and benchmark with synthetic kernel

```bash
mkdir -p test_out

cat > test_out/test_kernel.py << 'EOF'
# dtype: bfloat16
import torch

def reference(A, B):
    return torch.mm(A, B)

def optimized(A, B):
    # Same as reference — should pass correctness and show 1.0x speedup
    return torch.mm(A, B)
EOF

python3 scripts/kernel_agent.py correctness \
    --kernel test_out/test_kernel.py \
    --shapes '[[[64, 4096], [4096, 4096]]]'

python3 scripts/kernel_agent.py benchmark \
    --kernel test_out/test_kernel.py \
    --shapes '[[[64, 4096], [4096, 4096]]]'

python3 scripts/kernel_agent.py serving-test \
    --kernel test_out/test_kernel.py \
    --n 4096 --k 4096 \
    --m-values "1,2,4,8,16,32,64" \
    --iterations 50
```

---

## E2E Validation (requires running pipeline)

After running the full optimize pipeline with `opencode`:

```bash
opencode "Use vllm-optimize skill for Qwen/Qwen3-8B"
# ... complete the guided setup and let it run ...

# Then validate the output directory:
python3 tests/validate.py --skill-dir ./skills/vllm-optimize-v2
```

For a complete pipeline run and output validation, see the skill's GUIDE.md.

---

## Expected Pass Rate

- **Structural validation**: 100% pass required (0 failures)
- **Script unit tests**: 100% pass required
- **E2E validation**: All phase artifacts present, verdict table populated
