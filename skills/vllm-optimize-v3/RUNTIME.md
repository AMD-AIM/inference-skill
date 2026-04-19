# Runtime Configuration

Read this file after Round 1 answers and before executing any phase.

---

## Required Variables

Resolve ALL of these before loading any phase doc. Replace `{{VAR}}` in phase docs.

| Variable | Default | Source |
|----------|---------|--------|
| `MODEL` | (required) | User input |
| `OUTPUT_DIR` | `./vllm_opt_<model>_<timestamp>` | Intake Q2 |
| `PROFILE_DIR` | `{{OUTPUT_DIR}}/profiles` | derived |
| `RESULTS_DIR` | `{{OUTPUT_DIR}}/results` | derived |
| `REPORT_DIR` | `{{OUTPUT_DIR}}/report` | derived |
| `SCRIPTS_DIR` | `{{OUTPUT_DIR}}/scripts` | derived |
| `PROBLEMS_DIR` | `{{OUTPUT_DIR}}/problems` | derived |
| `OPTIMIZED_DIR` | `{{OUTPUT_DIR}}/optimized` | derived |
| `PROGRESS_FILE` | `{{OUTPUT_DIR}}/progress.json` | derived |
| `GPUS` | auto | Intake Q3 |
| `GPU_VENDOR` | detected | Phase 0 |
| `GPU_ARCH` | detected | Phase 0 |
| `TP` | 1 | Intake Q4 |
| `ISL` | 1024 | Intake Q5 |
| `OSL` | 1024 | Intake Q5 |
| `CONCURRENCY_LEVELS` | `4,8,16,32,64,128` | Intake Q6 |
| `DTYPE` | `bfloat16` (AMD) / `float16` (NVIDIA) | auto from GPU_VENDOR |
| `MAX_MODEL_LEN` | 4096 | fixed |
| `GPU_MEM_UTIL` | 0.90 | fixed |
| `PROFILE_ITERATIONS` | 128 | fixed |
| `OPTIMIZE_PRIORITY_THRESHOLD` | 5.0 | fixed (% of GPU time) |
| `MAX_OPTIMIZATION_ATTEMPTS` | 8 | Intake Q7 |
| `MAX_CONSECUTIVE_REJECTIONS` | 3 | Intake Q7 |
| `MODE` | `optimize` | Intake Q1 |
| `HF_ENDPOINT` | `""` | auto-detected |
| `HF_HUB_DISABLE_XET` | `""` | auto-detected |

---

## Workspace Bootstrap

Before executing any phase doc:

```bash
mkdir -p "{{OUTPUT_DIR}}" "{{PROFILE_DIR}}" "{{RESULTS_DIR}}" \
         "{{REPORT_DIR}}" "{{SCRIPTS_DIR}}" "{{PROBLEMS_DIR}}" "{{OPTIMIZED_DIR}}" \
         "{{OUTPUT_DIR}}/logs"

# Copy bundled scripts to SCRIPTS_DIR
SKILL_SCRIPTS="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/scripts"
cp "$SKILL_SCRIPTS"/*.py "{{SCRIPTS_DIR}}/"

# Write config.json
python3 -c "
import json, datetime
cfg = {
  'MODEL': '{{MODEL}}',
  'OUTPUT_DIR': '{{OUTPUT_DIR}}',
  'TP': {{TP}},
  'ISL': {{ISL}},
  'OSL': {{OSL}},
  'DTYPE': '{{DTYPE}}',
  'CONCURRENCY_LEVELS': [{{CONCURRENCY_LEVELS}}],
  'MODE': '{{MODE}}',
  'MAX_OPTIMIZATION_ATTEMPTS': {{MAX_OPTIMIZATION_ATTEMPTS}},
  'MAX_CONSECUTIVE_REJECTIONS': {{MAX_CONSECUTIVE_REJECTIONS}},
  'timestamp': datetime.datetime.now().isoformat(),
}
with open('{{OUTPUT_DIR}}/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config written.')
"

# Write initial progress.json
python3 -c "
import json
p = {'phases_completed': [], 'current_phase': None, 'status': 'running'}
with open('{{PROGRESS_FILE}}', 'w') as f:
    json.dump(p, f, indent=2)
"

# Phase logging helper — source this in every phase
# Usage:  phase_start env "Phase 0"
#         phase_done  env "GPU=gfx1100"
#         phase_fail  env "rocminfo not found"
# All phase output should be piped through:  tee "{{OUTPUT_DIR}}/logs/phase_N_name.log"
phase_start() {
    local name=$1; local n=$2
    echo ""
    echo "=== Phase ${n}: ${name} — STARTING ===" | tee -a "{{OUTPUT_DIR}}/logs/phase_${n}_${name}.log"
    python3 -c "
import json, datetime
p = json.load(open('{{PROGRESS_FILE}}'))
p['current_phase'] = '${name}'
p['status'] = 'running'
p['phase_${n}_start'] = datetime.datetime.now().isoformat()
with open('{{PROGRESS_FILE}}','w') as f: json.dump(p,f,indent=2)
" 2>/dev/null || true
}

phase_done() {
    local name=$1; local n=$2; local metric=${3:-""}
    echo "=== Phase ${n}: ${name} — DONE ${metric} ===" | tee -a "{{OUTPUT_DIR}}/logs/phase_${n}_${name}.log"
    python3 -c "
import json, datetime
p = json.load(open('{{PROGRESS_FILE}}'))
p['phases_completed'].append('${name}')
p['current_phase'] = None
p['status'] = 'idle'
p['phase_${n}_done'] = datetime.datetime.now().isoformat()
p['phase_${n}_metric'] = '${metric}'
with open('{{PROGRESS_FILE}}','w') as f: json.dump(p,f,indent=2)
" 2>/dev/null || true
}

phase_fail() {
    local name=$1; local n=$2; local reason=${3:-"unknown error"}
    echo "=== Phase ${n}: ${name} — FAILED: ${reason} ===" | tee -a "{{OUTPUT_DIR}}/logs/phase_${n}_${name}.log"
    python3 -c "
import json, datetime
p = json.load(open('{{PROGRESS_FILE}}'))
p['current_phase'] = '${name}'
p['status'] = 'failed'
p['phase_${n}_failed'] = datetime.datetime.now().isoformat()
p['phase_${n}_error'] = '${reason}'
with open('{{PROGRESS_FILE}}','w') as f: json.dump(p,f,indent=2)
" 2>/dev/null || true
    exit 1
}
```

After running bootstrap, print the output directory so it's visible:
```bash
echo "Output directory: {{OUTPUT_DIR}}"
echo "Logs:             {{OUTPUT_DIR}}/logs/"
echo "Progress:         {{PROGRESS_FILE}}"
```

---

## Phase Map

| Mode | Phases executed |
|------|----------------|
| `optimize` | 0→1→2→3→4→5→6 |
| `profile-only` | 0→1→2→3 |
| `optimize-only` | 0→1→4→5→6 (requires existing Phase 3 artifacts) |

For `optimize-only`: verify that `{{RESULTS_DIR}}/real_shapes.json` and `{{RESULTS_DIR}}/gap_analysis.json` exist before starting Phase 4.

---

## Progress Tracking

Update `{{PROGRESS_FILE}}` at the START and END of each phase:

```python
# At phase start:
{"phases_completed": [...prev], "current_phase": "env", "status": "running"}

# At phase end:
{"phases_completed": [...prev, "env"], "current_phase": null, "status": "idle",
 "details": {"phase": "env", "result": "ok", ...}}
```

---

## Execution Rules

1. Execute phases in order. Never skip a phase in the sequence.
2. If filter application produces zero targets → stop with clear error.
3. Never modify `/opt/`, `/usr/`, or pip-installed packages.
4. For `optimize-only`: read existing artifacts, do not re-run Phase 3.
5. All long-running commands: print config + log path + expected duration before starting.
6. Surface progress every 30-60 seconds during benchmark/profile runs.
7. On any phase failure: write `status: failed` to progress.json, print the error, stop.

### Critical Agent Execution Rules (non-negotiable)

8. **Never split a phase into multiple bash calls.** Each phase doc contains one `{ ... } 2>&1 | tee -a "$PHASE_LOG"` block. Execute it as a **single bash call** with a sufficiently large timeout. Splitting destroys the log and breaks the tee wrapper.
   - Phase 0: timeout 120s
   - Phase 1: timeout 600s  (server startup can take 5+ min)
   - Phase 2: timeout 3600s (bench + profiling at high concurrency can exceed 30 min)
   - Phase 3: timeout 600s
   - Phase 4: timeout 3600s (TunableOps tuning can take 30+ min)
   - Phase 5: timeout 3600s (two E2E benchmark sweeps + server restart)
   - Phase 6: timeout 120s

9. **Never use `pkill -f` patterns** to clean up vLLM processes. Kill only by PID file:
   ```bash
   kill -9 $(cat "{{OUTPUT_DIR}}/vllm.pid" 2>/dev/null) 2>/dev/null || true
   ```
   `pkill -f` scans all processes and hangs on zombie processes. EngineCore is a child of the API server PID and dies automatically when the parent is killed.

10. **Never read `/proc/{PID}/environ`** to detect environment variables of a running process. Use the `gpu_selection.txt` file written by Phase 1 instead.

---

## Bundled Assets

```
scripts/select_gpus.py          — pick least-utilized GPU(s)
scripts/extract_shapes.py       — extract real (M,K,N) from profiler traces
scripts/kernel_breakdown.py     — GPU kernel time breakdown from traces
scripts/kernel_agent.py         — optimization scaffold (setup/benchmark/correctness/accept/reject/serving-test)
scripts/gemm_patch.py           — runtime GEMM dispatch patch via meta_path import hook
references/TRITON_KNOWLEDGE.md  — AMD Triton optimization knowledge base
```

If any required script is missing, stop immediately with:
```
ERROR: Required script missing: {{SCRIPTS_DIR}}/<name>.py
The skill installation may be incomplete. Re-run install.sh.
```
