# Phase 4: Profiling {{SKIP_LABEL}}

## Objective
Re-run selected benchmarks with profiling enabled to capture detailed performance traces.

{{PROFILE_SKIP_NOTE}}

## Steps

### 1. Select Profiling Configs
Choose a representative subset of configs to profile (typically one concurrency level, one sequence length).
If `{{FILTER_TP}}`, `{{FILTER_CONC_START}}`/`{{FILTER_CONC_END}}`, and `{{FILTER_SEQ}}` are set, use those to narrow configs. Otherwise, pick a low-concurrency config (e.g., conc=4) with the default sequence length.

### 2. Create Profiles Directory
```bash
mkdir -p "{{PROFILE_DIR}}"
# Clean stale root-owned traces from previous runs (created by Docker as root)
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json.gz 2>/dev/null || true
sudo rm -f {{REPO_DIR}}/profiles/*.pt.trace.json 2>/dev/null || true
```

### 3. Start Persistent Profiling Container
Start **one** persistent container for all profiling runs with access to **all** GPUs. Add profiling environment variables.
GPU selection happens later at `docker exec` time (not at container start).

**3a. Detect GPU vendor:**
```bash
if [[ "$RUNNER" == mi* ]]; then
    GPU_VENDOR="amd"
else
    GPU_VENDOR="nvidia"
fi
```

**3b. Set GPU device flags (NO GPU visibility env vars):**
The container gets access to all GPUs. Visibility is restricted per-run at `docker exec` time.
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

**3c. Start container:**
```bash
CONTAINER_NAME="inferencex-profile-{{CONFIG_KEY}}"
DOCKER_RUN_LOG="{{PROFILE_DIR}}/${CONTAINER_NAME}_docker_run.log"

echo "DOCKER_RUN_LOG: $DOCKER_RUN_LOG"
echo "RUN_CMD: docker run -d --name $CONTAINER_NAME --label inferencex-pipeline=true --entrypoint /bin/bash $GPU_FLAGS --shm-size 64g --ipc=host --network=host -v {{REPO_DIR}}:/workspace -v {{HF_CACHE}}:/root/.cache/huggingface -w /workspace -e HF_HOME=/root/.cache/huggingface -e HF_HUB_CACHE=/root/.cache/huggingface/hub -e PROFILE=1 -e SGLANG_TORCH_PROFILER_DIR=/workspace/profiles -e VLLM_TORCH_PROFILER_DIR=/workspace/profiles -e VLLM_RPC_TIMEOUT=1800000 -e SGLANG_PROFILE_WITH_STACK=True -e SGLANG_PROFILE_RECORD_SHAPE=True -e SGLANG_ENABLE_PROFILER_METADATA=1 -e SGLANG_WARMUP_TIMEOUT=10000 $IMAGE -c 'sleep infinity'"

docker run -d \
    --name "$CONTAINER_NAME" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 64g \
    --ipc=host \
    --network=host \
    -v {{REPO_DIR}}:/workspace \
    -v {{HF_CACHE}}:/root/.cache/huggingface \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    -e PROFILE=1 \
    -e SGLANG_TORCH_PROFILER_DIR=/workspace/profiles \
    -e VLLM_TORCH_PROFILER_DIR=/workspace/profiles \
    -e VLLM_RPC_TIMEOUT=1800000 \
    -e SGLANG_PROFILE_WITH_STACK=True \
    -e SGLANG_PROFILE_RECORD_SHAPE=True \
    -e SGLANG_ENABLE_PROFILER_METADATA=1 \
    -e SGLANG_WARMUP_TIMEOUT=10000 \
    $IMAGE \
    -c "sleep infinity" \
    > "$DOCKER_RUN_LOG" 2>&1
EXIT_CODE=$?
echo "Container start exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker run log ==="
    tail -n 50 "$DOCKER_RUN_LOG"
fi
```

{{DRY_RUN_NOTE}}

### 3½. Restrict Trace Export to Rank 0 Only
When all TP workers write multi-GB torch traces simultaneously through the same bind-mounted filesystem, I/O contention corrupts the gzip streams (CRC failures, truncated files). Restricting trace file writes to rank 0 eliminates the contention — rank-0 alone is sufficient for TraceLens single-rank reports, gap analysis, and phase-split roofline.

**For vLLM** — skip profiler creation entirely on non-rank-0 workers (no barrier needed):
```bash
if [ "$FRAMEWORK" = "vllm" ]; then
    docker exec "$CONTAINER_NAME" python3 -c "
import re

path = '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py'
with open(path) as f:
    content = f.read()

# Only create profiler on rank 0
content = content.replace(
    'if profiler_config.profiler == \"torch\":',
    'if profiler_config.profiler == \"torch\" and self.local_rank == 0:',
)

# Don't raise error on non-rank-0 workers when start/stop is called
content = re.sub(
    r'(def profile\(self, is_start: bool = True\):\s*\n\s*)if self\.profiler is None:\s*\n\s*raise RuntimeError\(',
    r'\1if self.profiler is None:\n            return  # rank > 0: no profiler, silently skip\n        if False:\n            raise RuntimeError(',
    content,
)

with open(path, 'w') as f:
    f.write(content)
print('Patched gpu_worker.py: profiler enabled for rank 0 only')
"
fi
```

**For SGLang** — all ranks must still create profilers and participate in the `torch.distributed.barrier()` inside `_ProfilerTorch.stop()`, so we only skip the `export_chrome_trace()` call on non-rank-0 workers:
```bash
if [ "$FRAMEWORK" = "sglang" ]; then
    docker exec "$CONTAINER_NAME" python3 -c "
import glob

paths = glob.glob('/usr/local/lib/python3.*/dist-packages/sglang/srt/utils/profile_utils.py')
if not paths:
    print('WARNING: SGLang profile_utils.py not found, skipping rank-0 patch')
else:
    path = paths[0]
    with open(path) as f:
        content = f.read()

    # Only export chrome trace for rank 0 (all ranks still profile + barrier)
    old = '            self.torch_profiler.export_chrome_trace(\n                os.path.join(self.output_dir, filename)\n            )'
    new = '            if self.tp_rank == 0:\n                self.torch_profiler.export_chrome_trace(\n                    os.path.join(self.output_dir, filename)\n                )'
    if old in content:
        content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        print('Patched profile_utils.py: trace export for rank 0 only')
    else:
        print('WARNING: export_chrome_trace pattern not found in profile_utils.py, skipping')
"
fi
```

### 3a. Inject Profiler Config (framework-specific)
First restore the benchmark script to its original state (previous runs may have patched the host copy via bind mount):
```bash
cd {{REPO_DIR}} && git checkout -- "$BENCHMARK_SCRIPT" benchmarks/benchmark_lib.sh 2>/dev/null || true
```

Then apply framework-specific profiler configuration.

**If `$FRAMEWORK` is `vllm`:**

vLLM >= 0.15 requires `--profiler-config.*` CLI args on the `vllm serve` command to register the `/start_profile` and `/stop_profile` API endpoints. The `VLLM_TORCH_PROFILER_DIR` env var is deprecated; without `--profiler-config.*` args, the profiling routes are never attached and calls to `/start_profile` silently fail, producing no torch traces.

Patch the target benchmark script inside the container to inject `--profiler-config.*` args into the `vllm serve` command:
```bash
docker exec \
    -e OSL="${OSL}" -e CONC="${CONC}" -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.5}" \
    "$CONTAINER_NAME" python3 - "/workspace/$BENCHMARK_SCRIPT" <<'PYEOF'
import sys, os, re, math

target = sys.argv[1]
prof_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '/workspace/profiles')

osl = int(os.environ.get('OSL', '512'))
conc = int(os.environ.get('CONC', '32'))
rrr = float(os.environ.get('RANDOM_RANGE_RATIO', '0.5'))

# Phase-split profiling: position the window to capture both
# prefill-decode (mixed) and decode-only phases so the
# split_vllm_trace_annotation.py (--find-steady-state --num-steps 32)
# can extract both phase traces for roofline analysis.
#
# With num_prompts = conc * 10 (step 3c disables capping), the
# workload runs in ~10 "waves" of conc concurrent requests.
# Each iteration produces one token per active sequence.
#   total_iters ≈ 10 * avg_osl
# The mixed→decode-only transition occurs when the last wave
# finishes prefilling, at ~90% of total iterations.
num_prompts = conc * 10
avg_osl = osl * (1 + rrr) / 2 if rrr < 1 else osl
total_iters = int(num_prompts * avg_osl / conc)
transition = int(0.9 * total_iters)

# Profile 256 iterations centered on the estimated transition:
# ~128 mixed-phase steps + ~128 decode-only steps.
max_iters = 256
delay_iters = max(0, transition - max_iters // 2)

print(f'Computed profiler iterations: delay={delay_iters}, max={max_iters}  '
      f'(OSL={osl}, CONC={conc}, RANDOM_RANGE_RATIO={rrr}, '
      f'avg_osl={avg_osl:.0f}, total_est={total_iters}, transition_est={transition})')

profiler_args = (
    '{{ENFORCE_EAGER_ARG}}'
    '--profiler-config.profiler torch '
    '--profiler-config.torch_profiler_dir ' + prof_dir + ' '
    '--profiler-config.torch_profiler_record_shapes True '
    '--profiler-config.torch_profiler_with_stack True '
    '--profiler-config.torch_profiler_with_memory False '
    '--profiler-config.torch_profiler_with_flops False '
    '--profiler-config.torch_profiler_use_gzip True '
    '--profiler-config.ignore_frontend True '
    '--profiler-config.delay_iterations ' + str(delay_iters) + ' '
    '--profiler-config.max_iterations ' + str(max_iters)
)
with open(target) as fh:
    content = fh.read()
content = re.sub(r'--enforce-eager\s+', '', content)
content = re.sub(r'--profiler-config\.\S+\s+\S+\s*', '', content)
content = re.sub(r'--ignore_frontend\s+\S+\s*', '', content)
new_content = content.replace('vllm serve ', 'vllm serve ' + profiler_args + ' ', 1)
if new_content != content:
    with open(target, 'w') as fh:
        fh.write(new_content)
    print(f'Patched {target} with --profiler-config.* args')
else:
    print(f'No "vllm serve" found in {target}, nothing to patch')
PYEOF
```

**If `$FRAMEWORK` is `sglang`:**

SGLang profiling is controlled via environment variables (already set in the container at step 3c: `SGLANG_TORCH_PROFILER_DIR`, `SGLANG_PROFILE_WITH_STACK`, `SGLANG_PROFILE_RECORD_SHAPE`, `SGLANG_ENABLE_PROFILER_METADATA`, `SGLANG_WARMUP_TIMEOUT`). No `--profiler-config.*` CLI args are needed. The only script patching required is to inject `--disable-cuda-graph` for eager-mode profiling (SGLang's equivalent of vLLM's `--enforce-eager`) and to remove any existing `--cuda-graph-max-bs` arguments that conflict with eager mode:
```bash
docker exec \
    "$CONTAINER_NAME" python3 - "/workspace/$BENCHMARK_SCRIPT" <<'PYEOF'
import sys, re

target = sys.argv[1]
with open(target) as fh:
    content = fh.read()

original = content

enforce_eager = '{{ENFORCE_EAGER_ARG}}'.strip() != ''

if enforce_eager:
    # Remove any existing --disable-cuda-graph or --cuda-graph-max-bs flags
    content = re.sub(r'--disable-cuda-graph\s*', '', content)
    content = re.sub(r'--cuda-graph-max-bs\s+\S+\s*', '', content)

    # Inject --disable-cuda-graph into the sglang.launch_server command
    for pattern in [r'(sglang\.launch_server\s)', r'(sglang\.launch_server$)']:
        new_content = re.sub(pattern, r'\1--disable-cuda-graph ', content, count=1,
                             flags=re.MULTILINE)
        if new_content != content:
            content = new_content
            break

if content != original:
    with open(target, 'w') as fh:
        fh.write(content)
    print(f'Patched {target} for SGLang profiling (eager={enforce_eager})')
else:
    print(f'No changes needed for {target}')
PYEOF
```

NOTE: The container bind-mounts `{{REPO_DIR}}:/workspace`, so these changes affect the host repo. Step 6 cleans up generated files, and the `git checkout` above ensures a clean starting state.

### 3b. Disable Relay Trace Staging
The `move_profile_trace_for_relay()` function in `benchmark_lib.sh` copies the rank trace to the repo root as a relay file. This is for CI/CD workflows and not needed here — we collect rank traces directly from the profiles directory. Neutralize the function **call** inside the container by replacing it with a bash no-op (`:`) so the enclosing `if` block remains syntactically valid:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
import re
with open('/workspace/benchmarks/benchmark_lib.sh') as f:
    content = f.read()
# Only replace the bare function call (not the definition).
# Use word-boundary matching to avoid clobbering the 'function_name() {' definition line.
content = re.sub(
    r'^(\s*)move_profile_trace_for_relay\s*$',
    r'\1: # move_profile_trace_for_relay (disabled)',
    content,
    flags=re.MULTILINE,
)
with open('/workspace/benchmarks/benchmark_lib.sh', 'w') as f:
    f.write(content)
print('Disabled move_profile_trace_for_relay')
"
```

### 3c. Keep Full Prompt Count for Steady-State Profiling
By default `benchmark_lib.sh` caps `num_prompts` to `max_concurrency` when `PROFILE=1`, producing a single-batch run with no mixed prefill+decode steady state. Disable this cap so the benchmark sends `conc * 10` prompts, giving the profiler a continuous-flow workload with both prefill-decode and decode-only phases for phase-split roofline analysis:
```bash
docker exec "$CONTAINER_NAME" python3 -c "
import re
with open('/workspace/benchmarks/benchmark_lib.sh') as f:
    content = f.read()
content = re.sub(
    r'^(\s*)num_prompts=\"\\\$max_concurrency\"',
    r'\1: # num_prompts=\"\$max_concurrency\" (disabled for steady-state profiling)',
    content,
    flags=re.MULTILINE,
)
with open('/workspace/benchmarks/benchmark_lib.sh', 'w') as f:
    f.write(content)
print('Disabled num_prompts capping — benchmark will use original num_prompts (conc * 10)')
"
```

### 4. Run Each Profile via `docker exec`
For each selected config, run the benchmark script with profiling env vars. If GPUs were manually specified, restrict visibility; otherwise use all GPUs available in the container.

**CRITICAL — use `CUDA_VISIBLE_DEVICES`, NEVER `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`.**
`ROCR_VISIBLE_DEVICES=3` re-indexes GPU 3 as device 0 at the ROCm runtime level. The benchmark scripts then copy `ROCR_VISIBLE_DEVICES` to `HIP_VISIBLE_DEVICES` with the same value (`3`), but HIP only sees device 0 after ROCR filtering, causing `RuntimeError: No HIP GPUs are available`. `CUDA_VISIBLE_DEVICES` works correctly on both AMD (ROCm/HIP) and NVIDIA without triggering re-indexing conflicts.

```bash
MANUAL_GPUS="{{GPUS}}"
if [ -n "$MANUAL_GPUS" ]; then
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=$MANUAL_GPUS"
    echo "Using manually specified GPUs: $MANUAL_GPUS"
else
    GPU_ENV=""
fi

PROFILE_RESULT="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-ep${EP}_conc${CONC}_profile"
```

**Print the log file path and the full docker exec command to the terminal** so the user can monitor progress and reproduce the run:
```bash
echo "DOCKER_LOG: $DOCKER_RUN_LOG"
echo "RUN_CMD: docker exec $GPU_ENV -e MODEL=$MODEL -e TP=$TP -e EP_SIZE=$EP -e CONC=$CONC -e ISL=$ISL -e OSL=$OSL -e MAX_MODEL_LEN=$MAX_MODEL_LEN -e RANDOM_RANGE_RATIO=0.5 -e RESULT_FILENAME=$PROFILE_RESULT -e PRECISION=$PRECISION -e FRAMEWORK=$FRAMEWORK -e EXP_NAME=$EXP_NAME $CONTAINER_NAME /bin/bash /workspace/$BENCHMARK_SCRIPT"
```

Then start the profiling benchmark run (output appends to the container's `DOCKER_RUN_LOG`):
```bash
docker exec \
    $GPU_ENV \
    -e MODEL=$MODEL \
    -e TP=$TP \
    -e EP_SIZE=$EP \
    -e CONC=$CONC \
    -e ISL=$ISL \
    -e OSL=$OSL \
    -e MAX_MODEL_LEN=$MAX_MODEL_LEN \
    -e RANDOM_RANGE_RATIO=0.5 \
    -e RESULT_FILENAME=$PROFILE_RESULT \
    -e PRECISION=$PRECISION \
    -e FRAMEWORK=$FRAMEWORK \
    -e EXP_NAME=$EXP_NAME \
    "$CONTAINER_NAME" \
    /bin/bash /workspace/$BENCHMARK_SCRIPT \
    >> "$DOCKER_RUN_LOG" 2>&1
EXIT_CODE=$?
echo "Profile exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Last 50 lines of docker log ==="
    tail -n 50 "$DOCKER_RUN_LOG"
fi
```

IMPORTANT: The docker exec runs in the **foreground** writing stdout/stderr to the log file (no output is printed to the terminal). If the command fails (non-zero exit code), the last 50 lines of the log are printed to help diagnose the issue. On success, only the exit code line is shown.

### 5. Clean Up Container
After **all** profiling runs are complete, stop and remove the container:
```bash
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

### 6. Collect Profile Traces and Benchmark Results
Copy the **rank-0 torch profiler trace** (produced by vLLM to `VLLM_TORCH_PROFILER_DIR` or SGLang to `SGLANG_TORCH_PROFILER_DIR`) and benchmark result JSONs to the output directory, then clean up all generated files from the repo. With the rank-0-only profiler patch (step 3½), only one trace file is produced, avoiding the I/O contention that corrupts gzip streams when all TP workers write multi-GB traces simultaneously.
```bash
# Torch profiler traces written to the profiles subdirectory.
# With rank-0-only profiling (step 3½), only one rank trace is produced.
# Defensively skip any async_llm traces that may appear — they contain only
# frontend CPU scheduling and lack the GPU kernels / Input Dims needed for
# shape analysis.
for f in {{REPO_DIR}}/profiles/*.json*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *async_llm*) rm -f "$f" ;;
        *)           cp "$f" "{{PROFILE_DIR}}/" && rm -f "$f" ;;
    esac
done

# Copy InferenceX profiler summary (profiler_out_0.txt) and rename with config context
if [ -f "{{REPO_DIR}}/profiles/profiler_out_0.txt" ]; then
    cp "{{REPO_DIR}}/profiles/profiler_out_0.txt" "{{PROFILE_DIR}}/profiler_out_0.txt"
    rm -f "{{REPO_DIR}}/profiles/profiler_out_0.txt"
    echo "Collected profiler_out_0.txt"
fi

# Copy benchmark result JSONs from the repo to the output results directory
mkdir -p "{{OUTPUT_DIR}}/results"
cp {{REPO_DIR}}/results/*.json "{{OUTPUT_DIR}}/results/" 2>/dev/null || true
rm -f {{REPO_DIR}}/results/*.json 2>/dev/null || true

echo "Collected trace files:"
ls -lh "{{PROFILE_DIR}}/"
echo "Collected benchmark results:"
ls -lh "{{OUTPUT_DIR}}/results/" 2>/dev/null || echo "(none)"
```

### 7. Profile Summary
List captured trace files and their sizes.
Note: traces can be viewed at https://ui.perfetto.dev/

## Completion
Update progress.json:
```json
{
  "phase": "profile",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile"],
  "current_step": "profiling complete",
  "details": {
    "profile_runs": <N>,
    "trace_files": [<list of trace files>]
  }
}
```
