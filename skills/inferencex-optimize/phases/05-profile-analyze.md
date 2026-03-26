# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces to identify GPU kernel-level performance bottlenecks and optimization opportunities.

{{PROFILE_ANALYSIS_NOTE}}

## IMPORTANT: Always Re-run Analysis From Scratch
When this phase is entered (including via `--from-phase profile-analyze`), **always re-run the full analysis pipeline from step 1**, even if previous analysis artifacts already exist. Delete stale results first so every run produces fresh, consistent output.

```bash
echo "Cleaning previous profile analysis artifacts..."
rm -rf "{{OUTPUT_DIR}}/results/gap_analysis"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_collective_csvs"
rm -rf "{{OUTPUT_DIR}}/results/phase_split"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
rm -f  "{{OUTPUT_DIR}}/results/profile_analysis.json"
rm -f  "{{OUTPUT_DIR}}/results/tracelens_rank0.log"
rm -f  "{{OUTPUT_DIR}}/results/tracelens_collective.log"
rm -f  "{{OUTPUT_DIR}}/results/gpu_arch.json"
echo "Cleanup done — starting fresh analysis"
```

After cleanup, verify that profile trace files (the **source** data, not previous analysis output) exist in `{{PROFILE_DIR}}/`. If no trace files exist there, print a warning and skip to step 4.

## Steps

### 1. Discover and Validate Trace Files on the Host
Locate torch profiler trace files in `{{PROFILE_DIR}}/`, filtering out async_llm (frontend-only) traces, docker logs, and benchmark result JSONs. Validate that each candidate file actually contains `traceEvents` (Chrome Trace Event format) — benchmark result JSONs (with keys like `request_throughput`, `model_id`) will cause `KeyError: 'traceEvents'` in TraceLens.

Detect trace roles explicitly:
- Generic per-rank traces by matching `*-rank-N*.json.gz` / `*-rank-N*.json` (also `rank0`, `rank1` without the dash)
- SGLang split per-rank traces by matching `*TP-N-DECODE*` and `*TP-N-EXTEND*`
- Merged/full traces by matching `merged-*.json.gz` / `merged-*.json`

For SGLang-style split traces, use:
- `RANK0_EXTEND_TRACE` as the primary single-trace input for TraceLens CSV generation
- `MERGED_TRACE` as the preferred input for phase splitting
- `TP-N-EXTEND` traces (one per rank) as the preferred basis for multi-rank collective analysis
- `WORLD_SIZE` as the number of unique per-rank IDs, not the number of trace files

**CRITICAL — DO NOT take shortcuts with trace validation.** PyTorch profiler traces place
`deviceProperties` metadata before `traceEvents`, so the key typically appears 2–5 KB
into the decompressed content. **Never** check only the first N characters/bytes — this
will incorrectly reject valid traces. The script below streams 64 KB (more than enough to
cover metadata) and performs a string search, which is fast even for 1 GB+ gzipped files.
Run it **exactly as written**:

```bash
python3 -c "
import gzip, glob, re, os

trace_dir = '{{PROFILE_DIR}}'
valid_traces = []
rank_ids = set()
tp_extend_ranks = set()
full_rank_ranks = set()
merged_trace = ''
rank0_full_trace = ''
rank0_extend_trace = ''
rank0_decode_trace = ''

PEEK_BYTES = 65536  # 64 KB decompressed — covers all metadata before traceEvents

for f in sorted(glob.glob(os.path.join(trace_dir, '*.json*'))):
    basename = os.path.basename(f)
    lower_name = basename.lower()
    if '_docker.log' in basename:
        continue
    if 'async_llm' in lower_name:
        print(f'SKIPPED (async_llm frontend trace): {f}')
        continue
    try:
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt') as fh:
            prefix = fh.read(PEEK_BYTES)
        if '\"traceEvents\"' not in prefix:
            print(f'SKIPPED (no traceEvents key in first {PEEK_BYTES} bytes): {f}')
            continue

        trace_role = 'other'
        rank = None
        phase = None

        merged_match = lower_name.startswith('merged-') or re.search(r'(^|[-_])merged(?:[-_.]|$)', lower_name)
        tp_phase_match = re.search(r'(?:^|[-_])tp[-_]?(\d+)[-_](decode|extend)(?:[-_.]|$)', basename, re.I)
        rank_match = re.search(r'(?:^|[-_])rank[-_]?(\d+)(?:[-_.]|$)', basename, re.I)

        if merged_match:
            trace_role = 'merged'
            if not merged_trace:
                merged_trace = f
        elif tp_phase_match:
            trace_role = 'tp-phase'
            rank = int(tp_phase_match.group(1))
            phase = tp_phase_match.group(2).upper()
            rank_ids.add(rank)
            if phase == 'EXTEND':
                tp_extend_ranks.add(rank)
            if rank == 0 and phase == 'EXTEND' and not rank0_extend_trace:
                rank0_extend_trace = f
            if rank == 0 and phase == 'DECODE' and not rank0_decode_trace:
                rank0_decode_trace = f
        elif rank_match:
            trace_role = 'rank'
            rank = int(rank_match.group(1))
            rank_ids.add(rank)
            full_rank_ranks.add(rank)
            if rank == 0 and not rank0_full_trace:
                rank0_full_trace = f

        valid_traces.append(f)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        descriptor = [trace_role]
        if rank is not None:
            descriptor.append(f'rank {rank}')
        if phase:
            descriptor.append(phase)
        print(f\"VALID torch trace ({', '.join(descriptor)}, {size_mb:.1f} MB compressed): {f}\")
    except Exception as e:
        print(f'ERROR reading {f}: {e}')

print(f'TRACE_COUNT={len(valid_traces)}')
if valid_traces:
    if rank0_extend_trace:
        tracelens_primary_trace = rank0_extend_trace
        tracelens_primary_role = 'rank0-extend'
    elif rank0_full_trace:
        tracelens_primary_trace = rank0_full_trace
        tracelens_primary_role = 'rank0-full'
    elif merged_trace:
        tracelens_primary_trace = merged_trace
        tracelens_primary_role = 'merged-fallback'
    else:
        tracelens_primary_trace = valid_traces[0]
        tracelens_primary_role = 'first-valid-fallback'

    if merged_trace:
        phase_split_input_trace = merged_trace
        phase_split_input_role = 'merged'
    elif rank0_full_trace:
        phase_split_input_trace = rank0_full_trace
        phase_split_input_role = 'rank0-full'
    else:
        phase_split_input_trace = ''
        phase_split_input_role = 'unavailable'

    world_size = len(rank_ids) if rank_ids else 1
    if world_size > 1 and len(tp_extend_ranks) == world_size:
        collective_trace_mode = 'tp-extend'
    elif world_size > 1 and len(full_rank_ranks) == world_size:
        collective_trace_mode = 'rank-full'
    else:
        collective_trace_mode = 'unavailable'

    print(f'WORLD_SIZE={world_size}')
    print(f'COLLECTIVE_TRACE_MODE={collective_trace_mode}')
    print(f'MERGED_TRACE={merged_trace}')
    print(f'RANK0_FULL_TRACE={rank0_full_trace}')
    print(f'RANK0_EXTEND_TRACE={rank0_extend_trace}')
    print(f'RANK0_DECODE_TRACE={rank0_decode_trace}')
    print(f'TRACELENS_PRIMARY_TRACE={tracelens_primary_trace}')
    print(f'TRACELENS_PRIMARY_ROLE={tracelens_primary_role}')
    print(f'PHASE_SPLIT_INPUT_TRACE={phase_split_input_trace}')
    print(f'PHASE_SPLIT_INPUT_ROLE={phase_split_input_role}')
else:
    print('WARNING: No valid torch profiler traces found. Trace analysis will be skipped.')
"
```

If TRACE_COUNT is 0, print a warning and skip to step 4 (bottleneck analysis using benchmark data only). Do NOT run trace analysis on files that lack `traceEvents`.

### 2. Gap Analysis (Kernel Profiling) — Primary
Run gap analysis **first** — it uses only standard Python (no external dependencies) and is the primary kernel-level analysis method.

This produces a ranked list of the most expensive GPU kernels from the profiling trace. Since `delay_iterations` and `max_iterations` in Phase 4 already capture only steady-state iterations, no additional time windowing is needed — the full trace is analyzed.

The gap analysis pipeline:
1. **Filter by category** — include only `kernel` and `gpu` events (case-insensitive substring matching), exclude `gpu_user_annotation`
2. **Aggregate per kernel** — group by kernel name, sum total CUDA time, count calls
3. **Merge across ranks** — combine stats from all rank traces into a single ranking
4. **Rank by total duration** — sort kernels by cumulative GPU time descending

**IMPORTANT**: This script processes large trace files (potentially millions of events). Set a long bash timeout (at least 600 seconds). The trace file loading step alone can take 30+ seconds for a 100MB+ gzipped trace.

The pipeline deploys `trace_analyzer.py` to `{{SCRIPTS_DIR}}/`. Use it for gap analysis:

```bash
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}" \
    --gap-analysis \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --start-pct 0 --end-pct 100 --top-k 20
```

You can also run a full (non-windowed) kernel summary:
```bash
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}"
```

The gap analysis CSV follows this format (matching InferenceX `gen_kstats_clamped_traces.py` output):
```
Name, Calls, Self CUDA total (us), Avg time (us), % Total
```

This reveals which GPU kernels dominate steady-state inference time. Typical bottleneck categories for vLLM:
- **GEMM kernels** (e.g., `ck_fmha_*`, `hipblas*`) — matrix multiply for attention and FFN layers
- **Communication kernels** (e.g., `ncclAllReduce*`, `allgather*`) — collective ops for tensor parallelism
- **Custom attention** (e.g., `paged_attention_*`, `flash_attn_*`) — KV cache operations
- **Quantization** (e.g., `dequant*`, `mxfp4_*`) — precision conversion overhead

### 3. TraceLens Analysis 
TraceLens provides additional insights beyond gap analysis (GPU timeline, operator-level breakdown, collective communication analysis). It requires external installation but is **required** — it provides GPU timeline, operator-level breakdown, and collective communication analysis that gap analysis alone cannot.

**IMPORTANT**: The `pip install` can take several minutes due to dependency compilation. Set a bash timeout of at least **300 seconds** for the install command. If it times out or fails, **retry the installation once** before reporting an error.

**Check if TraceLens is already installed, then clone (or extract tarball) and install only if missing:**
```bash
export PATH="$HOME/.local/bin:$PATH"

tracelens_cli_ready() {
    command -v TraceLens_generate_perf_report_pytorch &>/dev/null
}

if tracelens_cli_ready; then
    echo "TraceLens CLI already available"
else
    if [ ! -d "$HOME/TraceLens-internal" ]; then
        echo "Cloning TraceLens-internal..."
        if ! git clone git@github.com:AMD-AGI/TraceLens-internal.git "$HOME/TraceLens-internal"; then
            echo "Git clone failed (see error above), extracting from bundled tarball..."
            rm -rf "$HOME/TraceLens-internal"
            TARBALL="{{SCRIPTS_DIR}}/TraceLens-internal.tar.gz"
            if [ -f "$TARBALL" ]; then
                if tar xzf "$TARBALL" -C "$HOME"; then
                    echo "Extracted TraceLens-internal from tarball"
                else
                    echo "ERROR: tar extraction failed (exit code $?)"
                    echo "TRACELENS_INSTALL_FAILED=true"
                fi
            else
                echo "ERROR: TraceLens tarball not found at $TARBALL"
                echo "TRACELENS_INSTALL_FAILED=true"
            fi
        fi
    fi
    if [ -d "$HOME/TraceLens-internal" ]; then
        echo "Installing TraceLens (this may take a few minutes)..."
        pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
        hash -r
        if tracelens_cli_ready; then
            echo "TraceLens CLI installed successfully"
        else
            echo "First install attempt failed — retrying..."
            pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
            hash -r
            if tracelens_cli_ready; then
                echo "TraceLens CLI installed successfully on retry"
            else
                echo "TRACELENS_INSTALL_FAILED=true"
                echo "ERROR: TraceLens installation failed after retry"
            fi
        fi
    fi
fi
```

If the output contains `TRACELENS_INSTALL_FAILED=true` after the retry, report the installation error but still proceed to step 4 using the gap analysis data. Do NOT skip TraceLens analysis without attempting the retry. Note: user-local installs place the CLI in `~/.local/bin`, so add that directory to `PATH` before checking or invoking `TraceLens_generate_*`.

**If TraceLens is available, run the primary single-trace performance report:**
For SGLang split traces, prefer `RANK0_EXTEND_TRACE` so the TraceLens CSVs represent real rank-local compute rather than a merged multi-rank view or a decode-only stub. Fall back to `RANK0_FULL_TRACE`, then `MERGED_TRACE`, only when needed.
```bash
export PATH="$HOME/.local/bin:$PATH"

TRACELENS_PRIMARY_TRACE="<selected single-trace path from step 1>"
TRACELENS_PRIMARY_ROLE="<selected single-trace role from step 1>"
mkdir -p "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"

if [ -n "$TRACELENS_PRIMARY_TRACE" ] && [ -f "$TRACELENS_PRIMARY_TRACE" ]; then
    echo "TraceLens primary trace role: $TRACELENS_PRIMARY_ROLE"
    TraceLens_generate_perf_report_pytorch \
        --profile_json_path "$TRACELENS_PRIMARY_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
        --enable_kernel_summary \
        2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_rank0.log"

    echo "Primary single-trace report exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/"
else
    echo "No suitable single-trace input found — skipping primary TraceLens report"
fi
```

This produces CSV files including:
- `gpu_timeline.csv` — GPU activity timeline
- `ops_summary.csv` — Operator-level time breakdown
- `ops_summary_by_category.csv` — Time grouped by op category
- `kernel_summary.csv` — GPU kernel execution statistics
- `coll_analysis.csv` — Collective communication analysis

**If WORLD_SIZE > 1, also run the multi-rank collective report:**
This requires a clean per-rank trace set. `WORLD_SIZE` from step 1 is the number of unique rank IDs detected from per-rank filenames; merged traces do not count toward it. Prefer one `TP-N-EXTEND` trace per rank for SGLang split traces, otherwise fall back to one full `rank-N` trace per rank.
```bash
export PATH="$HOME/.local/bin:$PATH"

WORLD_SIZE=<unique per-rank world size from step 1>
COLLECTIVE_TRACE_MODE="<tp-extend|rank-full|unavailable from step 1>"
COLLECTIVE_TRACE_DIR="{{OUTPUT_DIR}}/results/collective_trace_inputs"

if [ "$WORLD_SIZE" -gt 1 ] && [ "$COLLECTIVE_TRACE_MODE" != "unavailable" ]; then
    rm -rf "$COLLECTIVE_TRACE_DIR"
    mkdir -p "$COLLECTIVE_TRACE_DIR"

    COLLECTIVE_TRACE_COUNT=$(python3 -c "
import glob, os, re, sys

profile_dir = sys.argv[1]
stage_dir = sys.argv[2]
mode = sys.argv[3]
selected = {}

for f in sorted(glob.glob(os.path.join(profile_dir, '*.json*'))):
    name = os.path.basename(f)
    if mode == 'tp-extend':
        match = re.search(r'(?:^|[-_])tp[-_]?(\\d+)[-_]extend(?:[-_.]|$)', name, re.I)
        if match:
            selected.setdefault(int(match.group(1)), f)
    elif mode == 'rank-full':
        if re.search(r'(^|[-_])merged(?:[-_.]|$)', name.lower()):
            continue
        if re.search(r'(?:^|[-_])tp[-_]?(\\d+)[-_](decode|extend)(?:[-_.]|$)', name, re.I):
            continue
        match = re.search(r'(?:^|[-_])rank[-_]?(\\d+)(?:[-_.]|$)', name, re.I)
        if match:
            selected.setdefault(int(match.group(1)), f)

for entry in os.listdir(stage_dir):
    path = os.path.join(stage_dir, entry)
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)

for rank, src in sorted(selected.items()):
    # TraceLens collective mode supports .json.gz when passed via --trace_pattern.
    os.symlink(src, os.path.join(stage_dir, f'rank{rank}_trace.json.gz'))

print(len(selected))
" "{{PROFILE_DIR}}" "$COLLECTIVE_TRACE_DIR" "$COLLECTIVE_TRACE_MODE")

    echo "Collective trace mode: $COLLECTIVE_TRACE_MODE"
    echo "Collective staged trace count: $COLLECTIVE_TRACE_COUNT"
    ls -lh "$COLLECTIVE_TRACE_DIR"

    if [ "$COLLECTIVE_TRACE_COUNT" -eq "$WORLD_SIZE" ]; then
        mkdir -p "{{OUTPUT_DIR}}/results/tracelens_collective_csvs"

        TraceLens_generate_multi_rank_collective_report_pytorch \
            --trace_pattern "$COLLECTIVE_TRACE_DIR/rank*_trace.json.gz" \
            --world_size "$WORLD_SIZE" \
            --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_collective_csvs" \
            2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_collective.log"
        COLLECTIVE_EXIT_CODE=${PIPESTATUS[0]}

        echo "Multi-rank collective report exit code: $COLLECTIVE_EXIT_CODE"
        ls -lh "{{OUTPUT_DIR}}/results/tracelens_collective_csvs/"
    else
        echo "Collective trace staging mismatch (expected $WORLD_SIZE traces) — skipping multi-rank collective report"
    fi
else
    echo "No clean multi-rank trace set available — skipping multi-rank collective report"
fi
```

**Phase-Split Roofline Analysis (Prefill-Decode vs Decode-Only):**

This sub-step splits the phase-split input trace into prefill-decode and decode-only phases using TraceLens-internal's `split_vllm_trace_annotation.py`, then runs the inference-specific TraceLens report with roofline analysis on each phase. Prefer `MERGED_TRACE` when available because it preserves both phases in one file; fall back to `RANK0_FULL_TRACE` only when no merged/full trace exists. Do **not** try to split `RANK0_EXTEND_TRACE` or `RANK0_DECODE_TRACE` because those traces are already phase-specific.

**Auto-detect GPU and create GPU arch JSON** (required for roofline calculations):
```bash
python3 -c "
import json, subprocess, re, os

gpu_arch_path = '{{OUTPUT_DIR}}/results/gpu_arch.json'

PLATFORM_SPECS = {
    'MI300X': {'name': 'MI300X', 'mem_bw_gbps': 5300, 'memory_gb': 192, 'max_achievable_tflops': {'matrix_fp16': 654, 'matrix_bf16': 708, 'matrix_fp32': 163, 'matrix_fp64': 81, 'matrix_fp8': 1273, 'matrix_int8': 2600, 'vector_fp16': 163, 'vector_bf16': 163, 'vector_fp32': 81, 'vector_fp64': 40}},
    'MI325X': {'name': 'MI325X', 'mem_bw_gbps': 6000, 'memory_gb': 256, 'max_achievable_tflops': {'matrix_fp16': 794, 'matrix_bf16': 843, 'matrix_fp32': 194, 'matrix_fp64': 97, 'matrix_fp8': 1519, 'matrix_int8': 3094, 'vector_fp16': 194, 'vector_bf16': 194, 'vector_fp32': 97, 'vector_fp64': 48}},
    'MI350X': {'name': 'MI350X', 'mem_bw_gbps': 6000, 'memory_gb': 288, 'max_achievable_tflops': {'matrix_fp16': 794, 'matrix_bf16': 843, 'matrix_fp32': 194, 'matrix_fp64': 97, 'matrix_fp8': 1519, 'matrix_int8': 3094, 'vector_fp16': 194, 'vector_bf16': 194, 'vector_fp32': 97, 'vector_fp64': 48}},
    'MI355X': {'name': 'MI355X', 'mem_bw_gbps': 8000, 'memory_gb': 288, 'max_achievable_tflops': {'matrix_fp16': 1686, 'matrix_bf16': 1686, 'matrix_fp32': 137, 'matrix_fp64': 68, 'matrix_fp8': 3567, 'matrix_fp6': 4574, 'matrix_fp4': 5663, 'matrix_int8': 7134, 'vector_fp16': 274, 'vector_bf16': 274, 'vector_fp32': 137, 'vector_fp64': 68}},
    'MI400': {'name': 'MI400', 'mem_bw_gbps': 19600, 'memory_gb': 432, 'max_achievable_tflops': {'matrix_fp16': 2500, 'matrix_bf16': 2500, 'matrix_fp32': 1250, 'matrix_fp64': 625, 'matrix_fp8': 20000, 'matrix_fp4': 40000, 'matrix_int8': 20000, 'vector_fp16': 625, 'vector_bf16': 625, 'vector_fp32': 312, 'vector_fp64': 156}},
}

gpu_name = None
try:
    result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=10)
    for line in result.stdout.splitlines():
        for key in PLATFORM_SPECS:
            if key.lower().replace('x', '') in line.lower().replace('x', ''):
                gpu_name = key
                break
        if gpu_name:
            break
except Exception:
    pass

if not gpu_name:
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if 'gfx' in line.lower():
                if 'gfx942' in line.lower():
                    gpu_name = 'MI300X'
                elif 'gfx950' in line.lower():
                    gpu_name = 'MI355X'
                break
    except Exception:
        pass

if gpu_name and gpu_name in PLATFORM_SPECS:
    spec = PLATFORM_SPECS[gpu_name]
    os.makedirs(os.path.dirname(gpu_arch_path), exist_ok=True)
    with open(gpu_arch_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f'GPU_ARCH_DETECTED={gpu_name}')
    print(f'GPU_ARCH_JSON={gpu_arch_path}')
else:
    print(f'GPU_ARCH_DETECTED=unknown')
    print('WARNING: Could not detect GPU model for roofline analysis. Roofline data will be omitted.')
"
```

**Split the phase-split input trace into prefill-decode and decode-only phases:**
```bash
PHASE_SPLIT_INPUT_TRACE="<phase-split input trace path from step 1>"
PHASE_SPLIT_INPUT_ROLE="<phase-split input role from step 1>"
SPLIT_SCRIPT="$HOME/TraceLens-internal/examples/custom_workflows/split_vllm_trace_annotation.py"
PHASE_SPLIT_DIR="{{OUTPUT_DIR}}/results/phase_split"

if [ -n "$PHASE_SPLIT_INPUT_TRACE" ] && [ -f "$PHASE_SPLIT_INPUT_TRACE" ] && [ -f "$SPLIT_SCRIPT" ]; then
    mkdir -p "$PHASE_SPLIT_DIR"
    echo "Splitting trace into prefill-decode and decode-only phases..."
    echo "Phase-split input role: $PHASE_SPLIT_INPUT_ROLE"
    python3 "$SPLIT_SCRIPT" "$PHASE_SPLIT_INPUT_TRACE" \
        -o "$PHASE_SPLIT_DIR" \
        --find-steady-state \
        --num-steps 32 \
        2>&1 | tail -30

    echo "Phase split exit code: $?"
    ls -lh "$PHASE_SPLIT_DIR/"
else
    echo "PHASE_SPLIT_UNAVAILABLE=true"
    echo "Skipping phase-split roofline analysis (phase-split input trace or script unavailable)"
fi
```

**Identify the phase-specific trace files** from the split output. The splitter produces files named `prefilldecode_*` and `decode_*`:
```bash
if [ -d "$PHASE_SPLIT_DIR" ] && [ "$(ls -A $PHASE_SPLIT_DIR/*.json.gz 2>/dev/null)" ]; then
    PREFILL_DECODE_TRACE=$(ls "$PHASE_SPLIT_DIR"/prefilldecode_*.json.gz 2>/dev/null | head -1)
    DECODE_ONLY_TRACE=$(ls "$PHASE_SPLIT_DIR"/decode_*.json.gz 2>/dev/null | head -1)

    echo "PREFILL_DECODE_TRACE=$PREFILL_DECODE_TRACE"
    echo "DECODE_ONLY_TRACE=$DECODE_ONLY_TRACE"
else
    echo "No phase-split traces found — skipping per-phase roofline analysis"
fi
```

**Run TraceLens inference report with roofline on the prefill-decode phase:**
```bash
INFERENCE_REPORT_SCRIPT="$HOME/TraceLens-internal/TraceLens/Reporting/generate_perf_report_pytorch_inference.py"
GPU_ARCH_JSON="{{OUTPUT_DIR}}/results/gpu_arch.json"

if [ -n "$PREFILL_DECODE_TRACE" ] && [ -f "$PREFILL_DECODE_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
    echo "Running TraceLens roofline on prefill-decode phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$PREFILL_DECODE_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Prefill-decode roofline exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs/"
else
    echo "Skipping prefill-decode roofline (trace or script unavailable)"
fi
```

**Run TraceLens inference report with roofline on the decode-only phase:**
```bash
if [ -n "$DECODE_ONLY_TRACE" ] && [ -f "$DECODE_ONLY_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
    echo "Running TraceLens roofline on decode-only phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$DECODE_ONLY_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Decode-only roofline exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs/"
else
    echo "Skipping decode-only roofline (trace or script unavailable)"
fi
```

**Parse TraceLens results** (if the reports were generated successfully):
Read the generated CSV files from `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` (and `tracelens_collective_csvs/` if present) and extract key insights. The directory name is kept for backward compatibility, but it should be interpreted as the **primary single-trace TraceLens output** selected in step 1, not blindly assumed to be a literal full rank-0 trace.
- From `ops_summary.csv`: top time-consuming operations, their cumulative GPU time
- From `kernel_summary.csv`: most expensive GPU kernels, call counts, average duration
- From `ops_summary_by_category.csv`: time distribution across categories (GEMM, attention, communication, etc.)
- From `coll_analysis.csv`: collective communication overhead and patterns
- From `gpu_timeline.csv`: GPU utilization and idle gaps

Also read the phase-specific CSV files from `tracelens_prefill_decode_csvs/` and `tracelens_decode_only_csvs/` (if present) and extract:
- From `unified_perf_summary.csv`: per-op roofline analysis (FLOPS/byte, TFLOPS/s, bound type, bound distance). Key columns: `name`, `op category`, `FLOPS/Byte`, `Compute Spec`, `TFLOPS/s_mean`, `Pct Roofline_mean`, `Kernel Time (µs)_sum`, `Percentage (%)`.
- From `SDPA_fwd.csv` / `FLASH_ATTN_fwd.csv`: attention roofline metrics per phase
- **From `GEMM.csv`** (CRITICAL for the report): per-GEMM-shape roofline data. Key columns: `name`, `param: M`, `param: N`, `param: K`, `FLOPS/Byte_first`, `TFLOPS/s_mean`, `Compute Spec`, `Pct Roofline_mean`, `Kernel Time (µs)_sum`, `name_count`. This CSV contains one row per unique GEMM shape — include ALL shapes in the report's roofline tables.
- From `gpu_timeline.csv`: GPU utilization comparison between prefill-decode and decode-only phases
- From `ops_summary_by_category.csv`: category time distribution differences between phases

**Display TraceLens results to the console** so the user can see key findings:
```bash
TRACELENS_PRIMARY_ROLE="<selected single-trace role from step 1>"

echo ""
echo "============================================"
echo "  TraceLens Analysis Summary (Primary Single Trace)"
echo "============================================"
echo "  Source role: $TRACELENS_PRIMARY_ROLE"

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/gpu_timeline.csv" ]; then
    echo ""
    echo "--- GPU Timeline ---"
    cat "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/gpu_timeline.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary_by_category.csv" ]; then
    echo ""
    echo "--- Ops Summary by Category ---"
    cat "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary_by_category.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary.csv" ]; then
    echo ""
    echo "--- Top Ops Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/kernel_summary.csv" ]; then
    echo ""
    echo "--- Top Kernel Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/kernel_summary.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/GEMM.csv" ]; then
    echo ""
    echo "--- GEMM Kernel Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/GEMM.csv"
fi

echo ""
echo "============================================"
echo "  Phase-Split Roofline Analysis"
echo "============================================"

for PHASE_LABEL in "Prefill-Decode" "Decode-Only"; do
    if [ "$PHASE_LABEL" = "Prefill-Decode" ]; then
        PHASE_DIR="{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
    else
        PHASE_DIR="{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
    fi

    if [ -d "$PHASE_DIR" ] && [ "$(ls -A $PHASE_DIR/*.csv 2>/dev/null)" ]; then
        echo ""
        echo "--- $PHASE_LABEL Phase ---"

        if [ -f "$PHASE_DIR/gpu_timeline.csv" ]; then
            echo ""
            echo "  GPU Timeline ($PHASE_LABEL):"
            cat "$PHASE_DIR/gpu_timeline.csv"
        fi

        if [ -f "$PHASE_DIR/ops_summary_by_category.csv" ]; then
            echo ""
            echo "  Ops by Category ($PHASE_LABEL):"
            cat "$PHASE_DIR/ops_summary_by_category.csv"
        fi

        if [ -f "$PHASE_DIR/unified_perf_summary.csv" ]; then
            echo ""
            echo "  Roofline / Unified Perf Summary ($PHASE_LABEL, first 25 lines):"
            head -25 "$PHASE_DIR/unified_perf_summary.csv"
        fi

        if [ -f "$PHASE_DIR/GEMM.csv" ]; then
            echo ""
            echo "  GEMM Roofline ($PHASE_LABEL, first 25 lines):"
            head -25 "$PHASE_DIR/GEMM.csv"
        fi

        for ATTN_CSV in "$PHASE_DIR/SDPA_fwd.csv" "$PHASE_DIR/FLASH_ATTN_fwd.csv"; do
            if [ -f "$ATTN_CSV" ]; then
                echo ""
                echo "  Attention Roofline ($PHASE_LABEL, first 25 lines):"
                head -25 "$ATTN_CSV"
                break
            fi
        done
    else
        echo ""
        echo "  $PHASE_LABEL phase: no roofline data available"
    fi
done

echo ""
echo "============================================"
```

Save TraceLens analysis to `{{OUTPUT_DIR}}/results/profile_analysis.json`:
```json
{
  "tracelens_version": "<version>",
  "trace_file": "<same as tracelens_primary_trace, kept for backward compatibility>",
  "trace_roles": {
    "merged_trace": "<path or null>",
    "rank0_full_trace": "<path or null>",
    "rank0_extend_trace": "<path or null>",
    "rank0_decode_trace": "<path or null>",
    "collective_trace_mode": "<tp-extend|rank-full|unavailable>",
    "tracelens_primary_trace": "<path or null>",
    "tracelens_primary_role": "<rank0-extend|rank0-full|merged-fallback|first-valid-fallback>",
    "phase_split_input_trace": "<path or null>",
    "phase_split_input_role": "<merged|rank0-full|unavailable>"
  },
  "num_ranks_analyzed": <WORLD_SIZE from step 1>,
  "top_ops": [{"name": "...", "total_time_us": ..., "pct": ...}, ...],
  "top_kernels": [{"name": "...", "calls": ..., "avg_time_us": ..., "pct": ...}, ...],
  "category_breakdown": {"gemm": ..., "attention": ..., "communication": ..., ...},
  "collective_overhead_pct": <if multi-rank>,
  "gpu_utilization_pct": <estimated from timeline>,
  "output_csv_dirs": ["tracelens_rank0_csvs/", "tracelens_collective_csvs/"],
  "gpu_arch": "<detected GPU model or null>",
  "phase_split": {
    "available": true,
    "prefill_decode_trace": "<path to prefill-decode trace or null>",
    "decode_only_trace": "<path to decode-only trace or null>",
    "execution_details": "<contents of phase_split/execution_details.json>"
  },
  "roofline": {
    "prefill_decode": {
      "available": true,
      "csv_dir": "tracelens_prefill_decode_csvs/",
      "gpu_timeline": {"computation_pct": ..., "communication_pct": ..., "idle_pct": ...},
      "category_breakdown": {"gemm": ..., "attention": ..., ...},
      "top_roofline_ops": [{"name": "...", "flops_per_byte": ..., "tflops_s": ..., "bound_type": "memory|compute", "bound_distance_pct": ...}, ...]
    },
    "decode_only": {
      "available": true,
      "csv_dir": "tracelens_decode_only_csvs/",
      "gpu_timeline": {"computation_pct": ..., "communication_pct": ..., "idle_pct": ...},
      "category_breakdown": {"gemm": ..., "attention": ..., ...},
      "top_roofline_ops": [{"name": "...", "flops_per_byte": ..., "tflops_s": ..., "bound_type": "memory|compute", "bound_distance_pct": ...}, ...]
    }
  }
}
```

### 4. Identify Profile Bottlenecks

From **gap analysis** (step 2 — always available when traces exist):
- Identify the top-K most expensive steady-state kernels from `gap_analysis.csv`
- Compare kernel time distribution across ranks (look for load imbalance)
- Identify whether bottleneck is compute-bound (GEMM-heavy) or communication-bound (collective-heavy)

From **TraceLens** results (step 3 — should always be available; fall back to gap analysis only if install failed after retry):
- Rank GPU kernels by cumulative time from `kernel_summary.csv`
- Quantify collective communication overhead from `coll_analysis.csv` (time spent in AllReduce, AllGather, etc.)
- Detect GPU idle gaps from `gpu_timeline.csv` indicating pipeline bubbles or CPU-bound phases
- Analyze time distribution across op categories from `ops_summary_by_category.csv`
- Preserve `trace_roles` semantics when describing the TraceLens source: only call it `rank-0` if the selected role is `rank0-extend` or `rank0-full`; describe merged fallbacks explicitly as merged/full-trace inputs

From **phase-split roofline analysis** (step 3 — available when trace annotations support phase detection):
- Compare GPU utilization and category breakdown between prefill-decode and decode-only phases
- Identify whether prefill is compute-bound (expected for large-batch GEMM) or decode is memory-bound (expected for single-token attention)
- From `unified_perf_summary.csv`: extract per-op roofline metrics (FLOPS/byte, TFLOPS/s, bound type, distance to roofline)
- From `GEMM.csv` / `SDPA_fwd.csv`: compare GEMM and attention arithmetic intensity between phases
- Flag ops far from the roofline ceiling as optimization opportunities (e.g., low TFLOPS/s relative to achievable peak)

Save profile bottleneck findings to `{{OUTPUT_DIR}}/results/profile_analysis.json` (merge with TraceLens data if already created in step 3).

### 5. Generate Profiling Report

Generate the profiling report at `{{REPORT_DIR}}/profiling_report.md`. This is a **standalone** report — it does NOT include benchmark results (those live in `benchmark_report.md` from Phase 3).

Read `{{OUTPUT_DIR}}/results/profile_analysis.json` and `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` to populate the report. Preserve trace-role semantics from step 1: only call a trace `rank-0` if it is actually a single-rank trace, and describe merged/full-trace inputs explicitly when they are used.

The report MUST use the following template:

```markdown
# InferenceX Profiling Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>
- **Tensor Parallelism**: <TP value>
- **Sequence Length**: ISL=<ISL>, OSL=<OSL>
- **Concurrency**: <concurrency used for profiling>

## Trace Structure
- **Per-rank traces detected**: <WORLD_SIZE>
- **Merged trace available**: <yes/no>
- **Collective trace basis**: <TP-N-EXTEND / full rank-N / unavailable>
- **Primary single-trace TraceLens input**: <rank-0 EXTEND trace / rank-0 full trace / merged fallback>
- **Phase-split input**: <merged trace / rank-0 full trace / unavailable>

## GPU Utilization

| Metric | Full Trace | Prefill-Decode | Decode-Only |
|--------|------------|----------------|-------------|
| Computation Time (%) | <from gpu_timeline.csv> | <from prefill-decode gpu_timeline.csv> | <from decode-only gpu_timeline.csv> |
| Exposed Comm Time (%) | ... | ... | ... |
| Exposed Memcpy Time (%) | ... | ... | ... |
| GPU Busy Time (%) | ... | ... | ... |
| GPU Idle Time (%) | ... | ... | ... |

## Top GPU Kernels (Steady-State)

From gap analysis of the profiled steady-state iterations:

| Rank | Kernel Name | Calls | Total Time (us) | Avg (us) | % Total |
|------|-------------|-------|-----------------|----------|---------|
| 1 | ... | ... | ... | ... | ... |

## Kernel Category Breakdown

From TraceLens primary single-trace ops_summary_by_category:

| Category | Count | Total Time (ms) | % of Kernel Time |
|----------|-------|-----------------|------------------|
| ... | ... | ... | ... |

## Phase-Split Roofline Analysis

The phase-split input trace (prefer merged/full trace, not already-split phase traces) is split into prefill-decode and decode-only phases using TraceLens-internal's `split_vllm_trace_annotation.py`, then analyzed with `generate_perf_report_pytorch_inference.py` for per-phase roofline insights against <GPU> specs (<mem_bw> GB/s HBM bandwidth, <peak_tflops> TFLOPS bf16 peak).

### Prefill-Decode Phase (<N> steps, BS=<batch_size>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from prefill-decode gpu_timeline.csv> |
| GPU Busy Time (%) | <from prefill-decode gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

**GEMM Roofline (prefill-decode)** — REQUIRED, read from `tracelens_prefill_decode_csvs/GEMM.csv`. List ALL GEMM shapes. For each row, extract `name`, `param: M`, `param: N`, `param: K`, `FLOPS/Byte_first`, `TFLOPS/s_mean`, `Compute Spec`, `Pct Roofline_mean`. Determine bound type: if `FLOPS/Byte < mem_bw / peak_tflops` → memory-bound, else → compute-bound.

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline (%) |
|---------|-------|------------|----------|------------|------------------|
| <from GEMM.csv> | <M>×<N>×<K> | <FLOPS/Byte_first> | <TFLOPS/s_mean> | <memory or compute> | <Pct Roofline_mean> |

**Attention Roofline (prefill-decode)** — if `SDPA_fwd.csv` or `FLASH_ATTN_fwd.csv` exists, include it in the same format.

**Top ops by time (prefill-decode)** — from `unified_perf_summary.csv`, list top 10 ops with their roofline metrics:

| Op Name | Category | Time (ms) | % Total | FLOPS/Byte | TFLOPS/s | Pct Roofline (%) |
|---------|----------|-----------|---------|------------|----------|------------------|
| <from unified_perf_summary.csv> | ... | ... | ... | ... | ... | ... |

### Decode-Only Phase (<N> steps, BS=<batch_size>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from decode-only gpu_timeline.csv> |
| GPU Busy Time (%) | <from decode-only gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

**GEMM Roofline (decode-only)** — REQUIRED, read from `tracelens_decode_only_csvs/GEMM.csv`. Same format as prefill-decode.

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline (%) |
|---------|-------|------------|----------|------------|------------------|
| <from GEMM.csv> | <M>×<N>×<K> | <FLOPS/Byte_first> | <TFLOPS/s_mean> | <memory or compute> | <Pct Roofline_mean> |

**Attention Roofline (decode-only)** — if `SDPA_fwd.csv` or `FLASH_ATTN_fwd.csv` exists, include it.

**Top ops by time (decode-only)** — from `unified_perf_summary.csv`, top 10 ops:

| Op Name | Category | Time (ms) | % Total | FLOPS/Byte | TFLOPS/s | Pct Roofline (%) |
|---------|----------|-----------|---------|------------|----------|------------------|
| <from unified_perf_summary.csv> | ... | ... | ... | ... | ... | ... |

### Phase Comparison

| Metric | Prefill-Decode | Decode-Only |
|--------|----------------|-------------|
| GPU Computation (%) | ... | ... |
| GPU Busy (%) | ... | ... |
| FusedMoE Time (%) | ... | ... |
| GEMM Time (%) | ... | ... |
| Attention Time (%) | ... | ... |
| RMSNorm Time (%) | ... | ... |
| Communication Time (%) | ... | ... |
| Dominant Bound | compute / memory | compute / memory |
| Batch Size | ... | ... |

## Profile Bottlenecks & Optimization Opportunities
- <bottleneck 1: description and recommendation>
- <bottleneck 2: description and recommendation>
- ...

## Raw Profile Data
- Gap analysis: `results/gap_analysis/`
- TraceLens primary single-trace CSVs: `results/tracelens_rank0_csvs/` (directory name kept for backward compatibility)
- Collective analysis staged traces: `results/collective_trace_inputs/`
- Phase-split traces: `results/phase_split/`
- Prefill-decode roofline CSVs: `results/tracelens_prefill_decode_csvs/`
- Decode-only roofline CSVs: `results/tracelens_decode_only_csvs/`
- GPU arch config: `results/gpu_arch.json`
- Profile analysis JSON: `results/profile_analysis.json`
- Profiler summary: `profiles/profiler_out_0.txt`
- Primary single-trace input: `profiles/<trace_file_name>`
- Phase-split input trace: `profiles/<phase_split_input_trace_file_name or unavailable>`
- Traces viewable at: https://ui.perfetto.dev/
```

**Print the final report path:**
```bash
echo ""
echo "============================================"
echo "  Profiling Report Generated"
echo "============================================"
echo "Report: {{REPORT_DIR}}/profiling_report.md"
echo "============================================"
```

## Completion
Update progress.json (include "profile" in phases_completed only if profiling was run):
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "gap_analysis": <true if step 2 succeeded, false otherwise>,
    "tracelens_analysis": <true if step 3 succeeded, false otherwise>,
    "phase_split_roofline": <true if phase-split roofline analysis succeeded, false otherwise>,
    "gpu_arch_detected": "<GPU model name or null>",
    "report": "{{REPORT_DIR}}/profiling_report.md"
  }
}
```
