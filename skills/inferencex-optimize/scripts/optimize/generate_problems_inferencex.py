#!/usr/bin/env python3
"""
Problem file generator for the InferenceX optimization pipeline.

Reads fusion opportunities, gap analysis, and optional TraceLens GEMM.csv
to auto-generate problem files for kernel optimization.

Adapted from the original generate_problems.py to work with InferenceX
profiling outputs (gap_analysis.json + TraceLens CSVs).

Usage:
    python generate_problems_inferencex.py \
        --fusion-opportunities problems/fusion_opportunities.json \
        --gap-analysis results/gap_analysis/gap_analysis.json \
        --gemm-csv results/tracelens_decode_only_csvs/GEMM.csv \
        --gpu-arch results/gpu_arch.json \
        --model-shapes problems/model_shapes.json \
        --framework vllm \
        --output-dir problems/

Part of the inferencex-optimize skill. Can be used standalone.
"""
import argparse
import ast
import csv
import json
import os
import re
import sys

csv.field_size_limit(sys.maxsize)

from classify_kernel import GEAK_MODE_MAP, SKIP_KERNEL_TYPES, classify_kernel


def generate_fusion_problems(fusions_path, model_shapes, output_dir):
    """Generate fused operator problem files."""
    if not os.path.isfile(fusions_path):
        print("No fusion_opportunities.json -- run analyze_fusion_inferencex.py first")
        return 0

    H = model_shapes.get("hidden_size", 4096)
    I = model_shapes.get("intermediate_size", 11008)

    TEMPLATES = {
        "fused_residual_rmsnorm": f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.eps = eps
    def forward(self, hidden_states, residual):
        residual = hidden_states + residual
        variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
        output = residual * torch.rsqrt(variance + self.eps)
        return (self.weight * output).to(torch.bfloat16), residual

batch_size, seq_len, hidden_size = 1, 1, {H}
def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda"),
    ]
def get_init_inputs():
    return [hidden_size]
""",
        "fused_swiglu": f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

batch_size, seq_len, intermediate_size = 1, 1, {I}
def get_inputs():
    return [torch.randn(batch_size, seq_len, intermediate_size * 2, dtype=torch.bfloat16, device="cuda")]
def get_init_inputs():
    return []
""",
    }

    with open(fusions_path) as f:
        fusions = json.load(f)
    generated = 0
    print(f"Fusion opportunities: {len(fusions)}")
    for f in fusions:
        name = f.get("name", "")
        pct = f.get("combined_percent", 0)
        priority = f.get("priority", "MEDIUM")
        fname = os.path.join(output_dir, f"problem_{name}.py")
        if os.path.exists(fname):
            print(f"  [{priority}] {name} ({pct:.1f}%) -- already exists")
            continue
        template = TEMPLATES.get(name)
        if template:
            with open(fname, "w") as fw:
                fw.write(template)
            print(f"  [{priority}] {name} ({pct:.1f}%) -> {os.path.basename(fname)}")
            generated += 1
        else:
            print(f"  [{priority}] {name} ({pct:.1f}%) -- no template, agent should create manually")
    return generated


def _extract_gpu_kernel_name(kernel_details_str):
    """Extract the actual GPU kernel name from GEMM.csv kernel_details column."""
    if not kernel_details_str:
        return ""
    # kernel_details contains a Python list repr with 'name': 'Cijk_...' entries
    match = re.search(r"'name':\s*'([^']+)'", kernel_details_str)
    return match.group(1) if match else ""


def _extract_ck_tile_params(kernel_name):
    """Extract CK tile parameters from a Cijk_* kernel name."""
    params = {}
    for key in ["MT", "MI", "WG", "ISA", "SK", "WS"]:
        match = re.search(rf"{key}(\d+(?:x\d+)*)", kernel_name)
        if match:
            params[key] = match.group(1)
    return params


def _extract_kernel_family(kernel_names):
    """Dynamically discover the GPU kernel family from a list of kernel names.

    Computes the longest common prefix (LCP) among all kernel names, then
    cleans it up by stripping trailing underscores and partial tokens.

    This works for any kernel naming convention without hardcoding:
    - Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT... -> Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs
    - kernel_moe_mxgemm_2lds_... -> kernel_moe_mxgemm_2lds
    - MoeFlatmmKernel_... -> MoeFlatmmKernel

    If only one kernel name, strips known numeric config suffixes
    (underscore + digits/x patterns that look like tile configs).

    Returns the family name string.
    """
    names = [n for n in kernel_names if n]
    if not names:
        return "unknown_kernel"

    if len(names) == 1:
        name = names[0]
        # Strip trailing numeric config params: _MT128x64x128_MI16x16x1_...
        # Find where the first _<LETTERS><DIGITS> config param starts
        # by looking for _<2-letter key><digits possibly with x>
        match = re.search(r'_[A-Z]{2}\d+(?:x\d+)*(?:_|$)', name)
        if match:
            family = name[:match.start()]
        else:
            family = name
        return family.rstrip("_") or name

    # Multiple names: compute longest common prefix
    lcp = names[0]
    for n in names[1:]:
        while not n.startswith(lcp):
            lcp = lcp[:-1]
            if not lcp:
                break
    if not lcp:
        return names[0]  # fallback to first name if no common prefix

    # Clean up: strip trailing partial token (cut at last underscore boundary)
    if lcp.endswith("_"):
        family = lcp.rstrip("_")
    else:
        # LCP may have cut mid-token; back up to last underscore
        last_us = lcp.rfind("_")
        if last_us > 0:
            family = lcp[:last_us]
        else:
            family = lcp

    return family or names[0]


def generate_gemm_problems(gemm_csv_path, gpu_arch, output_dir, priority_threshold):
    """Generate grouped GEMM problem files, one per GPU kernel family.

    Groups GEMM shapes by GPU kernel family (discovered dynamically from
    the kernel_details column in GEMM.csv), and generates a single
    multi-shape problem file per family.

    Returns (generated_count, gemm_groups) where gemm_groups is a list of
    group metadata dicts for manifest generation.
    """
    if not gemm_csv_path or not os.path.isfile(gemm_csv_path):
        print("No GEMM.csv available -- skipping GEMM problem generation")
        return 0, []

    # Load GPU specs for ridge point and roofline calculation
    ridge_point = None
    peak_tflops = None
    if gpu_arch:
        mem_bw_gbps = gpu_arch.get("mem_bw_gbps")
        peak_tflops_val = gpu_arch.get("max_achievable_tflops", {}).get("matrix_bf16")
        if mem_bw_gbps and peak_tflops_val:
            peak_tflops = peak_tflops_val
            ridge_point = peak_tflops / (mem_bw_gbps / 1000)
    if ridge_point is None:
        print("WARNING: gpu_arch missing or incomplete — skipping roofline gating for GEMM problems")

    # Parse GEMM.csv — collect all shapes with their GPU kernel names
    all_shapes = []
    with open(gemm_csv_path) as f:
        for row in csv.DictReader(f):
            try:
                op_name = row.get("name", "").strip()
                M = int(float(row.get("param: M", 0) or 0))
                N = int(float(row.get("param: N", 0) or 0))
                K = int(float(row.get("param: K", 0) or 0))
                flops_byte = float(row.get("FLOPS/Byte_first", 0) or 0)
                kernel_time = float(row.get("Kernel Time (µs)_sum", 0) or 0)
                tflops_s = float(row.get("TFLOPS/s_mean", 0) or 0)
                pct_roofline_raw = row.get("Pct Roofline_mean", "")
                if pct_roofline_raw and pct_roofline_raw.strip():
                    pct_roofline = float(pct_roofline_raw)
                elif peak_tflops and peak_tflops > 0 and tflops_s > 0:
                    pct_roofline = tflops_s / peak_tflops * 100.0
                else:
                    pct_roofline = 100
            except (ValueError, TypeError):
                continue
            if M == 0 or N == 0 or K == 0:
                continue

            # Extract actual GPU kernel name from kernel_details
            gpu_kernel = _extract_gpu_kernel_name(
                row.get("kernel_details__summarize_kernel_stats", "")
                or row.get("trunc_kernel_details", "")
            )
            ck_params = _extract_ck_tile_params(gpu_kernel) if gpu_kernel else {}
            bound = "memory" if ridge_point is not None and flops_byte < ridge_point else "compute" if ridge_point is not None else "unknown"

            all_shapes.append({
                "M": M, "N": N, "K": K,
                "pct_roofline": pct_roofline,
                "flops_byte": flops_byte,
                "tflops_s": tflops_s,
                "kernel_time_us": kernel_time,
                "bound": bound,
                "gpu_kernel": gpu_kernel,
                "ck_tile_params": ck_params,
                "op_name": op_name or "unknown_gemm",
            })

    if not all_shapes:
        print("No valid GEMM shapes found in GEMM.csv")
        return 0, []

    # Group shapes by GPU kernel family (dynamically discovered)
    # Step 1: Collect all GPU kernel names per PyTorch op, then discover families
    op_kernels = {}  # op_name -> list of gpu_kernel names
    for s in all_shapes:
        op_kernels.setdefault(s["op_name"], []).append(s["gpu_kernel"])

    # Step 2: For each op, compute kernel family from its GPU kernels
    # Then build final groups keyed by kernel family
    groups = {}  # family_name -> list of shape dicts
    shape_to_family = {}  # (M,N,K,gpu_kernel) -> family_name

    for op_name, kernels in op_kernels.items():
        unique_kernels = list(dict.fromkeys(k for k in kernels if k))
        if not unique_kernels:
            # No GPU kernel info — fallback to op name
            family = op_name
        else:
            family = _extract_kernel_family(unique_kernels)
        for k in kernels:
            shape_to_family[k if k else op_name] = family

    for s in all_shapes:
        family = shape_to_family.get(s["gpu_kernel"] if s["gpu_kernel"] else s["op_name"], "unknown_kernel")
        groups.setdefault(family, []).append(s)

    generated = 0
    gemm_groups = []
    for family_name, shapes in groups.items():
        # Sort shapes by kernel_time descending (most expensive first)
        shapes.sort(key=lambda s: -s["kernel_time_us"])
        total_time = sum(s["kernel_time_us"] for s in shapes)

        # Sanitize family name for filename: lowercase, underscores
        safe_name = family_name.lower().replace("::", "_").replace(" ", "_").replace(",", "")
        fname = os.path.join(output_dir, f"problem_{safe_name}.py")

        # Build shapes list for the problem file
        shapes_lines = []
        for s in shapes:
            gpu_k = s["gpu_kernel"][:80] if s["gpu_kernel"] else "unknown"
            shapes_lines.append(
                f'    ({s["M"]}, {s["N"]}, {s["K"]}, {s["kernel_time_us"]:.1f}, '
                f'"{gpu_k}"),'
            )
        shapes_str = "\n".join(shapes_lines)

        # Determine the PyTorch op used (for the forward() call)
        op_name = shapes[0].get("op_name", "aten::mm")

        code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

# Grouped GEMM problem for GPU kernel family: {family_name}
# Parent PyTorch op: {op_name}
# Uses F.linear (NT layout) to match framework dispatch via UnquantizedLinearMethod.apply.
# F.linear(x, w) computes x @ w.T where w is [N, K] -- this is what SGLang/vLLM actually call.
# Do NOT use torch.mm(a, b) with b=[K, N] (NN layout) -- that triggers different hipBLASLt
# kernels and GEAK will find spurious NT-layout speedups that the framework already has.
# Shapes sorted by kernel time (most expensive first)
# Format: (M, N, K, kernel_time_us, gpu_kernel_name)
SHAPES = [
{shapes_str}
]


class Model(nn.Module):
    def forward(self, x, w):
        return F.linear(x, w)


def get_inputs():
    """Return inputs for the most expensive shape (first in SHAPES)."""
    M, N, K = SHAPES[0][0], SHAPES[0][1], SHAPES[0][2]
    return [
        torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),
        torch.randn(N, K, dtype=torch.bfloat16, device="cuda"),
    ]


def get_init_inputs():
    return []


def get_all_inputs():
    """Return inputs for ALL shapes, for comprehensive benchmarking."""
    all_inputs = []
    for M, N, K, _time, _kern in SHAPES:
        all_inputs.append([
            torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),
            torch.randn(N, K, dtype=torch.bfloat16, device="cuda"),
        ])
    return all_inputs
'''

        if not os.path.exists(fname):
            with open(fname, "w") as fw:
                fw.write(code)
            generated += 1

        # Collect unique GPU kernel names
        unique_gpu_kernels = list(dict.fromkeys(
            s["gpu_kernel"] for s in shapes if s["gpu_kernel"]
        ))

        print(f"\n  Family: {family_name} ({len(shapes)} shapes, total {total_time:.0f} us)")
        for s in shapes:
            gpu_k = s["gpu_kernel"][:60] if s["gpu_kernel"] else "?"
            print(f"    {s['M']:>5d}x{s['N']:>5d}x{s['K']:>5d}  "
                  f"eff={s['pct_roofline']:.0f}%  {s['bound']:8s}  "
                  f"{s['kernel_time_us']:.0f}us  {gpu_k}")
        print(f"  -> {os.path.basename(fname)}")

        gemm_groups.append({
            "family_name": family_name,
            "op_name": shapes[0].get("op_name", ""),
            "safe_name": safe_name,
            "file": os.path.basename(fname),
            "shapes": [{
                "M": s["M"], "N": s["N"], "K": s["K"],
                "kernel_time_us": round(s["kernel_time_us"], 1),
                "pct_roofline": round(s["pct_roofline"], 1),
                "tflops_s": round(s["tflops_s"], 1) if s["tflops_s"] else None,
                "gpu_kernel": s["gpu_kernel"],
                "ck_tile_params": s["ck_tile_params"],
                "bound": s["bound"],
            } for s in shapes],
            "total_kernel_time_us": round(total_time, 1),
            "gpu_kernels": unique_gpu_kernels,
        })

    return generated, gemm_groups


def _parse_traced_dims(dims_str):
    """Parse Input Dims string from ops_unique_args.csv into a list of shape tuples.

    Example: '((705, 7168), (7168, 2112))' -> [(705, 7168), (7168, 2112)]
    Handles nested tuples, scalars, and empty dims.
    """
    if not dims_str or not dims_str.strip():
        return []
    try:
        parsed = ast.literal_eval(dims_str)
        if isinstance(parsed, tuple):
            return [s for s in parsed if isinstance(s, tuple) and len(s) > 0]
        return []
    except (ValueError, SyntaxError):
        return []


def _dtype_to_torch(dtype_str):
    """Map TraceLens dtype string to torch dtype name.

    Reads dtype strings from trace data and returns the corresponding
    torch dtype string for code generation.
    """
    mapping = {
        "c10::BFloat16": "torch.bfloat16",
        "c10::Half": "torch.float16",
        "c10::Float": "torch.float32",
        "c10::Float8_e4m3fn": "torch.float8_e4m3fn",
        "c10::Float8_e8m0fnu": "torch.float8_e8m0fnu",
        "c10::Float4_e2m1fn_x2": "torch.uint8",  # FP4 packed (2 values per byte)
        "float": "torch.float32",
        "int": "torch.int32",
        "unsigned char": "torch.uint8",
    }
    return mapping.get(dtype_str.strip("' "), "torch.bfloat16")


def _extract_base_kernel_name(full_name):
    """Extract the base function name from a full C++ mangled GPU kernel name.

    Strips 'void' prefix, template args, namespaces, and _Z mangling to get
    a clean, short function name suitable for grouping and filenames.

    Discovered dynamically from whatever kernel names appear in traces.
    """
    name = full_name.strip()
    # Strip 'void ' prefix
    if name.startswith("void "):
        name = name[5:]
    # Strip template args: first '<' to end
    tmpl_idx = name.find("<")
    if tmpl_idx > 0:
        name = name[:tmpl_idx]
    # Strip function args: first '(' to end
    paren_idx = name.find("(")
    if paren_idx > 0:
        name = name[:paren_idx]
    # Extract last component from namespace::func
    if "::" in name:
        name = name.split("::")[-1]
    # Handle _Z/_ZN mangled names: extract meaningful function name
    if name.startswith("_ZN") or name.startswith("_Z"):
        # Walk the mangled name extracting <length><identifier> segments
        segments = []
        pos = 3 if name.startswith("_ZN") else 2
        while pos < len(name):
            m = re.match(r"(\d+)", name[pos:])
            if m:
                length = int(m.group(1))
                pos += len(m.group(1))
                if pos + length <= len(name):
                    seg = name[pos:pos + length]
                    # Only keep segments that look like identifiers
                    if re.match(r"[a-zA-Z_]\w*$", seg):
                        segments.append(seg)
                pos += length
            else:
                break
        if segments:
            # Use the last meaningful segment (the function name)
            name = segments[-1]
    name = name.strip("_")
    return name if name else full_name[:60]


def _parse_op_trace_dims(trace_info):
    """Parse traced input dims/types into structured tensors by dtype category.

    Returns dict with:
        bf16_tensors: [(shape_tuple, arg_idx), ...]
        fp4_tensors: [(shape_tuple, dtype_str, arg_idx), ...]
        fp8_tensors: [(shape_tuple, dtype_str, arg_idx), ...]
        scalars: [(value_str, arg_idx), ...]
    """
    result = {"bf16_tensors": [], "fp4_tensors": [], "fp8_tensors": [], "scalars": []}
    dims_raw = trace_info.get("input_dims", "")
    types_raw = trace_info.get("input_types", "")
    concrete_raw = trace_info.get("concrete_inputs", "")
    try:
        dims = list(ast.literal_eval(dims_raw)) if dims_raw else []
    except (ValueError, SyntaxError):
        dims = []
    try:
        types_list = list(ast.literal_eval(types_raw)) if types_raw else []
    except (ValueError, SyntaxError):
        types_list = []
    try:
        concrete = list(ast.literal_eval(concrete_raw)) if concrete_raw else []
    except (ValueError, SyntaxError):
        concrete = []

    for i, dim in enumerate(dims):
        dtype = types_list[i] if i < len(types_list) else ""
        if not dim:
            # Check for scalar concrete value
            if i < len(concrete) and concrete[i] and concrete[i].strip():
                result["scalars"].append((concrete[i].strip(), i))
            continue
        dtype_lower = dtype.lower()
        if "bfloat16" in dtype_lower or "bf16" in dtype_lower:
            result["bf16_tensors"].append((tuple(dim), i))
        elif "float4" in dtype_lower or "fp4" in dtype_lower or "e2m1" in dtype_lower:
            result["fp4_tensors"].append((tuple(dim), dtype, i))
        elif "float8" in dtype_lower or "fp8" in dtype_lower or "e4m3" in dtype_lower or "e5m2" in dtype_lower:
            result["fp8_tensors"].append((tuple(dim), dtype, i))

    # Also collect scalar concrete inputs for args with empty dims
    for i, val in enumerate(concrete):
        if val and val.strip() and (i >= len(dims) or not dims[i]):
            if not any(s[1] == i for s in result["scalars"]):
                result["scalars"].append((val.strip(), i))
    return result


def _estimate_moe_gemm_roofline(kernel_group, op_trace_info, gpu_arch):
    """Estimate roofline efficiency for MoE GEMM kernel families.

    Derives FLOP count from traced tensor shapes:
    - BF16 output tensor gives N (hidden_dim or intermediate_chunk)
    - FP4_x2 weight tensor gives (num_experts, N_weight, K_packed) where K = K_packed * 2
    - top_k from concrete_inputs or 3D activation shape
    - M from batch dimension of output or activation

    Returns (roofline_efficiency, achieved_tflops, gflops_per_call, compute_spec) or Nones.
    """
    peak_tflops = gpu_arch.get("max_achievable_tflops", {}).get("matrix_fp4", 0) if gpu_arch else 0
    if not peak_tflops:
        return None, None, None, "matrix_fp4"

    parent_ops = kernel_group.get("parent_ops", [])
    kernels = kernel_group.get("kernels", [])
    if not parent_ops or not kernels:
        return None, None, None, "matrix_fp4"

    # Compute per parent_op FLOPs and aggregate
    total_weighted_tflops = 0
    total_time = 0
    total_gflops = 0

    for parent_op in parent_ops:
        ti = op_trace_info.get(parent_op, {})
        if not ti:
            continue
        parsed = _parse_op_trace_dims(ti)

        # Find batch_tokens from BF16 output tensor (largest 2D BF16 tensor)
        bf16_2d = [(s, i) for s, i in parsed["bf16_tensors"] if len(s) == 2]
        bf16_3d = [(s, i) for s, i in parsed["bf16_tensors"] if len(s) == 3]

        batch_tokens = None
        # 2D BF16 = (M, N) output for down-proj; 3D BF16 = (M, top_k, N) for up-proj
        if bf16_2d:
            largest = max(bf16_2d, key=lambda x: x[0][0] * x[0][1])
            batch_tokens = largest[0][0]
        elif bf16_3d:
            largest = max(bf16_3d, key=lambda x: x[0][0] * x[0][1] * x[0][2])
            batch_tokens = largest[0][0]

        if not batch_tokens:
            continue

        # Extract top_k from scalars or 3D BF16/FP4 activation
        top_k = None
        for val, _ in parsed["scalars"]:
            try:
                v = int(val)
                if 1 < v <= 64:  # reasonable top_k range
                    top_k = v
                    break
            except ValueError:
                pass
        if top_k is None and bf16_3d:
            top_k = bf16_3d[0][0][1]  # middle dim of (M, top_k, chunk)
        if top_k is None:
            top_k = 8  # default

        M_eff = batch_tokens * top_k

        # Get weight dimensions from FP4 3D tensor (num_experts, N_weight, K_packed)
        fp4_3d = [(s, d, i) for s, d, i in parsed["fp4_tensors"] if len(s) == 3]
        if not fp4_3d:
            continue
        # Use the weight tensor with the largest total elements
        weight = max(fp4_3d, key=lambda x: x[0][0] * x[0][1] * x[0][2])
        N_weight = weight[0][1]
        K_packed = weight[0][2]
        K = K_packed * 2  # FP4_x2 packing

        # GEMM FLOPs per call: 2 * M_eff * N_weight * K
        gflops = 2 * M_eff * N_weight * K / 1e9

        # Get kernel time for this parent_op
        op_kernels = [k for k in kernels if k["parent_op"] == parent_op]
        if not op_kernels:
            continue
        op_time = sum(k["duration_us_sum"] for k in op_kernels)
        op_count = op_kernels[0]["count"] if op_kernels else 1
        mean_time_us = op_time / op_count if op_count > 0 else op_time

        if mean_time_us <= 0:
            continue

        achieved = gflops / mean_time_us * 1e3  # TFLOPS/s
        total_weighted_tflops += achieved * op_time
        total_time += op_time
        total_gflops += gflops

    if total_time <= 0:
        return None, None, None, "matrix_fp4"

    avg_tflops = total_weighted_tflops / total_time
    eff = avg_tflops / peak_tflops * 100
    if eff > 100:
        print(f"  WARNING: MoE GEMM roofline eff {eff:.1f}% > 100% — capping at 99.0%")
        eff = 99.0
    return round(eff, 2), round(avg_tflops, 1), round(total_gflops, 2), "matrix_fp4"


def _estimate_attention_roofline(kernel_group, op_trace_info, gpu_arch):
    """Estimate roofline efficiency for attention kernel families.

    Derives FLOP count from Q, K, V tensor shapes (FP8, not packed):
    - Q: (seq_q, n_heads, d_qk)
    - V: (seq_q, n_heads, d_v)
    - seq_kv from concrete_inputs or defaults to seq_q

    Returns (roofline_efficiency, achieved_tflops, gflops_per_call, compute_spec) or Nones.
    """
    peak_tflops = gpu_arch.get("max_achievable_tflops", {}).get("matrix_fp8", 0) if gpu_arch else 0
    if not peak_tflops:
        # Fallback to BF16 peak
        peak_tflops = gpu_arch.get("max_achievable_tflops", {}).get("matrix_bf16", 0) if gpu_arch else 0
    if not peak_tflops:
        return None, None, None, "matrix_fp8"

    parent_ops = kernel_group.get("parent_ops", [])
    kernels = kernel_group.get("kernels", [])
    if not parent_ops or not kernels:
        return None, None, None, "matrix_fp8"

    # Use the primary parent op
    primary_op = parent_ops[0]
    ti = op_trace_info.get(primary_op, {})
    if not ti:
        return None, None, None, "matrix_fp8"

    parsed = _parse_op_trace_dims(ti)

    # Find Q tensor: first 3D FP8 tensor (seq_q, n_heads, d_qk)
    fp8_3d = [(s, d, i) for s, d, i in parsed["fp8_tensors"] if len(s) == 3]
    if len(fp8_3d) < 2:
        # Also check BF16 3D tensors (some attention uses BF16)
        bf16_3d = [(s, i) for s, i in parsed["bf16_tensors"] if len(s) == 3]
        if len(bf16_3d) >= 2:
            seq_q = bf16_3d[0][0][0]
            n_heads = bf16_3d[0][0][1]
            d_qk = bf16_3d[0][0][2]
            d_v = bf16_3d[1][0][2] if len(bf16_3d) > 1 else d_qk
        else:
            return None, None, None, "matrix_fp8"
    else:
        seq_q = fp8_3d[0][0][0]
        n_heads = fp8_3d[0][0][1]
        d_qk = fp8_3d[0][0][2]
        # V tensor: usually the 3rd FP8 tensor (might have different d_v)
        d_v = fp8_3d[2][0][2] if len(fp8_3d) > 2 else fp8_3d[1][0][2]

    # Determine seq_kv: check concrete_inputs for a matching scalar, or use seq_q
    seq_kv = seq_q
    for val, _ in parsed["scalars"]:
        try:
            v = int(val)
            if v == seq_q:
                seq_kv = v
                break
        except ValueError:
            try:
                v = float(val)
            except ValueError:
                pass

    # Check for causal masking from kernel family name
    family_name = kernel_group.get("family_name", "").lower()
    is_causal = "causal1" in family_name or "causal_true" in family_name

    # Attention FLOPs: QK^T + AV
    # Non-causal: 2 * n_heads * seq_q * seq_kv * (d_qk + d_v)
    # Causal: approximately half (n_heads * seq_q * (seq_q+1) * (d_qk + d_v))
    if is_causal:
        total_flops = n_heads * seq_q * (seq_q + 1) * (d_qk + d_v)
    else:
        total_flops = 2 * n_heads * seq_q * seq_kv * (d_qk + d_v)

    gflops = total_flops / 1e9

    # Kernel time: use total time and count from the kernels
    total_time_us = sum(k["duration_us_sum"] for k in kernels)
    total_count = kernels[0]["count"] if kernels else 1
    mean_time_us = total_time_us / total_count if total_count > 0 else total_time_us

    if mean_time_us <= 0:
        return None, None, None, "matrix_fp8"

    achieved = gflops / mean_time_us * 1e3  # TFLOPS/s
    eff = achieved / peak_tflops * 100

    if eff > 100:
        print(f"  WARNING: Attention roofline eff {eff:.1f}% > 100% — capping at 99.0%")
        eff = 99.0
    return round(eff, 2), round(achieved, 1), round(gflops, 2), "matrix_fp8"


def _estimate_kernel_family_roofline(kernel_group, op_trace_info, gpu_arch):
    """Estimate roofline efficiency for a kernel family based on kernel_type.

    Dispatches to type-specific estimators for GEMM-like kernels.
    Returns (roofline_efficiency, achieved_tflops, gflops_per_call, compute_spec) or all Nones.
    """
    kernel_type = kernel_group.get("kernel_type", "")
    if kernel_type == "moe_gemm":
        return _estimate_moe_gemm_roofline(kernel_group, op_trace_info, gpu_arch)
    elif kernel_type == "attention":
        return _estimate_attention_roofline(kernel_group, op_trace_info, gpu_arch)
    # Other types: no FLOP estimation
    return None, None, None, None


def generate_kernel_family_problems(kernel_summary_csv, ops_unique_args_csv,
                                    output_dir, pct_threshold, existing_families,
                                    gpu_arch=None):
    """Generate problem files for ALL significant kernel families from kernel_summary.csv.

    Groups kernels by parent cpu_op (from traces), discovers kernel families
    dynamically, and generates a problem file per family. Skips families
    already covered by GEMM.csv and non-optimizable types (communication,
    moe_sort).

    Returns (generated_count, kernel_groups) for manifest integration.
    """
    if not kernel_summary_csv or not os.path.isfile(kernel_summary_csv):
        print("No kernel_summary.csv -- skipping kernel family problem generation")
        return 0, []

    # Parse ops_unique_args.csv for input shapes/types per parent op
    op_trace_info = {}  # parent_op -> {dims, types, pct, ...}
    if ops_unique_args_csv and os.path.isfile(ops_unique_args_csv):
        with open(ops_unique_args_csv) as f:
            for row in csv.DictReader(f):
                name = row.get("name", "").strip()
                if name:
                    op_trace_info[name] = {
                        "input_dims": row.get("Input Dims", ""),
                        "input_types": row.get("Input type", ""),
                        "concrete_inputs": row.get("Concrete Inputs", ""),
                        "pct": float(row.get("Percentage (%)", 0) or 0),
                    }

    # Parse kernel_summary.csv — collect all GPU kernels
    all_kernels = []
    with open(kernel_summary_csv) as f:
        for row in csv.DictReader(f):
            try:
                parent_op = row.get("Parent cpu_op", "").strip()
                kernel_name = row.get("Kernel name", "").strip()
                dur_sum = float(row.get("Kernel duration (µs)_sum", 0) or 0)
                dur_mean = float(row.get("Kernel duration (µs)_mean", 0) or 0)
                count = int(float(row.get("Kernel duration (µs)_count", 0) or 0))
                pct = float(row.get("Percent of kernels time (%)", 0) or 0)
            except (ValueError, TypeError):
                continue
            if pct < 0.1 or not kernel_name:
                continue

            # Classify kernel — use both kernel name and parent op for accuracy
            kernel_type, _ = classify_kernel(kernel_name)
            # Refine classification using parent op when kernel name is generic
            parent_type, _ = classify_kernel(parent_op)
            if parent_type in SKIP_KERNEL_TYPES:
                continue
            if kernel_type in SKIP_KERNEL_TYPES:
                continue

            all_kernels.append({
                "parent_op": parent_op,
                "kernel_name": kernel_name,
                "base_kernel_name": _extract_base_kernel_name(kernel_name),
                "kernel_type": kernel_type if kernel_type != "other" and kernel_type != "ck" else parent_type,
                "duration_us_sum": dur_sum,
                "duration_us_mean": dur_mean,
                "count": count,
                "pct": pct,
            })

    if not all_kernels:
        print("No optimizable kernels found in kernel_summary.csv")
        return 0, []

    # Group kernels by parent_op (natural grouping from traces)
    # Then compute kernel family from base kernel names within each group
    parent_groups = {}
    for k in all_kernels:
        # When parent is hipGraphLaunch (CUDA graph), group by base kernel
        # name instead — all graph children share the same parent_op.
        if k["parent_op"] == "hipGraphLaunch":
            group_key = k["base_kernel_name"]
        else:
            group_key = k["parent_op"]
        parent_groups.setdefault(group_key, []).append(k)

    # Merge parent_ops that share the same base kernel family
    # e.g., ck_moe_stage1 and ck_moe_stage2 both dispatch kernel_moe_mxgemm_2lds
    family_map = {}  # family_name -> list of kernel dicts
    for parent_op, kernels in parent_groups.items():
        base_names = list(dict.fromkeys(k["base_kernel_name"] for k in kernels))
        family = _extract_kernel_family(base_names)
        family_map.setdefault(family, []).extend(kernels)

    # Filter: skip families already covered by GEMM.csv and below threshold
    existing_lower = {f.lower() for f in existing_families}

    generated = 0
    kernel_groups = []
    for family_name, kernels in family_map.items():
        safe_name = family_name.lower().replace("::", "_").replace(" ", "_").replace(",", "").replace("<", "").replace(">", "")
        # Truncate safe_name to reasonable length for filenames
        if len(safe_name) > 60:
            safe_name = safe_name[:60].rstrip("_")

        # Skip if this family is already covered by GEMM problem files
        if safe_name in existing_lower:
            continue

        total_pct = sum(k["pct"] for k in kernels)
        if total_pct < pct_threshold:
            continue

        # Sort by duration descending
        kernels.sort(key=lambda k: -k["duration_us_sum"])
        total_time = sum(k["duration_us_sum"] for k in kernels)
        kernel_type = kernels[0]["kernel_type"]

        geak_mode, geak_config = GEAK_MODE_MAP.get(kernel_type, ("kernel-url", "mini_kernel.yaml"))
        if geak_mode == "skip":
            continue

        # Collect parent ops and their trace info
        parent_ops = list(dict.fromkeys(k["parent_op"] for k in kernels))
        trace_info = {p: op_trace_info.get(p, {}) for p in parent_ops if p in op_trace_info}

        fname = os.path.join(output_dir, f"problem_{safe_name}.py")

        # Build KERNELS list for the problem file
        kernels_lines = []
        unique_kernel_names = []
        for k in kernels:
            kn_display = k["base_kernel_name"]
            kernels_lines.append(
                f'    ("{k["parent_op"]}", {k["duration_us_sum"]:.1f}, {k["pct"]:.2f}, '
                f'{k["count"]}, "{kn_display}"),'
            )
            if k["kernel_name"] not in unique_kernel_names:
                unique_kernel_names.append(k["kernel_name"])

        kernels_str = "\n".join(kernels_lines)

        # Build input tensor generation from traced shapes
        get_inputs_lines = []
        traced_dims_comment = []
        primary_parent = parent_ops[0] if parent_ops else ""
        ti = trace_info.get(primary_parent, {})
        if ti:
            dims = _parse_traced_dims(ti.get("input_dims", ""))
            types_raw = ti.get("input_types", "")
            # Parse types list
            try:
                types_list = list(ast.literal_eval(types_raw)) if types_raw else []
            except (ValueError, SyntaxError):
                types_list = []

            traced_dims_comment.append(f"# Input dims from trace: {ti.get('input_dims', '')[:120]}")
            traced_dims_comment.append(f"# Input types from trace: {types_raw[:120]}")

            for i, dim in enumerate(dims):
                dtype_str = types_list[i] if i < len(types_list) else ""
                torch_dtype = _dtype_to_torch(dtype_str)
                shape_str = ", ".join(str(d) for d in dim)
                get_inputs_lines.append(
                    f'        torch.randn({shape_str}, dtype={torch_dtype}, device="cuda"),'
                )

        if not get_inputs_lines:
            get_inputs_lines.append(
                '        torch.randn(1, 1, dtype=torch.bfloat16, device="cuda"),  # placeholder'
            )

        get_inputs_str = "\n".join(get_inputs_lines)
        traced_dims_str = "\n".join(traced_dims_comment) if traced_dims_comment else "# No traced input dims available"

        # Build parent ops comment
        parent_ops_str = ", ".join(parent_ops)

        code = f'''import torch
import torch.nn as nn

# GPU Kernel Family: {family_name}
# Parent ops: {parent_ops_str}
# Total GPU time: {total_time:.0f} us ({total_pct:.1f}% of GPU kernels time)
# Kernel type: {kernel_type} | GEAK mode: {geak_mode}
# Kernel variants sorted by total duration (most expensive first)
# Format: (parent_op, duration_us_sum, pct, count, gpu_kernel_name)
KERNELS = [
{kernels_str}
]

{traced_dims_str}


class Model(nn.Module):
    def forward(self, *inputs):
        # Parent op: {primary_parent}
        # Use kernel-url GEAK mode to optimize the GPU kernel source directly.
        # The actual invocation depends on the framework dispatch path.
        return inputs[0]


def get_inputs():
    """Traced input shapes for {primary_parent}."""
    return [
{get_inputs_str}
    ]


def get_init_inputs():
    return []
'''

        if not os.path.exists(fname):
            with open(fname, "w") as fw:
                fw.write(code)
            generated += 1

        print(f"\n  Family: {family_name} ({len(kernels)} variants, {total_pct:.1f}%, {total_time:.0f} us)")
        for k in kernels:
            kn = k["base_kernel_name"][:50]
            print(f"    {k['parent_op']:35s}  {k['pct']:5.2f}%  {k['duration_us_sum']:.0f}us  {kn}")
        print(f"  -> {os.path.basename(fname)}")

        grp = {
            "family_name": family_name,
            "safe_name": safe_name,
            "file": os.path.basename(fname),
            "kernel_type": kernel_type,
            "parent_ops": parent_ops,
            "total_pct": round(total_pct, 2),
            "total_kernel_time_us": round(total_time, 1),
            "gpu_kernels": unique_kernel_names,
            "kernels": [{
                "parent_op": k["parent_op"],
                "kernel_name": k["kernel_name"],
                "base_kernel_name": k["base_kernel_name"],
                "duration_us_sum": round(k["duration_us_sum"], 1),
                "pct": round(k["pct"], 2),
                "count": k["count"],
            } for k in kernels],
        }

        # Estimate roofline efficiency for GEMM-like kernel families
        eff, tflops, gflops, spec = _estimate_kernel_family_roofline(grp, op_trace_info, gpu_arch)
        if eff is not None:
            grp["roofline_efficiency"] = eff
            grp["achieved_tflops"] = tflops
            grp["gflops_per_call"] = gflops
            grp["compute_spec"] = spec
            print(f"  Roofline: {eff:.1f}% ({tflops:.0f} TFLOPS/s, {gflops:.1f} GFLOP/call, spec={spec})")

        kernel_groups.append(grp)

    return generated, kernel_groups


def generate_manifest(output_dir, fusions_path, bottlenecks_path, framework,
                      roofline_entries=None, gemm_groups=None, kernel_groups=None,
                      total_gpu_kernel_time_us=None, top_n=5):
    """Generate optimization_manifest.json."""
    manifest = {
        "framework": framework,
        "description": "Optimization manifest. All enabled by default; speedup > 1.0x filter in Phase 7 is the gate.",
        "optimizations": [],
    }

    # Add fusion problems — check already_fused flag to set correct geak_mode
    if os.path.isfile(fusions_path):
        with open(fusions_path) as _fh:
            _fusions_data = json.load(_fh)
        for f in _fusions_data:
            name = f.get("name", "")
            problem_file = f"problem_{name}.py"
            if os.path.isfile(os.path.join(output_dir, problem_file)):
                already_fused = f.get("already_fused", False)
                if already_fused:
                    # Already fused by vendor kernel — use kernel-url to optimize it
                    fused_kernel = f.get("fused_kernel", "")
                    fk_type, _ = classify_kernel(fused_kernel)
                    geak_mode, geak_config = GEAK_MODE_MAP.get(fk_type, ("kernel-url", "mini_kernel.yaml"))
                    manifest["optimizations"].append({
                        "name": name,
                        "file": problem_file,
                        "type": "fused",
                        "priority": f.get("priority", "MEDIUM"),
                        "kernel_type": fk_type,
                        "geak_mode": geak_mode,
                        "geak_config": geak_config,
                        "already_fused": True,
                        "fused_kernel": fused_kernel,
                        "fused_kernel_pct": f.get("fused_kernel_pct", 0),
                        "enabled": True,
                    })
                else:
                    # Not already fused — only use simple mode if triton pattern
                    manifest["optimizations"].append({
                        "name": name,
                        "file": problem_file,
                        "type": "fused",
                        "priority": f.get("priority", "MEDIUM"),
                        "kernel_type": "triton",
                        "geak_mode": "simple",
                        "geak_config": "geak.yaml",
                        "enabled": True,
                    })

    # Add grouped GEMM problems (one entry per GPU kernel family, not per shape)
    if gemm_groups:
        for grp in gemm_groups:
            # Determine kernel_type from actual GPU kernels
            gpu_kernels = grp.get("gpu_kernels", [])
            # If we have resolved GPU kernel names, these are vendor kernels (not aten)
            kernel_type = "vendor" if gpu_kernels else "aten_gemm"
            geak_mode, geak_config = GEAK_MODE_MAP.get(kernel_type, ("kernel-url", "mini_kernel.yaml"))
            manifest["optimizations"].append({
                "name": grp["safe_name"],
                "file": grp["file"],
                "type": "gemm",
                "priority": "HIGH" if len(grp["shapes"]) > 3 else "MEDIUM",
                "kernel_type": kernel_type,
                "geak_mode": geak_mode,
                "geak_config": geak_config,
                "family_name": grp.get("family_name", ""),
                "op_name": grp.get("op_name", ""),
                "shapes": grp["shapes"],
                "gpu_kernels": gpu_kernels,
                "total_kernel_time_us": grp["total_kernel_time_us"],
                "enabled": True,
            })

    # Add kernel family problems (MoE GEMM, attention, normalization, etc.)
    if kernel_groups:
        existing_files = {o["file"] for o in manifest["optimizations"]}
        for grp in kernel_groups:
            if grp["file"] in existing_files:
                continue
            geak_mode, geak_config = GEAK_MODE_MAP.get(grp["kernel_type"], ("kernel-url", "mini_kernel.yaml"))
            manifest["optimizations"].append({
                "name": grp["safe_name"],
                "file": grp["file"],
                "type": "kernel_family",
                "priority": "HIGH" if grp["total_pct"] > 5 else "MEDIUM" if grp["total_pct"] > 2 else "LOW",
                "kernel_type": grp["kernel_type"],
                "geak_mode": geak_mode,
                "geak_config": geak_config,
                "family_name": grp["family_name"],
                "parent_ops": grp["parent_ops"],
                "profiling_pct": grp["total_pct"],
                "total_kernel_time_us": grp["total_kernel_time_us"],
                "gpu_kernels": grp["gpu_kernels"],
                "kernels": grp["kernels"],
                "roofline_efficiency": grp.get("roofline_efficiency"),
                "achieved_tflops": grp.get("achieved_tflops"),
                "gflops_per_call": grp.get("gflops_per_call"),
                "compute_spec": grp.get("compute_spec"),
                "enabled": True,
            })

    # Add bottleneck-driven problems (individual ops not covered by fusion or GEMM)
    if os.path.isfile(bottlenecks_path):
        existing_files = {o["file"] for o in manifest["optimizations"]}
        with open(bottlenecks_path) as _fh:
            _bottleneck_data = json.load(_fh)
        for b in _bottleneck_data:
            if not b.get("optimizable", False):
                continue
            safe_name = b["name"][:40].replace("::", "_").replace(" ", "_").replace(",", "")
            problem_file = f"problem_{safe_name}.py"
            if problem_file in existing_files:
                continue
            if os.path.isfile(os.path.join(output_dir, problem_file)):
                manifest["optimizations"].append({
                    "name": safe_name,
                    "file": problem_file,
                    "type": "individual",
                    "priority": "HIGH" if b["pct"] > 10 else "MEDIUM" if b["pct"] > 5 else "LOW",
                    "kernel_type": b.get("kernel_type", "unknown"),
                    "original_kernel": b["name"],
                    "enabled": True,
                })

    # Add roofline-gated entries, dedup by original_kernel name
    if roofline_entries:
        existing_kernels = set()
        for o in manifest["optimizations"]:
            if "original_kernel" in o:
                existing_kernels.add(o["original_kernel"])
            existing_kernels.add(o.get("name", ""))
            # Also exclude sub-kernels already covered by a kernel_family entry
            for gk in o.get("gpu_kernels", []):
                existing_kernels.add(gk)
                base = _extract_base_kernel_name(gk)
                existing_kernels.add(base)
        for entry in roofline_entries:
            orig = entry.get("original_kernel", "")
            if orig in existing_kernels or entry["name"] in existing_kernels:
                continue
            manifest["optimizations"].append(entry)
            existing_kernels.add(orig)
            existing_kernels.add(entry["name"])

    # --- Priority scoring: rank by optimization potential ---
    # Recalculate profiling_pct for entries that have total_kernel_time_us but no pct
    if total_gpu_kernel_time_us and total_gpu_kernel_time_us > 0:
        for opt in manifest["optimizations"]:
            kt_us = opt.get("total_kernel_time_us", 0)
            if kt_us > 0 and opt.get("profiling_pct", 0) == 0:
                opt["profiling_pct"] = round(kt_us / total_gpu_kernel_time_us * 100, 2)

    for opt in manifest["optimizations"]:
        pct = opt.get("profiling_pct", 0)
        # For fused_kernel_pct entries without profiling_pct, use fused_kernel_pct
        if pct == 0 and opt.get("fused_kernel_pct", 0) > 0:
            pct = opt["fused_kernel_pct"]
            opt["profiling_pct"] = round(pct, 2)

        opt_type = opt.get("type", "")
        roofline_eff = opt.get("roofline_efficiency")

        # For GEMM entries: compute weighted average roofline efficiency from shapes
        if opt_type == "gemm" and "shapes" in opt and roofline_eff is None:
            shapes = opt["shapes"]
            total_time = sum(s.get("kernel_time_us", 0) for s in shapes)
            if total_time > 0:
                weighted_eff = sum(
                    s.get("pct_roofline", 0) * s.get("kernel_time_us", 0)
                    for s in shapes
                ) / total_time
                roofline_eff = round(weighted_eff, 2)
                opt["roofline_efficiency"] = roofline_eff

        # Priority score formula:
        # - Fusion: pct * 1.0 (no roofline efficiency concept)
        # - Others with roofline: pct * (1 - efficiency/100)
        # - Others without roofline: pct * 1.0 (treat as fully improvable)
        if opt_type == "fused" or roofline_eff is None:
            opt["priority_score"] = round(pct, 4)
        else:
            opt["priority_score"] = round(pct * (1 - roofline_eff / 100.0), 4)

    # Sort by priority_score descending
    manifest["optimizations"].sort(key=lambda o: -o.get("priority_score", 0))

    # Mark top N as optimize=True, rest as optimize=False
    for i, opt in enumerate(manifest["optimizations"]):
        opt["optimize"] = i < top_n

    manifest_path = os.path.join(output_dir, "optimization_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print ranking summary
    print(f"\nOptimization priority ranking (top {top_n} selected):")
    for i, opt in enumerate(manifest["optimizations"]):
        marker = "*" if opt.get("optimize") else " "
        pct = opt.get("profiling_pct", 0)
        eff = opt.get("roofline_efficiency")
        score = opt.get("priority_score", 0)
        eff_str = f"{eff:.0f}%" if eff is not None else "n/a"
        print(f"  {marker} {i+1}. {opt['name']:45s} pct={pct:5.1f}%  eff={eff_str:>5s}  score={score:6.2f}")
    print(f"\nSaved optimization_manifest.json ({len(manifest['optimizations'])} entries)")
    return manifest




def generate_roofline_gated_problems(roofline_path, gemm_csv_path, threshold=80.0, pct_threshold=1.0):
    """Generate manifest entries for roofline-gated optimization targets.

    Returns list of manifest entries (does NOT create problem .py files).
    Skips communication, moe_sort, and aten_gemm when GEMM.csv exists.
    """
    if not roofline_path or not os.path.isfile(roofline_path):
        return []
    with open(roofline_path) as f:
        bottlenecks = json.load(f)
    has_gemm_csv = gemm_csv_path and os.path.isfile(gemm_csv_path)

    entries = []
    for b in bottlenecks:
        kernel_type = b.get("kernel_type", "other")
        if kernel_type in SKIP_KERNEL_TYPES:
            continue
        if kernel_type == "aten_gemm" and has_gemm_csv:
            continue

        pct = b.get("pct", 0)
        eff = b.get("roofline_efficiency")
        has_pm = b.get("has_perf_model", False)

        # Gate: below roofline threshold, or no perf model and above pct threshold
        if eff is not None and eff >= threshold:
            continue
        if eff is None and pct < pct_threshold:
            continue
        if pct < pct_threshold:
            continue

        geak_mode, geak_config = GEAK_MODE_MAP.get(kernel_type, ("kernel-url", "mini_kernel.yaml"))
        if geak_mode == "skip":
            continue

        compute_spec = b.get("compute_spec", "")
        spec_confidence = b.get("spec_confidence", "")
        peak_tflops = b.get("peak_tflops")
        tflops_s = b.get("tflops_s")

        if has_pm and tflops_s is not None and peak_tflops:
            perf_note = f"{tflops_s:.1f} TFLOPS/s = {eff:.1f}% of {peak_tflops} {compute_spec} peak"
        else:
            infer_note = f" (inferred {compute_spec} via {spec_confidence})" if compute_spec else ""
            perf_note = f"no perf model — pct-gated only{infer_note}"

        safe_name = b["name"][:40].replace("::", "_").replace(" ", "_").replace(",", "")
        entries.append({
            "name": safe_name,
            "type": "roofline_gated",
            "priority": "HIGH" if pct > 10 else "MEDIUM" if pct > 3 else "LOW",
            "kernel_type": kernel_type,
            "geak_mode": geak_mode,
            "geak_config": geak_config,
            "profiling_pct": round(pct, 2),
            "compute_spec": compute_spec,
            "spec_confidence": spec_confidence,
            "peak_tflops": peak_tflops,
            "roofline_efficiency": eff,
            "tflops_s": tflops_s,
            "has_perf_model": has_pm,
            "performance_note": perf_note,
            "kernel_details": b.get("kernel_details", ""),
            "original_kernel": b["name"],
            "enabled": True,
        })

    return entries


def enrich_manifest_with_kernel_types(manifest_path, kernel_types_path):
    """Enrich optimization_manifest.json with kernel type classification data.

    Adds source_file, python_binding, geak_mode, geak_config, profiling_pct,
    profiling_pct_raw, and roofline_efficiency to each manifest entry based on
    kernel_type_classification.json from Phase 6 Step 1.5.
    """
    if not os.path.isfile(kernel_types_path):
        print(f"WARNING: kernel_type_classification.json not found at {kernel_types_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(kernel_types_path) as f:
        kernel_types = json.load(f)

    # Build lookup by kernel name (normalized)
    kt_lookup = {}
    for kt in kernel_types.get("classifications", []):
        name = kt.get("name", "").lower().replace("::", "_").replace(" ", "_")
        kt_lookup[name] = kt
        # Also index by short name
        if "::" in kt.get("name", ""):
            short = kt["name"].split("::")[-1].lower().replace(" ", "_")
            kt_lookup[short] = kt

    for opt in manifest.get("optimizations", []):
        opt_name = opt.get("name", "").lower()
        kernel_type = opt.get("kernel_type", "unknown")

        # Find matching kernel type classification
        kt = kt_lookup.get(opt_name)
        if not kt:
            # Try partial match — require at least 4 chars to avoid spurious matches
            for key, val in kt_lookup.items():
                if len(key) >= 4 and len(opt_name) >= 4 and (key in opt_name or opt_name in key):
                    kt = val
                    break
        if not kt:
            # Fallback: match by kernel_type (e.g., all GEMM problems share aten_gemm classification)
            for val in kernel_types.get("classifications", []):
                if val.get("kernel_type") == kernel_type:
                    kt = val
                    break

        # Set GEAK mode — only override if not already set by generate_manifest
        if "geak_mode" not in opt:
            geak_mode, geak_config = GEAK_MODE_MAP.get(kernel_type, ("kernel-url", "mini_kernel.yaml"))
            opt["geak_mode"] = geak_mode
            opt["geak_config"] = geak_config

        if kt:
            opt.setdefault("source_file", kt.get("source_file", ""))
            opt.setdefault("python_binding", kt.get("python_binding", ""))
            # Only set profiling_pct from classification if not already set (e.g., grouped GEMMs)
            if opt.get("profiling_pct", 0) == 0:
                opt["profiling_pct"] = kt.get("pct_optimizable", 0)
            opt.setdefault("profiling_pct_raw", kt.get("pct_total_raw", 0))
            opt.setdefault("roofline_efficiency", kt.get("roofline_efficiency", None))
            opt.setdefault("bottleneck_recommendation", kt.get("bottleneck_recommendation", ""))
        else:
            opt.setdefault("source_file", "")
            opt.setdefault("python_binding", "")
            if opt.get("profiling_pct", 0) == 0:
                opt["profiling_pct"] = 0
            opt.setdefault("profiling_pct_raw", 0)
            opt.setdefault("roofline_efficiency", None)
            opt.setdefault("bottleneck_recommendation", "")

    # Fusion-aware matching: fix profiling_pct for fused ops that couldn't match above
    FUSION_KEYWORDS = {
        "fused_residual_rmsnorm": ["rmsnorm", "add_rmsnorm"],
        "fused_swiglu": ["silu", "swiglu", "fused_moe"],
    }
    classifications = kernel_types.get("classifications", [])
    for opt in manifest.get("optimizations", []):
        if opt.get("type") != "fused" or opt.get("profiling_pct", 0) > 0:
            continue
        fusion_name = opt.get("name", "")
        keywords = FUSION_KEYWORDS.get(fusion_name, [])
        if not keywords:
            continue
        matched_pct = 0
        matched_kt = None
        for cls in classifications:
            cls_name_lower = cls.get("name", "").lower()
            if any(kw in cls_name_lower for kw in keywords):
                matched_pct += cls.get("pct_optimizable", 0)
                if matched_kt is None:
                    matched_kt = cls
        if matched_pct > 0:
            opt["profiling_pct"] = round(matched_pct, 2)
            if matched_kt:
                kt_type = matched_kt.get("kernel_type", opt.get("kernel_type", "unknown"))
                opt["kernel_type"] = kt_type
                geak_mode, geak_config = GEAK_MODE_MAP.get(kt_type, ("kernel-url", "mini_kernel.yaml"))
                opt["geak_mode"] = geak_mode
                opt["geak_config"] = geak_config
                opt["bottleneck_recommendation"] = matched_kt.get("bottleneck_recommendation", "")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Enriched manifest with kernel type data ({len(manifest['optimizations'])} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate problem files from InferenceX profiling data")
    parser.add_argument("--fusion-opportunities", default="", help="Path to fusion_opportunities.json")
    parser.add_argument("--gap-analysis", default="", help="Path to gap_analysis.json")
    parser.add_argument("--bottleneck-kernels", default="", help="Path to bottleneck_kernels.json")
    parser.add_argument("--gemm-csv", default="", help="Path to GEMM.csv from TraceLens")
    parser.add_argument("--gpu-arch", default="", help="Path to gpu_arch.json")
    parser.add_argument("--model-shapes", default="", help="Path to model_shapes.json")
    parser.add_argument("--framework", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--priority-threshold", type=float, default=5.0, help="Min %% for individual problem files")
    parser.add_argument("--output-dir", required=True, help="Output directory for problem files")
    parser.add_argument("--kernel-types", default="", help="Path to kernel_type_classification.json from Phase 6 Step 1.5")
    parser.add_argument("--roofline-bottlenecks", default="", help="Path to roofline_bottlenecks.json")
    parser.add_argument("--roofline-threshold", type=float, default=80.0, help="Roofline efficiency threshold (default: 80.0)")
    parser.add_argument("--kernel-summary-csv", default="", help="Path to kernel_summary.csv from TraceLens")
    parser.add_argument("--ops-unique-args-csv", default="", help="Path to ops_unique_args.csv from TraceLens")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model shapes
    model_shapes = {}
    if args.model_shapes and os.path.isfile(args.model_shapes):
        with open(args.model_shapes) as f:
            model_shapes = json.load(f)
    print(f"Model shapes: {model_shapes}")

    # Load GPU arch
    gpu_arch = None
    if args.gpu_arch and os.path.isfile(args.gpu_arch):
        with open(args.gpu_arch) as f:
            gpu_arch = json.load(f)
        print(f"GPU arch: {gpu_arch.get('name', 'unknown')}")

    # Generate fusion problems
    total = 0
    if args.fusion_opportunities:
        n = generate_fusion_problems(args.fusion_opportunities, model_shapes, args.output_dir)
        total += n or 0
    print()

    # Generate GEMM problems (grouped by GPU kernel family)
    n, gemm_groups = generate_gemm_problems(args.gemm_csv, gpu_arch, args.output_dir, args.priority_threshold)
    total += n or 0
    print()

    # Collect existing GEMM family names to avoid duplicates
    existing_gemm_families = set()
    for grp in gemm_groups:
        existing_gemm_families.add(grp["safe_name"])

    # Generate kernel family problems for ALL ops from kernel_summary.csv
    kernel_groups = []
    kernel_summary_csv = args.kernel_summary_csv
    ops_unique_args_csv = args.ops_unique_args_csv
    # Auto-discover CSVs from same directory as GEMM.csv if not specified
    if not kernel_summary_csv and args.gemm_csv:
        candidate = os.path.join(os.path.dirname(args.gemm_csv), "kernel_summary.csv")
        if os.path.isfile(candidate):
            kernel_summary_csv = candidate
    if not ops_unique_args_csv and args.gemm_csv:
        candidate = os.path.join(os.path.dirname(args.gemm_csv), "ops_unique_args.csv")
        if os.path.isfile(candidate):
            ops_unique_args_csv = candidate

    if kernel_summary_csv:
        n, kernel_groups = generate_kernel_family_problems(
            kernel_summary_csv, ops_unique_args_csv,
            args.output_dir, args.priority_threshold, existing_gemm_families,
            gpu_arch=gpu_arch
        )
        total += n or 0
    print()

    # Generate roofline-gated entries
    roofline_entries = generate_roofline_gated_problems(
        args.roofline_bottlenecks, args.gemm_csv, args.roofline_threshold
    )
    if roofline_entries:
        print(f"Roofline-gated optimization targets: {len(roofline_entries)}")
        for e in roofline_entries:
            print(f"  [{e['priority']}] {e['name']} ({e['profiling_pct']:.1f}%) {e['geak_mode']} -- {e['performance_note'][:60]}")
    print()

    # Compute total GPU kernel time for consistent profiling_pct calculation
    total_gpu_kernel_time_us = None
    if kernel_summary_csv and os.path.isfile(kernel_summary_csv):
        total_gpu_kernel_time_us = 0
        with open(kernel_summary_csv) as f:
            for row in csv.DictReader(f):
                try:
                    total_gpu_kernel_time_us += float(row.get("Kernel duration (µs)_sum", 0) or 0)
                except (ValueError, TypeError):
                    pass

    # Generate manifest
    bottlenecks_path = args.bottleneck_kernels or os.path.join(args.output_dir, "bottleneck_kernels.json")
    fusions_path = args.fusion_opportunities or os.path.join(args.output_dir, "fusion_opportunities.json")
    generate_manifest(args.output_dir, fusions_path, bottlenecks_path, args.framework,
                      roofline_entries, gemm_groups, kernel_groups,
                      total_gpu_kernel_time_us=total_gpu_kernel_time_us)

    # Enrich manifest with kernel type classification data
    if args.kernel_types:
        manifest_path = os.path.join(args.output_dir, "optimization_manifest.json")
        enrich_manifest_with_kernel_types(manifest_path, args.kernel_types)

    print(f"\nTotal problem files generated: {total}")
    problem_files = [f for f in sorted(os.listdir(args.output_dir)) if f.startswith("problem_") and f.endswith(".py")]
    for f in problem_files:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
