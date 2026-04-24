#!/usr/bin/env python3
"""
Kernel classification, precision inference, and roofline analysis.

Single source of truth for the kernel type taxonomy. Everything is driven by
KERNEL_TYPES -- a declarative registry that defines patterns, GEAK mode,
default compute spec, compute class, and skip behavior per type.

Imported by Phase 06 (`upstream-resolve` / `problem-generate`) for kernel
classification feeding `kernel_source_map.yaml` lookup. Part of the
inference-optimize skill.
"""
import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Kernel type registry -- single source of truth
#
# Order matters: first match wins in classify_kernel().
# Fields per entry:
#   patterns      -- substrings matched against lowercased kernel name
#   prefix        -- (optional) matched via str.startswith on lowercased name
#   label         -- human-readable category name
#   default_spec  -- fallback Compute Spec when no perf model (Tier 3)
#   compute_class -- "matrix" or "vector" (None = not optimizable)
#   geak          -- (mode, config) for GEAK optimization
#   skip          -- if True, excluded from roofline bottleneck / optimization
# ---------------------------------------------------------------------------

KERNEL_TYPES = [
    ("vendor",           {"patterns": ["hipblas", "hipblaslt", "cublas", "rocblas"],
                          "prefix": "cijk_",
                          "label": "Vendor GEMM",      "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("communication",    {"patterns": ["nccl", "allreduce", "allgather", "reduce_scatter",
                                       "cross_device_reduce", "record_param_comms"],
                          "label": "Communication",     "default_spec": None,
                          "compute_class": None,         "geak": ("skip", None),
                          "skip": True}),
    ("ck",               {"patterns": ["ck_fmha", "ck_tile", "_zn7ck_tile"],  # ck_fmha captured here (first-match-wins)
                          "label": "CK Kernel",         "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("hip",              {"patterns": ["wvsplitk", "wvsplit", "_rocm_c"],
                          "label": "HIP Custom",        "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("triton",           {"patterns": ["triton_red_fused"],
                          "prefix": "triton_",
                          "label": "Triton",            "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("simple", "geak.yaml")}),
    ("attention",        {"patterns": ["attention", "flash_attn", "paged_attention", "sdpa",
                                      "mla_pfl", "mla_reduce", "mla_prefill"],
                          "label": "Attention",         "default_spec": "matrix_fp8",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("normalization",    {"patterns": ["rmsnorm", "layernorm", "layer_norm"],
                          "label": "Normalization",     "default_spec": "vector_bf16",
                          "compute_class": "vector",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("activation",       {"patterns": ["silu", "gelu", "relu", "swish"],
                          "label": "Activation",        "default_spec": "vector_bf16",
                          "compute_class": "vector",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("moe_gemm",         {"patterns": ["kernel_moe_mxgemm", "ck_moe_stage",
                                       "moe_ck2stages_gemm", "moeflatmmkernel"],
                          "label": "MoE GEMM",         "default_spec": "matrix_fp4",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("moe_sort",         {"patterns": ["moe_sort", "mxfp4_quant_moe_sort", "moesortingmultiphase"],
                          "label": "MoE Sort",          "default_spec": "matrix_fp4",
                          "compute_class": "matrix",    "geak": ("skip", None),
                          "skip": True}),
    ("gemm_fp4",         {"patterns": ["gemm_afp4wfp4"],
                          "label": "FP4 GEMM",         "default_spec": "matrix_fp4",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("moe",              {"patterns": ["moe", "fused_moe", "moe_align"],
                          "label": "MoE",              "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("aten_gemm",        {"patterns": ["aten::mm", "aten::addmm", "aten::bmm"],
                          "label": "ATen GEMM",        "default_spec": "matrix_bf16",
                          "compute_class": "matrix",    "geak": ("kernel-url", "mini_kernel.yaml")}),
    ("aten_elementwise", {"patterns": ["aten::add", "aten::mul", "aten::copy", "elementwise"],
                          "label": "ATen Elementwise",  "default_spec": "vector_fp32",
                          "compute_class": "vector",    "geak": ("kernel-url", "mini_kernel.yaml")}),
]

# Derived lookups (computed once at import time)
OP_TYPE_DEFAULT_SPEC = {n: m["default_spec"] for n, m in KERNEL_TYPES if m.get("default_spec")}
OP_TYPE_DEFAULT_SPEC["other"] = "matrix_bf16"

MATRIX_KERNEL_TYPES = {n for n, m in KERNEL_TYPES if m.get("compute_class") == "matrix"}

GEAK_MODE_MAP = {n: m["geak"] for n, m in KERNEL_TYPES}
GEAK_MODE_MAP.update({
    "triton_composite": ("kernel-url", "mini_kernel.yaml"),
    "asm":              ("kernel-url", "mini_kernel.yaml"),
    "unknown":          ("kernel-url", "mini_kernel.yaml"),
    "other":            ("kernel-url", "mini_kernel.yaml"),
})

SKIP_KERNEL_TYPES = {n for n, m in KERNEL_TYPES if m.get("skip")}

_PARENT_CAT_FALLBACKS = [
    (["gemm"],                "aten_gemm",  "GEMM"),
    (["attention", "sdpa"],   "attention",  "Attention"),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_kernel(name, parent_category=""):
    """Classify a kernel by its name pattern. Returns (kernel_type, label)."""
    lower = name.lower()
    for ktype, meta in KERNEL_TYPES:
        prefix = meta.get("prefix")
        if prefix and lower.startswith(prefix):
            return ktype, meta["label"]
        if any(p in lower for p in meta["patterns"]):
            return ktype, meta["label"]
    if parent_category:
        pc = parent_category.lower()
        for keywords, ktype, label in _PARENT_CAT_FALLBACKS:
            if any(k in pc for k in keywords):
                return ktype, label
    return "other", "Other"


# ---------------------------------------------------------------------------
# Unified perf summary loader
# ---------------------------------------------------------------------------

def load_unified_perf_summary(tracelens_dir):
    """Load unified_perf_summary.csv -- roofline data for all ops."""
    path = os.path.join(tracelens_dir, "unified_perf_summary.csv")
    if not os.path.isfile(path):
        return []
    ops = []
    with open(path) as f:
        for row in csv.DictReader(f):
            has_pm = row.get("has_perf_model", "").strip().lower() == "true"
            tflops = flops_byte = None
            if has_pm:
                for key, attr in [("TFLOPS/s_mean", "tflops"), ("FLOPS/Byte", "flops_byte")]:
                    raw = row.get(key, "")
                    try:
                        val = float(raw) if raw and raw.strip() not in ("", "()") else None
                    except (ValueError, TypeError):
                        val = None
                    if attr == "tflops":
                        tflops = val
                    else:
                        flops_byte = val
            def _float(key, default=0.0):
                try:
                    return float(row.get(key, default) or default)
                except (ValueError, TypeError):
                    return default
            ops.append({
                "name":             row.get("name", ""),
                "op_category":      row.get("op category", ""),
                "pct":              _float("Percentage (%)", 0.0),
                "has_perf_model":   has_pm,
                "tflops_s":         tflops,
                "flops_byte":       flops_byte,
                "compute_spec":     row.get("Compute Spec", "").strip(),
                "kernel_time_sum_us": _float("Kernel Time (µs)_sum", 0.0),
                "kernel_details":   row.get("trunc_kernel_details", ""),
            })
    return ops


# ---------------------------------------------------------------------------
# Precision inference (3-tier cascade)
# ---------------------------------------------------------------------------

PRECISION_PATTERNS = [
    ("matrix_fp4",  ["mxfp4", "afp4wfp4", "mx4"]),
    ("matrix_fp8",  ["fp8", "e4m3", "e5m2", "mx8"]),
    ("matrix_fp6",  ["mxfp6", "mx6"]),
    ("matrix_bf16", ["bfloat16", "__hip_bfloat16"]),
    ("matrix_fp16", ["float16"]),
    ("matrix_int8", ["int8"]),
    ("matrix_fp32", ["float32"]),
    ("matrix_bf16", ["bf16"]),
    ("matrix_fp16", ["fp16", "half"]),
    ("matrix_fp4",  ["fp4"]),
    ("vector_fp32", ["copy_", "elementwise"]),
]

MODEL_PRECISION_MAP = {
    "fp4": "matrix_fp4",   "mxfp4": "matrix_fp4",
    "fp8": "matrix_fp8",   "e4m3": "matrix_fp8",
    "bf16": "matrix_bf16", "fp16": "matrix_fp16",
    "int8": "matrix_int8", "fp32": "matrix_fp32",
}


def infer_compute_spec(op_name, kernel_type, kernel_details, model_precision=None):
    """Infer Compute Spec via 3-tier cascade. Returns (spec, confidence)."""
    search_text = (op_name + " " + (kernel_details or "")).lower()

    for spec, patterns in PRECISION_PATTERNS:
        if any(p in search_text for p in patterns):
            return spec, "inferred_name"

    if model_precision and kernel_type in MATRIX_KERNEL_TYPES:
        spec = MODEL_PRECISION_MAP.get(model_precision.lower())
        if spec:
            return spec, "inferred_model"

    return OP_TYPE_DEFAULT_SPEC.get(kernel_type, "matrix_bf16"), "heuristic"


# ---------------------------------------------------------------------------
# Roofline bottleneck builder
# ---------------------------------------------------------------------------

def build_roofline_bottlenecks(unified_ops, gpu_arch, model_precision=None):
    """Build roofline-enriched bottleneck list from unified_perf_summary data."""
    max_tflops = gpu_arch.get("max_achievable_tflops", {}) if gpu_arch else {}
    bottlenecks = []
    for op in unified_ops:
        name = op["name"]
        kernel_type, _ = classify_kernel(name)

        if kernel_type in SKIP_KERNEL_TYPES:
            continue
        if op.get("op_category", "").lower() in ("record_param_comms",):
            continue

        has_pm = op["has_perf_model"]
        csv_spec = op.get("compute_spec", "")

        if has_pm and csv_spec:
            compute_spec, spec_confidence = csv_spec, "measured"
        else:
            compute_spec, spec_confidence = infer_compute_spec(
                name, kernel_type, op.get("kernel_details", ""), model_precision)

        peak_tflops = max_tflops.get(compute_spec)
        if peak_tflops is None and compute_spec:
            print(f"  Warning: compute_spec '{compute_spec}' not in gpu_arch for '{name}', skipping roofline calc")

        tflops_s = op.get("tflops_s")
        roofline_eff = None
        if has_pm and tflops_s is not None and peak_tflops and peak_tflops > 0:
            roofline_eff = round(tflops_s / peak_tflops * 100, 2)

        bottlenecks.append({
            "name": name, "kernel_type": kernel_type, "pct": op["pct"],
            "has_perf_model": has_pm, "compute_spec": compute_spec,
            "spec_confidence": spec_confidence, "peak_tflops": peak_tflops,
            "tflops_s": tflops_s, "roofline_efficiency": roofline_eff,
            "flops_byte": op.get("flops_byte"),
            "kernel_time_sum_us": op.get("kernel_time_sum_us", 0),
            "kernel_details": op.get("kernel_details", ""),
        })
    return bottlenecks
