"""Shared utilities for plugin generators."""


def detect_kernel_name(filename):
    """Extract kernel name from filename like problem_fused_rmsnorm_opt.py."""
    base = filename.replace("problem_", "").replace("_opt.py", "").replace("_opt", "")
    return base if base else None


def find_matching_op(kernel_name, kernel_map):
    """Find a matching entry in kernel_map for this kernel name."""
    # Exact match first
    if kernel_name in kernel_map:
        return kernel_map[kernel_name]
    # Then check if any pattern is a substring of the kernel name
    for pattern, target in kernel_map.items():
        if pattern in kernel_name:
            return target
    return None
