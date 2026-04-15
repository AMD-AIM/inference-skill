#!/usr/bin/env python3
"""
Generate SGLang Plugin from Optimized Kernels

Scans *_opt.py files, maps them to SGLang layer classes, and generates
a sglang_plugin/ package that patches target modules via monkey-patching.

SGLang does not have a CustomOp.register_oot() equivalent like vLLM,
so this plugin patches module-level class references at import time.

Usage:
    python generate_sglang_plugin.py --kernel-dir ./optimized
    python generate_sglang_plugin.py --kernel-dir ./problems --output-dir ./optimized/sglang_plugin

Part of the inferencex-optimize skill. Can be used standalone.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from plugin_utils import detect_kernel_name, find_matching_op

# Mapping from kernel problem name patterns to SGLang module targets.
# Each entry: pattern -> (sglang_module, class_name, adapter_type)
KERNEL_MAP = {
    "fused_residual_rmsnorm": ("sglang.srt.layers.layernorm", "RMSNorm", "rmsnorm_fused"),
    "fused_rmsnorm":          ("sglang.srt.layers.layernorm", "RMSNorm", "rmsnorm"),
    "rmsnorm":                ("sglang.srt.layers.layernorm", "RMSNorm", "rmsnorm"),
    "layernorm":              ("sglang.srt.layers.layernorm", "RMSNorm", "rmsnorm"),
    "fused_swiglu":           ("sglang.srt.layers.activation", "SiLuAndMul", "silu_and_mul"),
    "fused_silu_mul":         ("sglang.srt.layers.activation", "SiLuAndMul", "silu_and_mul"),
    "silu_mul":               ("sglang.srt.layers.activation", "SiLuAndMul", "silu_and_mul"),
    "gelu_mul":               ("sglang.srt.layers.activation", "GeluAndMul", "gelu_and_mul"),
}


def generate_adapter_code(kernel_file, module_path, class_name, adapter_type):
    """Generate the Python code for a monkey-patch adapter."""
    module_name = kernel_file.replace(".py", "")

    if adapter_type in ("rmsnorm_fused", "rmsnorm"):
        return f'''
def patch_{class_name.lower()}():
    """Patch {module_path}.{class_name} with optimized kernel."""
    import importlib
    target_module = importlib.import_module("{module_path}")
    Original{class_name} = getattr(target_module, "{class_name}")

    class Optimized{class_name}(Original{class_name}):
        _opt_cache = {{}}

        def forward(self, x, residual=None):
            h = self.weight.shape[0]
            key = (h, x.device, x.dtype)
            if key not in self._opt_cache:
                try:
                    from {module_name} import ModelNew
                    k = ModelNew(h).to(device=x.device, dtype=x.dtype)
                    k.weight.data.copy_(self.weight.data)
                    self._opt_cache[key] = k
                except Exception as e:
                    print(f"[sglang_plugin] {class_name} init failed: {{e}}, using default")
                    self._opt_cache[key] = None
            opt = self._opt_cache[key]
            if opt is None:
                return super().forward(x, residual)
            try:
                opt.weight.data.copy_(self.weight.data)
                if residual is not None:
                    return opt(x, residual)
                return opt(x)
            except Exception:
                return super().forward(x, residual)

    setattr(target_module, "{class_name}", Optimized{class_name})
    print(f"[sglang_plugin] Patched {module_path}.{class_name}")
'''

    elif adapter_type in ("silu_and_mul", "gelu_and_mul"):
        return f'''
def patch_{class_name.lower()}():
    """Patch {module_path}.{class_name} with optimized kernel."""
    import importlib
    target_module = importlib.import_module("{module_path}")
    Original{class_name} = getattr(target_module, "{class_name}")

    class Optimized{class_name}(Original{class_name}):
        _opt_kernel = None
        _init_failed = False

        def forward(self, x):
            if self._init_failed:
                return super().forward(x)
            if self._opt_kernel is None:
                try:
                    from {module_name} import ModelNew
                    self.__class__._opt_kernel = ModelNew()
                except Exception as e:
                    print(f"[sglang_plugin] {class_name} init failed: {{e}}, using default")
                    self.__class__._init_failed = True
                    return super().forward(x)
            try:
                d = x.shape[-1] // 2
                return self._opt_kernel(x[..., :d], x[..., d:])
            except Exception:
                return super().forward(x)

    setattr(target_module, "{class_name}", Optimized{class_name})
    print(f"[sglang_plugin] Patched {module_path}.{class_name}")
'''

    return None


def generate_plugin(kernel_dir, output_dir=None):
    """Generate the sglang_plugin package."""
    kernel_dir = os.path.abspath(kernel_dir)
    if output_dir is None:
        output_dir = os.path.join(kernel_dir, "sglang_plugin")
    os.makedirs(output_dir, exist_ok=True)

    opt_files = [f for f in os.listdir(kernel_dir) if f.endswith("_opt.py")]
    if not opt_files:
        print("No *_opt.py files found!")
        return {"status": "no_kernels"}

    print(f"Found {len(opt_files)} optimized kernel files:")
    for f in opt_files:
        print(f"  {f}")

    registrations = []
    skipped = []

    for opt_file in sorted(opt_files):
        kernel_name = detect_kernel_name(opt_file)
        if not kernel_name:
            skipped.append((opt_file, "could not parse name"))
            continue

        match = find_matching_op(kernel_name, KERNEL_MAP)
        if not match:
            skipped.append((opt_file, f"no SGLang module mapping for '{kernel_name}'"))
            continue

        module_path, class_name, adapter_type = match

        code = generate_adapter_code(opt_file, module_path, class_name, adapter_type)
        if code is None:
            skipped.append((opt_file, f"adapter type '{adapter_type}' not supported"))
            continue

        if any(r["class_name"] == class_name for r in registrations):
            skipped.append((opt_file, f"already have a patch for {class_name}"))
            continue

        registrations.append({
            "kernel_file": opt_file,
            "kernel_name": kernel_name,
            "module_path": module_path,
            "class_name": class_name,
            "adapter_type": adapter_type,
            "adapter_code": code,
        })
        print(f"  + {opt_file} -> {class_name} ({adapter_type})")

    for f, reason in skipped:
        print(f"  - {f}: {reason}")

    if not registrations:
        print("\nNo kernels could be mapped to SGLang modules.")
        return {"status": "no_mappable_kernels", "skipped": skipped}

    # Generate __init__.py
    patch_calls = []
    patch_functions = []
    for reg in registrations:
        patch_functions.append(reg["adapter_code"])
        patch_calls.append(f"        patch_{reg['class_name'].lower()}()")

    init_code = f'''"""
SGLang Plugin — Patches SGLang modules with optimized Triton kernels.
Auto-generated by generate_sglang_plugin.py. Do not edit manually.

Uses module-level monkey-patching since SGLang does not have
a CustomOp.register_oot() mechanism like vLLM.
"""
import os
import sys

_KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

{"".join(patch_functions)}

def patch_all():
    """Apply all optimized kernel patches to SGLang modules."""
    print("[sglang_plugin] Applying optimized kernel patches...")
    try:
{chr(10).join(patch_calls)}
        print("[sglang_plugin] All patches applied")
    except Exception as e:
        print(f"[sglang_plugin] Patch error: {{e}}")

patch_all()
'''

    init_path = os.path.join(output_dir, "__init__.py")
    with open(init_path, "w") as f:
        f.write(init_code)
    print(f"\nGenerated: {init_path}")

    # Generate launcher
    launcher_code = f'''#!/usr/bin/env python3
"""
Launch SGLang with optimized kernel patches applied.
Auto-generated by generate_sglang_plugin.py.

Usage:
    python3 run_patched_sglang.py --model <MODEL> --port 8193
"""
import sys
import os

sys.path.insert(0, "{os.path.abspath(output_dir)}")
sys.path.insert(0, os.path.dirname("{os.path.abspath(output_dir)}"))
import sglang_plugin  # noqa: F401 — triggers patch_all()

if __name__ == "__main__":
    from sglang.launch_server import prepare_server_args, run_server
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
'''

    launcher_path = os.path.join(os.path.dirname(output_dir), "run_patched_sglang.py")
    with open(launcher_path, "w") as f:
        f.write(launcher_code)
    os.chmod(launcher_path, 0o755)
    print(f"Generated: {launcher_path}")

    # Save manifest
    manifest = {
        "patched": [
            {"kernel": r["kernel_file"], "patches": r["class_name"], "type": r["adapter_type"]}
            for r in registrations
        ],
        "skipped": [{"file": f, "reason": r} for f, r in skipped],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPlugin generated with {len(registrations)} patches")
    return {"status": "ok", "registrations": len(registrations), "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SGLang plugin from optimized kernels")
    parser.add_argument("--kernel-dir", required=True, help="Directory containing *_opt.py files")
    parser.add_argument("--output-dir", default=None, help="Output directory for sglang_plugin/")
    args = parser.parse_args()

    result = generate_plugin(args.kernel_dir, args.output_dir)
    if result.get("status") != "ok":
        print(f"\nStatus: {result.get('status')}")
        sys.exit(1)
