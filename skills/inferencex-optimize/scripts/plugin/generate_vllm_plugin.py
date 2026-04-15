#!/usr/bin/env python3
"""
Generate vLLM Plugin from Optimized Kernels

Scans *_opt.py files, maps them to vLLM CustomOps, and generates
a vllm_plugin/ package that registers replacements via CustomOp.register_oot().

This uses vLLM's OFFICIAL extension mechanism — no monkey-patching.
See: https://docs.vllm.ai/en/latest/design/custom_op/

Usage:
    python generate_vllm_plugin.py --kernel-dir ./optimized
    python generate_vllm_plugin.py --kernel-dir ./problems --output-dir ./optimized/vllm_plugin

Part of the inferencex-optimize skill. Can be used standalone.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

from plugin_utils import detect_kernel_name, find_matching_op

# Mapping from kernel problem name patterns to vLLM CustomOp targets.
# Each entry: pattern → (vllm_module, class_name, adapter_type)
KERNEL_MAP = {
    "fused_residual_rmsnorm": ("vllm.model_executor.layers.layernorm", "RMSNorm", "rmsnorm_fused"),
    "fused_rmsnorm":          ("vllm.model_executor.layers.layernorm", "RMSNorm", "rmsnorm"),
    "rmsnorm":                ("vllm.model_executor.layers.layernorm", "RMSNorm", "rmsnorm"),
    "layernorm":              ("vllm.model_executor.layers.layernorm", "RMSNorm", "rmsnorm"),
    "layer_norm":             ("vllm.model_executor.layers.layernorm", "RMSNorm", "rmsnorm"),
    "fused_swiglu":           ("vllm.model_executor.layers.activation", "SiluAndMul", "silu_and_mul"),
    "fused_silu_mul":         ("vllm.model_executor.layers.activation", "SiluAndMul", "silu_and_mul"),
    "silu_mul":               ("vllm.model_executor.layers.activation", "SiluAndMul", "silu_and_mul"),
    "gelu_mul":               ("vllm.model_executor.layers.activation", "GeluAndMul", "gelu_and_mul"),
    "fused_rope":             ("vllm.model_executor.layers.rotary_embedding", "RotaryEmbeddingBase", "rope"),
    "rope":                   ("vllm.model_executor.layers.rotary_embedding", "RotaryEmbeddingBase", "rope"),
}


def load_model_new(filepath: str):
    """Load ModelNew class from an opt file to inspect its interface."""
    spec = importlib.util.spec_from_file_location("_temp_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "ModelNew"):
            cls = mod.ModelNew
            init_sig = inspect.signature(cls.__init__)
            params = [p for p in init_sig.parameters if p != "self"]
            return {"class": cls, "init_params": params}
    except Exception as e:
        print(f"  Warning: could not load {filepath}: {e}")
    return None


def generate_adapter_code(kernel_file: str, module_path: str, class_name: str,
                          adapter_type: str, init_params: list[str]) -> str:
    """Generate the Python code for a CustomOp adapter."""
    module_name = kernel_file.replace(".py", "")

    # Determine init_args forwarding
    if init_params:
        # Common case: hidden_size is the first param
        init_from_self = "self.weight.shape[0]" if "hidden_size" in str(init_params) or len(init_params) == 1 else "self.weight.shape[0]"
    else:
        init_from_self = None

    if adapter_type == "rmsnorm_fused":
        return f'''
class Optimized{class_name}({class_name}):
    """Replaces {class_name} with fused residual+norm Triton kernel."""
    _opt_cache = {{}}

    def forward_hip(self, x, residual=None):
        h = self.weight.shape[0]
        key = (h, x.device, x.dtype)
        if key not in self._opt_cache:
            try:
                from {module_name} import ModelNew
                k = ModelNew(h).to(device=x.device, dtype=x.dtype)
                k.weight.data.copy_(self.weight.data)
                self._opt_cache[key] = k
            except Exception as e:
                print(f"[vllm_plugin] {class_name} init failed: {{e}}, using default")
                self._opt_cache[key] = None
        opt = self._opt_cache[key]
        if opt is None:
            return super().forward_hip(x, residual)
        try:
            opt.weight.data.copy_(self.weight.data)
            if residual is not None:
                hidden = x + residual
                return opt(hidden), hidden
            return opt(x)
        except Exception:
            return super().forward_hip(x, residual)

    def forward_cuda(self, x, residual=None):
        return self.forward_hip(x, residual)
'''

    elif adapter_type == "rmsnorm":
        return f'''
class Optimized{class_name}({class_name}):
    """Replaces {class_name} with optimized Triton kernel."""
    _opt_cache = {{}}

    def forward_hip(self, x, residual=None):
        h = self.weight.shape[0]
        key = (h, x.device, x.dtype)
        if key not in self._opt_cache:
            try:
                from {module_name} import ModelNew
                k = ModelNew(h).to(device=x.device, dtype=x.dtype)
                k.weight.data.copy_(self.weight.data)
                self._opt_cache[key] = k
            except Exception as e:
                print(f"[vllm_plugin] {class_name} init failed: {{e}}, using default")
                self._opt_cache[key] = None
        opt = self._opt_cache[key]
        if opt is None:
            return super().forward_hip(x, residual)
        try:
            opt.weight.data.copy_(self.weight.data)
            if residual is not None:
                return super().forward_hip(x, residual)  # Fallback for fused case
            return opt(x)
        except Exception:
            return super().forward_hip(x, residual)

    def forward_cuda(self, x, residual=None):
        return self.forward_hip(x, residual)
'''

    elif adapter_type == "silu_and_mul":
        return f'''
class Optimized{class_name}({class_name}):
    """Replaces {class_name} with fused Triton kernel."""
    _opt_kernel = None
    _init_failed = False

    def forward_hip(self, x):
        if self._init_failed:
            return super().forward_hip(x)
        if self._opt_kernel is None:
            try:
                from {module_name} import ModelNew
                self.__class__._opt_kernel = ModelNew()
            except Exception as e:
                print(f"[vllm_plugin] {class_name} init failed: {{e}}, using default")
                self.__class__._init_failed = True
                return super().forward_hip(x)
        try:
            d = x.shape[-1] // 2
            return self._opt_kernel(x[..., :d], x[..., d:])
        except Exception:
            return super().forward_hip(x)

    def forward_cuda(self, x):
        return self.forward_hip(x)
'''

    elif adapter_type == "gelu_and_mul":
        return f'''
class Optimized{class_name}({class_name}):
    """Replaces {class_name} with fused Triton kernel."""
    _opt_kernel = None
    _init_failed = False

    def forward_hip(self, x):
        if self._init_failed:
            return super().forward_hip(x)
        if self._opt_kernel is None:
            try:
                from {module_name} import ModelNew
                self.__class__._opt_kernel = ModelNew()
            except Exception as e:
                print(f"[vllm_plugin] {class_name} init failed: {{e}}, using default")
                self.__class__._init_failed = True
                return super().forward_hip(x)
        try:
            d = x.shape[-1] // 2
            return self._opt_kernel(x[..., :d], x[..., d:])
        except Exception:
            return super().forward_hip(x)

    def forward_cuda(self, x):
        return self.forward_hip(x)
'''

    elif adapter_type == "rope":
        # RoPE is complex and model-specific, skip auto-generation
        return None

    return None


def generate_plugin(kernel_dir: str, output_dir: str | None = None) -> dict:
    """Generate the vllm_plugin package."""
    kernel_dir = os.path.abspath(kernel_dir)
    if output_dir is None:
        output_dir = os.path.join(kernel_dir, "vllm_plugin")
    os.makedirs(output_dir, exist_ok=True)

    # Scan for *_opt.py files
    opt_files = [f for f in os.listdir(kernel_dir) if f.endswith("_opt.py")]
    if not opt_files:
        print("No *_opt.py files found!")
        return {"status": "no_kernels"}

    print(f"Found {len(opt_files)} optimized kernel files:")
    for f in opt_files:
        print(f"  {f}")

    # Map each kernel to a vLLM CustomOp
    registrations = []
    skipped = []

    for opt_file in sorted(opt_files):
        kernel_name = detect_kernel_name(opt_file)
        if not kernel_name:
            skipped.append((opt_file, "could not parse name"))
            continue

        match = find_matching_op(kernel_name, KERNEL_MAP)
        if not match:
            skipped.append((opt_file, f"no vLLM CustomOp mapping for '{kernel_name}'"))
            continue

        module_path, class_name, adapter_type = match

        # Load and inspect the kernel
        info = load_model_new(os.path.join(kernel_dir, opt_file))
        init_params = info["init_params"] if info else []

        # Generate adapter code
        code = generate_adapter_code(opt_file, module_path, class_name, adapter_type, init_params)
        if code is None:
            skipped.append((opt_file, f"adapter type '{adapter_type}' not auto-supported"))
            continue

        # Check if this class is already registered (avoid duplicates)
        if any(r["class_name"] == class_name for r in registrations):
            skipped.append((opt_file, f"already have a registration for {class_name}"))
            continue

        registrations.append({
            "kernel_file": opt_file,
            "kernel_name": kernel_name,
            "module_path": module_path,
            "class_name": class_name,
            "adapter_type": adapter_type,
            "adapter_code": code,
        })
        print(f"  ✓ {opt_file} → {class_name} ({adapter_type})")

    for f, reason in skipped:
        print(f"  ✗ {f}: {reason}")

    if not registrations:
        print("\nNo kernels could be mapped to vLLM CustomOps.")
        return {"status": "no_mappable_kernels", "skipped": skipped}

    # Generate __init__.py
    init_code = _build_init_py(registrations, kernel_dir)
    init_path = os.path.join(output_dir, "__init__.py")
    with open(init_path, "w") as f:
        f.write(init_code)
    print(f"\nGenerated: {init_path}")

    # Generate run_patched_vllm.py
    launcher_code = _build_launcher(output_dir)
    launcher_path = os.path.join(os.path.dirname(output_dir), "run_patched_vllm.py")
    with open(launcher_path, "w") as f:
        f.write(launcher_code)
    os.chmod(launcher_path, 0o755)
    print(f"Generated: {launcher_path}")

    # Save manifest
    manifest = {
        "registered": [
            {"kernel": r["kernel_file"], "replaces": r["class_name"], "type": r["adapter_type"]}
            for r in registrations
        ],
        "skipped": [{"file": f, "reason": r} for f, r in skipped],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Plugin generated with {len(registrations)} CustomOp registrations")
    print(f"   Launch: python3 {launcher_path} --model <MODEL> --port 8193")

    return {"status": "ok", "registrations": len(registrations), "manifest": manifest}


def _build_init_py(registrations: list[dict], kernel_dir: str) -> str:
    """Build the __init__.py for the vllm_plugin package."""
    imports = set()
    adapter_blocks = []
    register_calls = []

    for reg in registrations:
        imports.add(f"from {reg['module_path']} import {reg['class_name']}")
        adapter_blocks.append(reg["adapter_code"])
        register_calls.append(
            f'        CustomOp.register_oot(_decorated_op_cls=Optimized{reg["class_name"]}, name="{reg["class_name"]}")'
        )

    register_block = "\n".join(register_calls)
    imports_block = "\n".join(imports)
    adapters_block = "".join(adapter_blocks)

    return f'''"""
vLLM Plugin — Registers optimized Triton kernels as CustomOps.
Auto-generated by generate_vllm_plugin.py. Do not edit manually.

Uses vLLM's official CustomOp.register_oot() mechanism.
See: https://docs.vllm.ai/en/latest/design/custom_op/
"""
import os
import sys

# Add kernel directory to path so *_opt.py modules can be imported
_KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

from vllm.model_executor.custom_op import CustomOp
{imports_block}

# ─── Adapter Classes ────────────────────────────────────────
{adapters_block}

# ─── Register All ───────────────────────────────────────────
def register_all():
    """Register all optimized kernels as vLLM CustomOps."""
    print("[vllm_plugin] Registering optimized CustomOps...")
    try:
{register_block}
        print("[vllm_plugin] ✅ All registrations complete")
    except Exception as e:
        print(f"[vllm_plugin] ⚠ Registration error: {{e}}")

register_all()
'''


def _build_launcher(output_dir: str) -> str:
    """Build the run_patched_vllm.py launcher script."""
    plugin_dir = os.path.abspath(output_dir)
    return f'''#!/usr/bin/env python3
"""
Launch vLLM with optimized kernel CustomOps registered.
Auto-generated by generate_vllm_plugin.py.

Usage:
    python3 run_patched_vllm.py serve --model Qwen/Qwen3-8B --port 8193
    python3 run_patched_vllm.py serve --model Qwen/Qwen3-8B --port 8193 --dtype auto
"""
import sys
import os

# Register optimized kernels BEFORE vLLM imports model code
sys.path.insert(0, "{plugin_dir}")
sys.path.insert(0, os.path.dirname("{plugin_dir}"))
import vllm_plugin  # noqa: F401 — triggers register_all()

if __name__ == "__main__":
    # Now start vLLM via its standard CLI — all args passed through
    # This is equivalent to running `vllm serve ...`
    from vllm.entrypoints.cli.main import main
    main()
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vLLM plugin from optimized kernels")
    parser.add_argument("--kernel-dir", required=True, help="Directory containing *_opt.py files")
    parser.add_argument("--output-dir", default=None, help="Output directory for vllm_plugin/")
    args = parser.parse_args()

    result = generate_plugin(args.kernel_dir, args.output_dir)
    if result.get("status") != "ok":
        print(f"\nStatus: {result.get('status')}")
        sys.exit(1)

