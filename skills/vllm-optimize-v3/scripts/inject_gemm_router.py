#!/usr/bin/env python3
"""
GEMM router injection for vLLM ROCm unquantized GEMM.

Loaded via sitecustomize.py at server startup.  Patches
  vllm.model_executor.layers.utils.rocm_unquantized_gemm
so that every unquantized GEMM call is dispatched to the implementation
that was measured fastest offline (recorded in routing_table.json).

Dispatch priority:
  1. routing_table match  →  wvSplitK / llmm1 / linear  (offline-tuned choice)
  2. no match             →  original vllm impl          (safe fallback)

Why patch here and not via TunableOps CSV:
  - TunableOps only intercepts torch.nn.functional.linear calls.
  - vLLM hard-routes n<=4 (conc=1/4 decode) through wvSplitK/LLMM1,
    bypassing TunableOps entirely.
  - This patch sits above the dispatch logic and covers ALL batch sizes.
  - TunableOps can still be active; "linear" entries benefit from it if
    a tuned CSV is also loaded.
"""
import json
import os
import sys

ROUTING_TABLE_PATH = os.environ.get(
    "VLLM_GEMM_ROUTING_TABLE",
    os.path.join(os.path.dirname(__file__), "routing_table.json"),
)

_routing: dict = {}   # key: "n,k,m" → {"impl": str, ...}
_loaded = False
_cu_count = None
_original_fn = None


def _load_routing():
    global _routing, _loaded
    if _loaded:
        return
    _loaded = True
    if not os.path.exists(ROUTING_TABLE_PATH):
        print(f"[gemm_router] WARNING: routing table not found at {ROUTING_TABLE_PATH}",
              file=sys.stderr)
        return
    try:
        data = json.load(open(ROUTING_TABLE_PATH))
        _routing = data.get("routing", {})
        n_entries = len(_routing)
        impls = {}
        for v in _routing.values():
            imp = v.get("impl", "?")
            impls[imp] = impls.get(imp, 0) + 1
        print(f"[gemm_router] Loaded {n_entries} routing entries: {impls}",
              flush=True)
    except Exception as e:
        print(f"[gemm_router] ERROR loading routing table: {e}", file=sys.stderr)


def _get_cu_count():
    global _cu_count
    if _cu_count is not None:
        return _cu_count
    try:
        from vllm._custom_ops import get_cu_count
        _cu_count = get_cu_count()
    except Exception:
        import torch
        _cu_count = torch.cuda.get_device_properties(0).multi_processor_count
    return _cu_count


def patched_rocm_unquantized_gemm(layer, x, weight, bias=None):
    """Replacement for vllm.model_executor.layers.utils.rocm_unquantized_gemm."""
    _load_routing()

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]
    key = f"{n},{k},{m}"

    impl = _routing.get(key, {}).get("impl") if _routing else None

    if impl == "wvSplitK" and m > 8 and 0 < n <= 4:
        try:
            import torch
            from vllm import _custom_ops as ops
            x_view = x.reshape(-1, x.size(-1))
            out = ops.wvSplitK(weight, x_view, _get_cu_count(), bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        except Exception:
            pass  # fall through to original

    elif impl == "llmm1" and n == 1 and k <= 8192 and m % 4 == 0 and bias is None:
        try:
            import torch
            from vllm import _custom_ops as ops
            x_view = x.reshape(-1, x.size(-1))
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        except Exception:
            pass  # fall through to original

    elif impl == "linear":
        import torch
        return torch.nn.functional.linear(x, weight, bias)

    # Fallback: original vLLM dispatch (wvSplitK/LLMM1/linear decided by vLLM)
    if _original_fn is not None:
        return _original_fn(layer, x, weight, bias)
    import torch
    return torch.nn.functional.linear(x, weight, bias)


def install():
    """Monkey-patch vllm.model_executor.layers.utils.rocm_unquantized_gemm."""
    global _original_fn
    try:
        import vllm.model_executor.layers.utils as _utils
        _original_fn = _utils.rocm_unquantized_gemm
        _utils.rocm_unquantized_gemm = patched_rocm_unquantized_gemm
        print("[gemm_router] Installed: vllm.model_executor.layers.utils"
              ".rocm_unquantized_gemm patched", flush=True)
    except ImportError:
        # vLLM not yet imported; re-hook via sys.meta_path instead
        _install_import_hook()


class _VllmImportHook:
    """Deferred patch: fires when vllm.model_executor.layers.utils is imported."""
    def find_module(self, fullname, path=None):
        if fullname == "vllm.model_executor.layers.utils":
            return self
        return None

    def load_module(self, fullname):
        import importlib
        import sys as _sys
        # Remove ourselves so we don't recurse
        _sys.meta_path = [m for m in _sys.meta_path if not isinstance(m, _VllmImportHook)]
        mod = importlib.import_module(fullname)
        global _original_fn
        _original_fn = mod.rocm_unquantized_gemm
        mod.rocm_unquantized_gemm = patched_rocm_unquantized_gemm
        print("[gemm_router] Installed (deferred): rocm_unquantized_gemm patched",
              flush=True)
        return mod


def _install_import_hook():
    sys.meta_path.insert(0, _VllmImportHook())
    print("[gemm_router] Import hook installed (vLLM not yet loaded)", flush=True)
