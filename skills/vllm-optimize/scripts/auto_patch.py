"""
Auto-patch vLLM GEMM dispatch with optimized Triton kernel.

Verified integration method:
  - meta_path import hook watches for vllm.model_executor.layers.linear
  - After load, replaces linear_mod.dispatch_unquantized_gemm
  - Tracks call counts in .call_stats.json

Bundled with the vllm-optimize skill. Copied to OPTIMIZED_DIR at Phase 8.
Requires: OPTIMIZED_DIR/gemm/best_kernel.py with optimized(A, B) function.

Injection: set PYTHONPATH=<OPTIMIZED_DIR>:$PYTHONPATH + sitecustomize.py
"""
import sys
import os
import importlib
import threading
import json


class _PatchFinder:
    _done = False

    def find_module(self, name, path=None):
        if name == "vllm.model_executor.layers.linear" and not self._done:
            return self
        return None

    def load_module(self, name):
        self.__class__._done = True
        sys.meta_path.remove(self)
        mod = importlib.import_module(name)
        sys.meta_path.insert(0, self)
        _apply_patch(mod)
        return mod


def _apply_patch(linear_mod):
    import torch
    import importlib.util

    patch_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(patch_dir, "gemm", "best_kernel.py")
    stats_path = os.path.join(patch_dir, ".call_stats.json")

    if not os.path.exists(kernel_path):
        print(f"[triton_patch] No kernel at {kernel_path}, skipping", flush=True)
        return

    spec = importlib.util.spec_from_file_location("_tk", kernel_path)
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    triton_fn = kmod.optimized

    original_dispatch = linear_mod.dispatch_unquantized_gemm
    original_gemm = original_dispatch()

    stats = {"triton": 0, "fallback": 0, "error": 0}
    lock = threading.Lock()

    def _save():
        try:
            with open(stats_path, "w") as f:
                json.dump({"pid": os.getpid(), **stats}, f)
        except Exception:
            pass

    def my_gemm(layer, x, weight, bias=None):
        n = x.numel() // x.size(-1)
        try:
            x_2d = x.reshape(-1, x.size(-1))
            out = triton_fn(x_2d, weight.t())
            if bias is not None:
                out = out + bias
            with lock:
                stats["triton"] += 1
                if stats["triton"] % 100 == 0:
                    _save()
            return out.reshape(*x.shape[:-1], weight.shape[0])
        except Exception as e:
            with lock:
                stats["error"] += 1
            if stats["error"] <= 3:
                print(f"[triton_patch] Error: {e}", flush=True)

        with lock:
            stats["fallback"] += 1
            if stats["fallback"] % 100 == 0:
                _save()
        return original_gemm(layer, x, weight, bias)

    def patched_dispatch():
        return my_gemm

    linear_mod.dispatch_unquantized_gemm = patched_dispatch
    _save()
    print(f"[triton_patch] Installed pid={os.getpid()} kernel={kernel_path}", flush=True)


sys.meta_path.insert(0, _PatchFinder())
