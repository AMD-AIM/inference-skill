"""
Runtime GEMM dispatch patching for vLLM on ROCm.

Injection method:
  1. This file is placed in OPTIMIZED_DIR.
  2. sitecustomize.py in OPTIMIZED_DIR does `import gemm_patch`.
  3. PYTHONPATH=OPTIMIZED_DIR:... ensures sitecustomize.py runs in ALL
     spawned processes (critical for TP>1 with VLLM_WORKER_MULTIPROC_METHOD=spawn).

Patch point:
  `vllm.model_executor.layers.linear.dispatch_unquantized_gemm`
  — this is the name binding that UnquantizedLinearMethod.apply() calls
    on every forward pass. Patching the underlying impl function does NOT
    work because torch.library custom ops capture references at registration.

Safety:
  - Clean fallback to original on ANY exception in the Triton kernel.
  - Call stats written to .call_stats.json every 100 calls.
  - Only patches on ROCm (gfx* arch) — no-op on NVIDIA if no best_kernel.py.

Constraint 3 satisfied: no files under /opt/, /usr/, or pip packages modified.

Note on server startup sequence (Phase 1 / Phase 5):
  Before starting the patched server, always kill VLLM::EngineCore workers
  FIRST (they hold GPU memory), then kill the API server. This is independent
  of this patch file — the startup scripts handle the kill order.
"""

import importlib
import json
import os
import sys
import threading


class _PatchFinder:
    """meta_path hook that intercepts vllm.model_executor.layers.linear import."""
    _done = False

    def find_module(self, name, path=None):
        if name == "vllm.model_executor.layers.linear" and not self._done:
            return self
        return None

    def load_module(self, name):
        # Mark done BEFORE loading to prevent infinite recursion
        self.__class__._done = True
        sys.meta_path.remove(self)

        # Load the real module
        mod = importlib.import_module(name)

        # Apply the patch
        try:
            _patch_gemm_dispatch(mod)
        except Exception as e:
            print(f"[gemm_patch] WARNING: patch failed: {e}", flush=True)

        return mod


def _patch_gemm_dispatch(linear_mod):
    """Replace dispatch_unquantized_gemm with Triton-dispatching version."""
    import torch
    import importlib.util

    patch_dir   = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(patch_dir, "gemm", "best_kernel.py")
    stats_path  = os.path.join(patch_dir, ".call_stats.json")

    if not os.path.exists(kernel_path):
        # No optimized kernel available — don't patch
        return

    # Load optimized kernel
    spec = importlib.util.spec_from_file_location("_opt_kernel", kernel_path)
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    triton_fn = kmod.optimized

    # Capture original dispatch
    original_dispatch = linear_mod.dispatch_unquantized_gemm
    original_gemm     = original_dispatch()

    # Stats tracking
    stats = {"triton": 0, "fallback": 0, "errors": 0}
    lock  = threading.Lock()

    def _save_stats():
        try:
            with open(stats_path, "w") as f:
                json.dump({"pid": os.getpid(), **stats}, f)
        except Exception:
            pass

    def _optimized_gemm(layer, x, weight, bias=None):
        """
        Drop-in replacement for rocm_unquantized_gemm.

        vLLM convention: weight shape is (out_features, in_features).
        Triton convention: A @ B where A is (M, K), B is (K, N).
        So we pass weight.t() as B.
        """
        try:
            x_2d = x.reshape(-1, x.size(-1))          # (M, K)
            out  = triton_fn(x_2d, weight.t())          # (M, N)
            if bias is not None:
                out = out + bias
            with lock:
                stats["triton"] += 1
                if stats["triton"] % 100 == 0:
                    _save_stats()
            # Restore original batch dimensions
            return out.reshape(*x.shape[:-1], weight.shape[0])

        except Exception as e:
            with lock:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    print(f"[gemm_patch] triton error (falling back): {e}", flush=True)
                stats["fallback"] += 1
                if stats["fallback"] % 100 == 0:
                    _save_stats()
            return original_gemm(layer, x, weight, bias)

    def _patched_dispatch():
        return _optimized_gemm

    linear_mod.dispatch_unquantized_gemm = _patched_dispatch
    _save_stats()
    print(f"[gemm_patch] Installed. pid={os.getpid()} kernel={kernel_path}", flush=True)


# Install the meta_path hook immediately when this module is imported
sys.meta_path.insert(0, _PatchFinder())
