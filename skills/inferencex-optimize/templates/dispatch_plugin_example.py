#!/usr/bin/env python3
"""Reference template for a shape-aware SGLang dispatch plugin.

This file is a starting point for Phase 8 Step 1.5 when a GEAK result changes
dispatch selection rather than replacing a kernel directly. Adapt the target
module, method name, decision table, and override hooks for the exact runtime
version inside the benchmark container.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

TARGET_MODULE = "sglang.srt.layers.linear"
TARGET_CLASS = "UnquantizedLinearMethod"


@dataclass(frozen=True)
class DispatchDecision:
    """One per-shape override emitted from micro-benchmark results."""

    backend: str
    note: str = ""


SHAPE_DECISIONS: Dict[Tuple[int, int, int], DispatchDecision] = {
    # (M, K, N): DispatchDecision(backend="ck", note="small-K winner"),
}


def lookup_decision(m_dim: int, k_dim: int, n_dim: int) -> Optional[DispatchDecision]:
    return SHAPE_DECISIONS.get((m_dim, k_dim, n_dim))


def apply_dispatch_override(decision: DispatchDecision):
    """Set the framework/backend knob for the next GEMM call.

    Return any state needed to restore the previous behavior afterward.
    """

    del decision
    return None


def restore_dispatch_override(previous_state) -> None:
    """Restore the prior dispatch state after the call completes."""

    del previous_state


def patch_dispatch() -> None:
    target_module = importlib.import_module(TARGET_MODULE)
    original_class = getattr(target_module, TARGET_CLASS)

    class OptimizedDispatch(original_class):
        def apply(self, layer, x, bias=None):
            out_features, in_features = layer.weight.shape
            m_dim = x.numel() // max(in_features, 1)
            decision = lookup_decision(m_dim, in_features, out_features)
            if decision is None:
                return super().apply(layer, x, bias=bias)

            previous_state = apply_dispatch_override(decision)
            try:
                return super().apply(layer, x, bias=bias)
            finally:
                restore_dispatch_override(previous_state)

    setattr(target_module, TARGET_CLASS, OptimizedDispatch)


if __name__ == "__main__":
    patch_dispatch()
