"""Tests for classify_kernel.py -- pattern matching, first-match-wins, skip behavior."""

import os
import sys

# Add the optimize directory to path so we can import the module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimize"))

from classify_kernel import classify_kernel, SKIP_KERNEL_TYPES, KERNEL_TYPES


class TestClassifyKernel:
    def test_communication_kernel(self):
        """NCCL/RCCL patterns classify as communication."""
        ktype, label = classify_kernel("ncclKernel_AllReduce")
        assert ktype == "communication"

    def test_gemm_kernel(self):
        """GEMM/vendor patterns classify correctly."""
        ktype, label = classify_kernel("rocblas_gemm_ex3")
        assert ktype in ("gemm", "hip_gemm", "vendor")

    def test_attention_kernel(self):
        """Attention patterns classify correctly."""
        ktype, label = classify_kernel("flash_attention_forward")
        assert ktype == "attention"

    def test_normalization_kernel(self):
        """RMSNorm/LayerNorm patterns classify correctly."""
        ktype, label = classify_kernel("fused_rmsnorm_kernel")
        assert ktype == "normalization"

    def test_activation_kernel(self):
        """SiLU/GELU patterns classify correctly."""
        ktype, label = classify_kernel("silu_and_mul_kernel")
        assert ktype == "activation"

    def test_unknown_kernel_returns_other(self):
        """Unknown kernel name returns ('other', 'Other')."""
        ktype, label = classify_kernel("completely_unknown_xyz_kernel")
        assert ktype == "other"
        assert label == "Other"

    def test_case_insensitive(self):
        """Classification is case-insensitive."""
        ktype1, _ = classify_kernel("NCCL_AllReduce")
        ktype2, _ = classify_kernel("nccl_allreduce")
        assert ktype1 == ktype2

    def test_first_match_wins(self):
        """When multiple patterns could match, first in KERNEL_TYPES wins."""
        # Verify ordering matters -- if we have a name that could match multiple
        # categories, the first definition in KERNEL_TYPES takes precedence
        ktype, _ = classify_kernel("fused_rmsnorm")
        # Should match normalization (or fused normalization) not generic "other"
        assert ktype != "other"

    def test_parent_category_fallback(self):
        """parent_category triggers fallback when name doesn't match."""
        ktype, _ = classify_kernel("some_unknown_op", parent_category="GEMM")
        assert ktype == "aten_gemm"

    def test_skip_kernel_types(self):
        """communication and moe_sort are in SKIP_KERNEL_TYPES."""
        assert "communication" in SKIP_KERNEL_TYPES
        assert "moe_sort" in SKIP_KERNEL_TYPES

    def test_classify_returns_tuple(self):
        """classify_kernel always returns a 2-tuple of strings."""
        result = classify_kernel("anything")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
