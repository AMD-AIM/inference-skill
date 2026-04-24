"""scripts/integrate -- Phase 8 integration utilities.

Replaces the legacy plugin-injection path. The library-rebuild approach
rebuilds each fork in-place via `pip install -e`, then verifies dispatch
with rocprofv3 kernel-trace, then runs the standard vLLM e2e benchmark
without any plugin wiring.
"""
