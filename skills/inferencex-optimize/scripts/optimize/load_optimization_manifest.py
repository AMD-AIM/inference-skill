#!/usr/bin/env python3
"""Load and filter optimization manifest, printing prioritized kernel list.

Usage: python3 load_optimization_manifest.py \
    --manifest <path> --geak-mode <mode> [--optimize-scope <all|fused_only>]
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Load optimization manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--geak-mode", required=True, choices=["full", "triton_only", "manual"])
    parser.add_argument("--optimize-scope", default="all", choices=["all", "fused_only"])
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))

    enabled = [o for o in manifest["optimizations"] if o.get("optimize", False)]
    if not enabled:
        enabled = [o for o in manifest["optimizations"] if o.get("enabled")]
        enabled = [o for o in enabled if o.get("profiling_pct", 100) >= 1.0]

    enabled = [o for o in enabled if o.get("geak_mode", "simple") != "skip"]

    if args.optimize_scope == "fused_only":
        enabled = [o for o in enabled if o.get("type") == "fused"]
        print(f"OPTIMIZE_SCOPE=fused_only: filtered to {len(enabled)} fused entries")

    if args.geak_mode == "triton_only":
        enabled = [o for o in enabled if o.get("geak_mode") == "simple"]
        print(f"GEAK_MODE=triton_only: filtered to {len(enabled)} simple-mode entries")

    for o in enabled:
        kt = o.get("kernel_type", "unknown")
        if o.get("geak_mode") == "simple" and kt != "triton":
            print(f"WARNING: {o['name']} has geak_mode=simple but kernel_type={kt} — overriding to kernel-url")
            o["geak_mode"] = "kernel-url"
            o["geak_config"] = "mini_kernel.yaml"

    enabled.sort(key=lambda o: -o.get("priority_score", o.get("profiling_pct", 0)))

    simple = [o for o in enabled if o.get("geak_mode") == "simple"]
    kernel_url = [o for o in enabled if o.get("geak_mode") == "kernel-url"]

    print(f"Total entries: {len(manifest['optimizations'])}")
    print(f"Enabled (impact >= 1.0%): {len(enabled)}")
    print(f"  simple mode: {len(simple)}")
    print(f"  kernel-url mode: {len(kernel_url)}")
    print()
    for o in enabled:
        pct = o.get("profiling_pct", 0)
        mode = o.get("geak_mode", "?")
        score = o.get("priority_score", 0)
        eff = o.get("roofline_efficiency")
        eff_str = f" eff={eff:.0f}%" if eff is not None else ""
        print(f"  [score={score:5.1f}] {o['name']:40s} pct={pct:5.1f}%  {o.get('kernel_type', '?'):15s}  mode={mode}{eff_str}")


if __name__ == "__main__":
    main()
