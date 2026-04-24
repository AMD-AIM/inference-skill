#!/usr/bin/env python3
"""Resolve a runtime kernel symbol to its upstream source coordinates.

Reads `resources/kernel_source_map.yaml` and matches the given symbol against
each entry's `symbol_pattern` (glob-style; first match wins). Returns the
entry as JSON on stdout. Used by Phase 6 (`upstream-resolve`) to build the
optimization manifest.

Usage:
  resolve_upstream_source.py --symbol <name> \
      [--map resources/kernel_source_map.yaml] \
      [--pins resources/library_pins.yaml] \
      [--vllm-version v0.19.1]

Exits non-zero (1) when no entry matches; downstream code records this as
`library: unknown` and adds the symbol to `unresolved_kernels.json`.
"""

import argparse
import fnmatch
import json
import os
import sys

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MAP = os.path.join(
    REPO_ROOT, "skills", "inference-optimize", "resources", "kernel_source_map.yaml"
)
DEFAULT_PINS = os.path.join(
    REPO_ROOT, "skills", "inference-optimize", "resources", "library_pins.yaml"
)


def load_map(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc.get("entries", [])


def load_pins(path, vllm_version):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        doc = yaml.safe_load(f)
    for block in doc.get("pins", []):
        if block.get("vllm_version") == vllm_version:
            return block.get("pins", {})
    return {}


def resolve(symbol, entries, pins):
    for entry in entries:
        pattern = entry.get("symbol_pattern", "")
        if fnmatch.fnmatchcase(symbol, pattern):
            resolved = dict(entry)
            resolved["matched_symbol"] = symbol
            lib = entry.get("library")
            if lib in pins:
                resolved["pinned_commit"] = pins[lib]
            return resolved
    return None


def main():
    parser = argparse.ArgumentParser(description="Resolve kernel symbol to upstream source")
    parser.add_argument("--symbol", required=True, help="Runtime kernel symbol")
    parser.add_argument("--map", default=DEFAULT_MAP, help="kernel_source_map.yaml path")
    parser.add_argument("--pins", default=DEFAULT_PINS, help="library_pins.yaml path")
    parser.add_argument("--vllm-version", default="v0.19.1")
    args = parser.parse_args()

    entries = load_map(args.map)
    pins = load_pins(args.pins, args.vllm_version)
    resolved = resolve(args.symbol, entries, pins)

    if resolved is None:
        json.dump(
            {
                "matched_symbol": args.symbol,
                "library": "unknown",
                "source_form": "unknown",
                "bucket": None,
                "geak_strategy": "unfeasible_record_only",
                "skip_reason": "unresolved_unknown_symbol",
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 1

    json.dump(resolved, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
