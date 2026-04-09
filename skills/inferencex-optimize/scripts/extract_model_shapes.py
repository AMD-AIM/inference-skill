#!/usr/bin/env python3
"""Extract model shapes from TraceLens GEMM.csv or HuggingFace config.

Usage: python3 extract_model_shapes.py --output <path> [--gemm-csv <path>] [--config-json <path>]
"""

import argparse
import collections
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Extract model shapes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gemm-csv", default="")
    parser.add_argument("--config-json", default="")
    args = parser.parse_args()

    shapes = {}

    if args.gemm_csv and os.path.isfile(args.gemm_csv):
        k_counts = collections.Counter()
        n_counts = collections.Counter()
        with open(args.gemm_csv) as f:
            for row in csv.DictReader(f):
                try:
                    K = int(float(row.get("param: K", 0) or 0))
                    N = int(float(row.get("param: N", 0) or 0))
                    if K > 0:
                        k_counts[K] += 1
                    if N > 0:
                        n_counts[N] += 1
                except (ValueError, TypeError):
                    continue
        if k_counts:
            shapes["hidden_size"] = k_counts.most_common(1)[0][0]
        if n_counts:
            top_n = [n for n, _ in n_counts.most_common(5)]
            shapes["intermediate_size"] = max(top_n) if top_n else 11008
        print(f"Inferred from GEMM.csv: {shapes}")
    elif args.config_json and os.path.isfile(args.config_json):
        try:
            cfg = json.load(open(args.config_json))
            model_name = cfg.get("model", cfg.get("hf_model", ""))
            if model_name:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
                tc = hf_cfg.get("text_config", hf_cfg)
                shapes = {
                    "hidden_size": tc.get("hidden_size", 4096),
                    "intermediate_size": tc.get("intermediate_size", 11008),
                    "num_attention_heads": tc.get("num_attention_heads", 32),
                    "num_key_value_heads": tc.get("num_key_value_heads", 8),
                    "head_dim": tc.get(
                        "head_dim",
                        tc.get("hidden_size", 4096) // max(tc.get("num_attention_heads", 1), 1),
                    ),
                }
                num_experts = tc.get("num_local_experts", tc.get("num_experts"))
                if num_experts is not None:
                    shapes["num_experts"] = num_experts
                    shapes["num_experts_per_tok"] = tc.get(
                        "num_experts_per_tok", tc.get("top_k", tc.get("num_selected_experts"))
                    )
                    moe_inter = tc.get("moe_intermediate_size", tc.get("expert_intermediate_size"))
                    if moe_inter is not None:
                        shapes["moe_intermediate_size"] = moe_inter
                print(f"Loaded from HuggingFace config: {shapes}")
        except Exception as e:
            print(f"Could not load model config: {e}")
            shapes = {"hidden_size": 4096, "intermediate_size": 11008}
            print(f"Using defaults: {shapes}")
    else:
        shapes = {"hidden_size": 4096, "intermediate_size": 11008}
        print(f"Using defaults: {shapes}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(shapes, f, indent=2)
    print("Saved model_shapes.json")


if __name__ == "__main__":
    main()
