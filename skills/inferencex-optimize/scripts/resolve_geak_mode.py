#!/usr/bin/env python3
"""Resolve effective GEAK mode from user preference and runtime environment.

Usage: python3 resolve_geak_mode.py --user-mode <auto|full|triton_only|manual> --env-info <path>
Outputs: EFFECTIVE_GEAK_MODE=<mode>
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Resolve GEAK mode")
    parser.add_argument("--user-mode", required=True, choices=["auto", "full", "triton_only", "manual"])
    parser.add_argument("--env-info", required=True, help="Path to env_info.json")
    args = parser.parse_args()

    geak = False
    api_key = False
    if os.path.isfile(args.env_info):
        env = json.load(open(args.env_info))
        geak = env.get("geak_available", False)
        api_key = env.get("llm_api_key_set", False)

    if args.user_mode == "manual":
        effective = "manual"
    elif not geak:
        print("WARNING: GEAK not available — falling back to manual mode")
        effective = "manual"
    elif not api_key:
        print("WARNING: GEAK installed but no LLM API key — falling back to manual mode")
        print("Set AMD_LLM_API_KEY, LLM_GATEWAY_KEY, or ANTHROPIC_API_KEY to enable GEAK")
        effective = "manual"
    elif args.user_mode == "triton_only":
        effective = "triton_only"
    elif args.user_mode == "full":
        effective = "full"
    else:
        effective = "full"

    print(f"User GEAK_MODE: {args.user_mode}")
    print(f"EFFECTIVE_GEAK_MODE={effective}")
    print(f"  GEAK available: {geak}, API key: {api_key}")


if __name__ == "__main__":
    main()
