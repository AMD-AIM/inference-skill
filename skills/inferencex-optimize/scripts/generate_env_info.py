#!/usr/bin/env python3
"""Generate env_info.json with GPU, GEAK, and API key detection results.

Usage: python3 generate_env_info.py --output <path> --geak-dir <dir>
Reads GEAK_AVAILABLE, GEAK_OE_AVAILABLE, LLM_API_KEY_SET from environment.
"""

import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Generate env_info.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--geak-dir", default="")
    args = parser.parse_args()

    gpu_vendor = "unknown"
    gpu_arch = "unknown"
    gpu_count = 0

    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import re
            arches = re.findall(r"gfx\w+", result.stdout)
            if arches:
                gpu_vendor = "amd"
                gpu_arch = arches[0].lower()
                gpu_count = len(set(arches))
    except Exception:
        pass

    if gpu_vendor == "unknown":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpu_vendor = "nvidia"
                gpu_count = int(result.stdout.strip().split("\n")[0])
            result2 = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result2.returncode == 0:
                cap = result2.stdout.strip().split("\n")[0].replace(".", "")
                gpu_arch = f"sm_{cap}"
        except Exception:
            pass

    env_info = {
        "gpu_vendor": gpu_vendor,
        "gpu_arch": gpu_arch,
        "gpu_count": gpu_count,
        "geak_available": os.environ.get("GEAK_AVAILABLE", "false").lower() == "true",
        "geak_dir": args.geak_dir,
        "geak_oe_available": os.environ.get("GEAK_OE_AVAILABLE", "false").lower() == "true",
        "llm_api_key_set": os.environ.get("LLM_API_KEY_SET", "false").lower() == "true",
        "runtime_type": "docker",
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(env_info, f, indent=2)
    print(json.dumps(env_info, indent=2))


if __name__ == "__main__":
    main()
