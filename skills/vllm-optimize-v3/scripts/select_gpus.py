#!/usr/bin/env python3
"""
Select the N least-utilized GPUs.

Usage:
    python select_gpus.py 1      # returns "0"
    python select_gpus.py 2      # returns "0,3" (two least busy)

Prints a comma-separated string of GPU IDs to stdout.
"""
import subprocess
import sys


def get_amd_utilization():
    """Return list of (gpu_id, utilization_pct) for AMD GPUs."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse", "--csv"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        utils = []
        for line in r.stdout.splitlines():
            # CSV format: GPU[N], utilization%
            line = line.strip()
            if not line or line.startswith("device") or line.startswith("GPU"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0].replace("GPU[", "").replace("]", "").strip())
                    util   = float(parts[1].replace("%", "").strip())
                    utils.append((gpu_id, util))
                except (ValueError, IndexError):
                    continue
        return utils if utils else None
    except Exception:
        return None


def get_amd_utilization_text():
    """Fallback: parse rocm-smi --showuse text output."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        utils = []
        for line in r.stdout.splitlines():
            if "GPU[" in line and "%" in line:
                import re
                m_id  = re.search(r"GPU\[(\d+)\]", line)
                m_pct = re.search(r"(\d+\.?\d*)%", line)
                if m_id and m_pct:
                    utils.append((int(m_id.group(1)), float(m_pct.group(1))))
        return utils if utils else None
    except Exception:
        return None


def get_nvidia_utilization():
    """Return list of (gpu_id, utilization_pct) for NVIDIA GPUs."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        utils = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    utils.append((int(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return utils if utils else None
    except Exception:
        return None


def get_torch_fallback(n_gpus):
    """Return [(0, 0), (1, 0), ...] when no monitoring tool is available."""
    try:
        import torch
        count = torch.cuda.device_count()
        return [(i, 0.0) for i in range(count)]
    except Exception:
        return [(i, 0.0) for i in range(n_gpus)]


def select_gpus(n: int) -> str:
    """Select n least-utilized GPUs, return comma-separated IDs."""
    utils = (
        get_amd_utilization()
        or get_amd_utilization_text()
        or get_nvidia_utilization()
        or get_torch_fallback(8)
    )

    if not utils:
        # Emergency fallback: 0..n-1
        return ",".join(str(i) for i in range(n))

    # Sort by utilization ascending, take first n
    utils.sort(key=lambda x: x[1])
    selected = [str(gpu_id) for gpu_id, _ in utils[:n]]
    return ",".join(selected)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(select_gpus(n))
