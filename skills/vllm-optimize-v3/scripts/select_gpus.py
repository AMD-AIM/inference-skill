#!/usr/bin/env python3
"""
Select the N GPUs with the most free VRAM.

Ranks GPUs by free VRAM (descending) so that GPUs with large resident
allocations (e.g., from a previous crashed run) are skipped automatically.
Falls back to utilization% ranking if VRAM info is unavailable.

Usage:
    python select_gpus.py 1      # returns "0"
    python select_gpus.py 2      # returns "0,3" (two with most free VRAM)

Prints a comma-separated string of GPU IDs to stdout.
"""
import subprocess
import sys


def get_amd_free_vram():
    """Return list of (gpu_id, free_bytes) for AMD GPUs via rocm-smi --showmeminfo vram."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        import re
        totals, used = {}, {}
        for line in r.stdout.splitlines():
            m = re.search(r"GPU\[(\d+)\].*VRAM Total Memory.*?(\d+)", line)
            if m:
                totals[int(m.group(1))] = int(m.group(2))
            m = re.search(r"GPU\[(\d+)\].*VRAM Total Used Memory.*?(\d+)", line)
            if m:
                used[int(m.group(1))] = int(m.group(2))
        result = []
        for gpu_id in sorted(totals.keys()):
            free = totals[gpu_id] - used.get(gpu_id, 0)
            result.append((gpu_id, free))
        return result if result else None
    except Exception:
        return None


def get_amd_utilization():
    """Fallback: return list of (gpu_id, utilization_pct) for AMD GPUs."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse", "--csv"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        utils = []
        for line in r.stdout.splitlines():
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


def get_nvidia_free_vram():
    """Return list of (gpu_id, free_bytes) for NVIDIA GPUs via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return None
        result = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    result.append((int(parts[0]), int(parts[1]) * 1024 * 1024))
                except ValueError:
                    continue
        return result if result else None
    except Exception:
        return None


def get_torch_fallback(n_gpus):
    """Return [(0, 0), (1, 0), ...] when no monitoring tool is available."""
    try:
        import torch
        count = torch.cuda.device_count()
        return [(i, 0) for i in range(count)]
    except Exception:
        return [(i, 0) for i in range(n_gpus)]


def select_gpus(n: int) -> str:
    """Select n GPUs with the most free VRAM, return comma-separated IDs."""
    free_vram = get_amd_free_vram() or get_nvidia_free_vram()

    if free_vram:
        # Sort by free VRAM descending — most free memory first
        free_vram.sort(key=lambda x: x[1], reverse=True)
        selected = [str(gpu_id) for gpu_id, _ in free_vram[:n]]
        return ",".join(selected)

    # Fallback: sort by utilization ascending (least busy first)
    utils = get_amd_utilization()
    if utils:
        utils.sort(key=lambda x: x[1])
        return ",".join(str(gpu_id) for gpu_id, _ in utils[:n])

    # Last resort: 0..n-1
    fallback = get_torch_fallback(8)
    return ",".join(str(gpu_id) for gpu_id, _ in fallback[:n])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(select_gpus(n))
