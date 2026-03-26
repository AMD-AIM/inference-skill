#!/usr/bin/env python3
"""Select the N least-utilized GPUs. Supports AMD (rocm-smi) and NVIDIA (nvidia-smi).

Usage:
    select_gpus.py <num_gpus>                 # prints comma-separated GPU indices
    select_gpus.py <num_gpus> --docker-flags  # prints Docker device flags for isolation
"""
import os
import subprocess
import sys
import json


def get_amd_gpus():
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        gpus = {}
        for key, info in data.items():
            if not key.startswith("card"):
                continue
            idx = int(key.replace("card", ""))
            use_str = str(info.get("GPU use (%)", "0"))
            use = float(use_str.replace("%", "").strip())
            gpus[idx] = [idx, use, 0.0]

        try:
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if mem_result.returncode == 0:
                mem_data = json.loads(mem_result.stdout)
                for key, info in mem_data.items():
                    if not key.startswith("card"):
                        continue
                    idx = int(key.replace("card", ""))
                    if idx in gpus:
                        used_bytes = float(
                            str(info.get("VRAM Total Used Memory (B)", "0"))
                        )
                        gpus[idx][2] = used_bytes / (1024 ** 2)
        except Exception:
            pass

        return [tuple(v) for v in gpus.values()]
    except Exception:
        return None


def get_nvidia_gpus():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
                gpus.append((idx, util, mem))
        return gpus
    except Exception:
        return None


def get_amd_pci_bus_map():
    """Return {gpu_index: pci_bus_address} from rocm-smi --showbus."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showbus", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        pci_map = {}
        for key, info in data.items():
            if key.startswith("card"):
                idx = int(key.replace("card", ""))
                pci_map[idx] = info.get("PCI Bus", "").lower()
        return pci_map
    except Exception:
        return {}


def get_amd_render_devices(gpu_indices):
    """Map rocm-smi GPU indices to /dev/dri/renderD* device paths.

    MI355X has 1 physical + 7 XCP render devices per GPU. We match them
    by PCI bus address (physical) and XCP index (platform devices).
    """
    pci_map = get_amd_pci_bus_map()
    if not pci_map:
        return []

    drm_dir = "/sys/class/drm"
    if not os.path.isdir(drm_dir):
        return []

    # Build reverse map: PCI address -> set of render device names
    pci_renders = {}
    # XCP devices: amdgpu_xcp_N. 7 XCPs per GPU, numbered gpu_idx*7 .. gpu_idx*7+6
    xcp_renders = {}

    for entry in os.listdir(drm_dir):
        if not entry.startswith("renderD"):
            continue
        dev_link_path = os.path.join(drm_dir, entry, "device")
        if not os.path.islink(dev_link_path):
            continue
        dev_link = os.readlink(dev_link_path)
        basename = dev_link.split("/")[-1]

        if ":" in basename and "." in basename:
            pci_addr = basename.lower()
            pci_renders.setdefault(pci_addr, []).append(entry)
        elif basename.startswith("amdgpu_xcp_"):
            try:
                xcp_idx = int(basename.replace("amdgpu_xcp_", ""))
                xcp_renders[xcp_idx] = entry
            except ValueError:
                pass

    render_devices = []
    for gpu_idx in gpu_indices:
        pci = pci_map.get(gpu_idx, "")
        if not pci:
            print(f"  WARNING: No PCI bus for GPU {gpu_idx}", file=sys.stderr)
            continue

        # Physical render device (matched by PCI address)
        for rd in pci_renders.get(pci, []):
            render_devices.append(f"/dev/dri/{rd}")

        # XCP render devices (7 per GPU, sequentially numbered)
        xcp_start = gpu_idx * 7
        for xcp_idx in range(xcp_start, xcp_start + 7):
            rd = xcp_renders.get(xcp_idx)
            if rd:
                render_devices.append(f"/dev/dri/{rd}")

    return sorted(set(render_devices))


def docker_flags(vendor, selected_indices):
    """Return Docker CLI flags for GPU isolation."""
    if vendor == "nvidia":
        csv = ",".join(str(i) for i in selected_indices)
        return f"--gpus device={csv}"

    # AMD: mount /dev/kfd + per-GPU render devices
    render_devs = get_amd_render_devices(selected_indices)
    if not render_devs:
        print("WARNING: Could not resolve render devices; "
              "falling back to --device=/dev/dri", file=sys.stderr)
        return "--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

    dev_flags = " ".join(f"--device={rd}" for rd in render_devs)
    return f"--device=/dev/kfd {dev_flags} --group-add video --security-opt seccomp=unconfined"


def main():
    if len(sys.argv) < 2:
        print("Usage: select_gpus.py <num_gpus> [--docker-flags]", file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    want_docker_flags = "--docker-flags" in sys.argv

    gpus = get_amd_gpus()
    vendor = "amd"
    if gpus is None:
        gpus = get_nvidia_gpus()
        vendor = "nvidia"

    if gpus is None or len(gpus) == 0:
        selected = list(range(n))
        if want_docker_flags:
            print(docker_flags(vendor, selected))
        else:
            print(",".join(str(i) for i in selected))
        sys.exit(0)

    gpus.sort(key=lambda x: (x[1], x[2]))

    selected = [g[0] for g in gpus[:n]]
    selected_str = [str(i) for i in selected]

    if want_docker_flags:
        flags = docker_flags(vendor, selected)
        print(flags)
    else:
        print(",".join(selected_str))

    print(f"GPU vendor: {vendor}", file=sys.stderr)
    print(f"Total GPUs available: {len(gpus)}", file=sys.stderr)
    print(f"Selected {n} most free: {','.join(selected_str)}", file=sys.stderr)
    for g in gpus:
        marker = " <-- selected" if g[0] in selected else ""
        print(f"  GPU {g[0]}: util={g[1]}%, mem_used={g[2]:.1f}MB{marker}", file=sys.stderr)
    if want_docker_flags and vendor == "amd":
        render_devs = get_amd_render_devices(selected)
        print(f"Render devices ({len(render_devs)} total): "
              + " ".join(os.path.basename(r) for r in render_devs), file=sys.stderr)


if __name__ == "__main__":
    main()
