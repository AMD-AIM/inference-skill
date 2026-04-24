#!/usr/bin/env python3
"""Clone or update upstream library forks at pinned commits.

For each library in `library_pins.yaml` whose tag is referenced by at least
one in-scope kernel in the optimization manifest (or all libs if invoked
with `--all`), this script:

  1. Clones the upstream repo into `<output_dir>/forks/<lib>/` if absent;
  2. Fetches and checks out the pinned commit;
  3. Creates a `geak/` branch off the pinned commit (idempotent: if the
     branch exists, leaves it intact);
  4. Records the fork's coordinates in `<output_dir>/forks/manifest.json`.

Also records `ck_branch_merged_status`: a one-shot probe that asks GitHub
whether GEAK upstream's `feature/ck-preprocess-main` branch has been
merged into main. Used by Phase 6 to decide whether `ck_template` rows in
`kernel_source_map.yaml` may be promoted from Bucket B to Bucket A.

Idempotent: rerunning is safe and re-uses existing checkouts.

Usage:
  fork_upstream.py --output-dir <path> [--vllm-version v0.19.1] \
      [--libs vllm,aiter,fla] [--map <path>] [--pins <path>] [--all]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MAP = os.path.join(
    REPO_ROOT, "skills", "inference-optimize", "resources", "kernel_source_map.yaml"
)
DEFAULT_PINS = os.path.join(
    REPO_ROOT, "skills", "inference-optimize", "resources", "library_pins.yaml"
)

REPO_URLS = {
    "vllm": "https://github.com/vllm-project/vllm.git",
    "aiter": "https://github.com/ROCm/aiter.git",
    "fla": "https://github.com/sustcsonglin/flash-linear-attention.git",
    "composable_kernel": "https://github.com/ROCm/composable_kernel.git",
    "rocblas": "https://github.com/ROCm/rocBLAS.git",
    "hipblaslt": "https://github.com/ROCm/hipBLASLt.git",
    "pytorch": "https://github.com/pytorch/pytorch.git",
}

REBUILD_COMMANDS = {
    "vllm": "pip install -e . --no-build-isolation",
    "aiter": "AITER_REBUILD=1 pip install -e .",
    "fla": "pip install -e .",
    "composable_kernel": "cmake -S . -B build && cmake --build build -j",
    "rocblas": "cmake -S . -B build && cmake --build build -j && cmake --install build",
    "hipblaslt": "cmake -S . -B build && cmake --build build -j && cmake --install build",
    "pytorch": "pip install -e . --no-build-isolation",
}


def run(cmd, cwd=None, check=True, capture=False):
    """Subprocess helper. Returns CompletedProcess."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
    )


def load_map(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc.get("entries", [])


def load_pins(path, vllm_version):
    with open(path) as f:
        doc = yaml.safe_load(f)
    for block in doc.get("pins", []):
        if block.get("vllm_version") == vllm_version:
            return block.get("pins", {})
    return {}


def libraries_referenced(entries, manifest_path):
    """Return the set of library tags referenced by in-scope kernels.

    If a manifest exists, restrict to its `optimize=true` kernels; else use
    every entry in the source map (intended for first-run bootstrap).
    """
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            doc = json.load(f)
        kernels = doc.get("optimizations", doc.get("kernels", []))
        return {k.get("library") for k in kernels if k.get("optimize", True)}
    return {e.get("library") for e in entries if e.get("library")}


def clone_or_update(repo_url, fork_path, commit):
    """Idempotently clone repo_url into fork_path and check out commit."""
    if not os.path.isdir(fork_path):
        os.makedirs(os.path.dirname(fork_path), exist_ok=True)
        run(["git", "clone", repo_url, fork_path])
    run(["git", "fetch", "--all", "--tags", "--quiet"], cwd=fork_path)
    run(["git", "checkout", "--quiet", commit], cwd=fork_path)


def ensure_geak_branch(fork_path):
    """Create the `geak/` branch off the current HEAD if it does not exist."""
    res = run(
        ["git", "rev-parse", "--verify", "--quiet", "geak/main"],
        cwd=fork_path,
        check=False,
        capture=True,
    )
    if res.returncode != 0:
        run(["git", "checkout", "-b", "geak/main"], cwd=fork_path)
    else:
        run(["git", "checkout", "geak/main"], cwd=fork_path)


def probe_ck_branch_merged():
    """Return True iff GEAK upstream's feature/ck-preprocess-main is merged.

    Best-effort: queries the GitHub API for the comparison. Network or
    rate-limit failures degrade to `False` (the conservative default that
    keeps `ck_template` rows in Bucket B).
    """
    url = "https://api.github.com/repos/AMD-AGI/GEAK/compare/main...feature/ck-preprocess-main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "inference-skill"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"probe_failed: {exc}"
    behind = data.get("behind_by", 0)
    ahead = data.get("ahead_by", 0)
    merged = ahead == 0 and behind > 0
    return merged, f"ahead={ahead} behind={behind}"


def main():
    parser = argparse.ArgumentParser(description="Clone/update upstream forks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vllm-version", default="v0.19.1")
    parser.add_argument("--libs", default=None,
                        help="Comma-separated subset of library tags to clone")
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--pins", default=DEFAULT_PINS)
    parser.add_argument("--manifest", default=None,
                        help="optimization_manifest.json (restrict to in-scope libs)")
    parser.add_argument("--all", action="store_true",
                        help="Clone every pinned library (ignore manifest scope)")
    args = parser.parse_args()

    entries = load_map(args.map)
    pins = load_pins(args.pins, args.vllm_version)
    if not pins:
        sys.stderr.write(
            f"ERROR: no pins for vllm_version={args.vllm_version} in {args.pins}\n"
        )
        return 2

    if args.libs:
        wanted = {x.strip() for x in args.libs.split(",") if x.strip()}
    elif args.all:
        wanted = set(pins.keys())
    else:
        wanted = libraries_referenced(entries, args.manifest)

    forks_root = os.path.join(args.output_dir, "forks")
    os.makedirs(forks_root, exist_ok=True)

    manifest = {"forks": {}}

    for lib in sorted(wanted):
        if lib not in pins:
            sys.stderr.write(f"WARNING: library '{lib}' has no pin entry; skipping\n")
            continue
        repo_url = REPO_URLS.get(lib)
        if repo_url is None:
            sys.stderr.write(f"WARNING: library '{lib}' has no known repo URL; skipping\n")
            continue
        commit = pins[lib]
        fork_path = os.path.join(forks_root, lib)
        try:
            clone_or_update(repo_url, fork_path, commit)
            ensure_geak_branch(fork_path)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"ERROR cloning {lib}: {exc}\n")
            manifest["forks"][lib] = {
                "repo_url": repo_url,
                "pinned_commit": commit,
                "fork_path": fork_path,
                "dirty": True,
                "error": str(exc),
            }
            continue
        manifest["forks"][lib] = {
            "repo_url": repo_url,
            "pinned_commit": commit,
            "fork_path": fork_path,
            "dirty": False,
            "rebuild_command": REBUILD_COMMANDS.get(lib, ""),
        }

    merged, ck_status_detail = probe_ck_branch_merged()
    manifest["ck_branch_merged_status"] = merged
    manifest["ck_branch_merged_detail"] = ck_status_detail
    manifest["vllm_version"] = args.vllm_version

    out_path = os.path.join(forks_root, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
