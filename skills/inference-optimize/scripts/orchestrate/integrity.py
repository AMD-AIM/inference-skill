"""Hash-based baseline integrity verification.

Computes SHA-256 hashes of baseline artifacts after the benchmark phase
and verifies they haven't changed before the integration phase. This
detects baseline drift that would invalidate speedup measurements.
"""
import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(path):
    """Compute SHA-256 hash of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path):
    """Hash a file or directory recursively. Returns list of [relpath, hash].

    Uses lists (not tuples) so the result survives JSON round-tripping.
    """
    entries = []
    if os.path.isfile(path):
        entries.append([os.path.basename(path), compute_file_hash(path)])
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, path)
                entries.append([relpath, compute_file_hash(fpath)])
    return sorted(entries)


def write_baseline_integrity(results_dir, baseline_files):
    """Write baseline integrity manifest after benchmark phase.

    Args:
        results_dir: Path to results directory (parent of baseline_integrity.json)
        baseline_files: List of paths (relative to output_dir parent of results_dir)
                       to include in integrity check. Directories are hashed recursively.

    Returns:
        Path to the written integrity file.
    """
    output_dir = os.path.dirname(results_dir)
    manifest = {"schema_version": "1.0", "files": {}}

    for rel_path in baseline_files:
        full_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(full_path):
            logger.warning(f"Baseline artifact not found: {rel_path}")
            continue
        hashes = _hash_path(full_path)
        manifest["files"][rel_path] = hashes

    integrity_path = os.path.join(results_dir, "baseline_integrity.json")
    with open(integrity_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Baseline integrity manifest written: {integrity_path}")
    return integrity_path


def verify_baseline_integrity(results_dir):
    """Verify baseline artifacts haven't changed since write_baseline_integrity.

    Returns:
        (match: bool, mismatches: list[dict])
        - match is True if all hashes match
        - mismatches is a list of {file, expected, actual} dicts
    """
    output_dir = os.path.dirname(results_dir)
    integrity_path = os.path.join(results_dir, "baseline_integrity.json")

    if not os.path.isfile(integrity_path):
        logger.warning("No baseline_integrity.json found -- backward compat, returning WARN")
        return True, []  # Missing = WARN not FAIL (backward compat)

    with open(integrity_path) as f:
        manifest = json.load(f)

    mismatches = []
    for rel_path, expected_hashes in manifest.get("files", {}).items():
        full_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(full_path):
            mismatches.append({"file": rel_path, "expected": "exists", "actual": "missing"})
            continue

        current_hashes = _hash_path(full_path)
        # Normalize: JSON roundtrip converts tuples to lists
        expected_normalized = [list(e) if isinstance(e, (list, tuple)) else e for e in expected_hashes]
        current_normalized = [list(e) for e in current_hashes]
        if current_normalized != expected_normalized:
            mismatches.append({
                "file": rel_path,
                "expected": expected_hashes,
                "actual": current_hashes,
            })

    match = len(mismatches) == 0
    if not match:
        logger.warning(f"Baseline integrity check failed: {len(mismatches)} mismatches")
    return match, mismatches
