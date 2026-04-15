#!/usr/bin/env python3
"""
Analyze PyTorch profiler Chrome-trace files for GPU kernel metrics.

Supports three modes:
  1. Full trace parsing — parses all kernel events from the first valid trace
  2. Gap analysis      — time-windowed, multi-rank kernel profiling via
                         the GapAnalyzer class (steady-state focus)
  3. TraceLens analysis — structured perf reports and multi-rank collective
                         analysis via the TraceLens CLI

Usage:
    # Full kernel summary (first valid trace in directory)
    python trace_analyzer.py <trace_dir>

    # Gap analysis with time window (multi-rank aware)
    python trace_analyzer.py <trace_dir> --gap-analysis [--start-pct 50 --end-pct 80]

    # Specify output directory for gap analysis CSVs/JSON
    python trace_analyzer.py <trace_dir> --gap-analysis --output-dir <dir>

    # Increase top-K kernel count / set minimum kernel duration
    python trace_analyzer.py <trace_dir> --gap-analysis --top-k 30 --min-dur 5

    # Generate clamped (time-windowed) trace files
    python trace_analyzer.py <trace_dir> --gap-analysis --clamped-traces

    # TraceLens analysis (auto-installs from GitHub if missing)
    python trace_analyzer.py <trace_dir> --tracelens [--num-ranks 8]

    # TraceLens with custom export format and output directory
    python trace_analyzer.py <trace_dir> --tracelens --export-csv --export-excel --output-dir <dir>

    # Compare two TraceLens reports
    python trace_analyzer.py <trace_dir> --tracelens-compare <dir1> <dir2> --labels baseline optimized

Handles both plain .json and gzip-compressed .json.gz trace files.
Uses Chrome Trace Event format (traceEvents with cat="kernel").
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# TraceLens CLI command names (use underscores, not hyphens)
CLI_GENERATE_REPORT = "TraceLens_generate_perf_report_pytorch"
CLI_MULTI_RANK_COLLECTIVE = "TraceLens_generate_multi_rank_collective_report_pytorch"
CLI_COMPARE_REPORTS = "TraceLens_compare_perf_reports_pytorch"
TRACELENS_INSTALL_URL = "git+https://github.com/AMD-AIG-AIMA/TraceLens.git"
TRACELENS_INTERNAL_REPO = "git@github.com:AMD-AGI/TraceLens-internal.git"
TRACELENS_LOCAL_DIR = os.path.join(os.path.expanduser("~"), "TraceLens-internal")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelMetrics:
    """Simple per-kernel summary used by the full-trace parser."""
    name: str
    time_ms: float
    percent: float
    calls: int


@dataclass
class GapAnalysisConfig:
    """Configuration for the GapAnalyzer pipeline."""
    enabled: bool = True
    trace_start_pct: float = 50.0
    trace_end_pct: float = 80.0
    categories: List[str] = field(default_factory=lambda: ["kernel", "gpu"])
    ignore_categories: List[str] = field(
        default_factory=lambda: ["gpu_user_annotation"],
    )
    top_k: int = 20
    min_duration_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceLensConfig:
    """Configuration for TraceLens analysis."""
    enabled: bool = True
    perf_report_enabled: bool = True
    multi_rank_report_enabled: bool = True
    collective_analysis: bool = True
    short_kernel_study: bool = False
    export_csv: bool = True
    export_excel: bool = False
    export_format: str = "csv"
    gpu_arch_config: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KernelStat:
    """Aggregated statistics for a single kernel name."""
    name: str
    total_duration_us: float = 0.0
    calls: int = 0
    durations_us: List[float] = field(default_factory=list)

    @property
    def avg_duration_us(self) -> float:
        return self.total_duration_us / self.calls if self.calls else 0.0

    @property
    def median_duration_us(self) -> float:
        return statistics.median(self.durations_us) if self.durations_us else 0.0

    @property
    def max_duration_us(self) -> float:
        return max(self.durations_us) if self.durations_us else 0.0

    @property
    def min_duration_us(self) -> float:
        return min(self.durations_us) if self.durations_us else 0.0


@dataclass
class RankResult:
    """Gap-analysis result for a single rank."""
    rank: int
    trace_file: str
    total_duration_us: float = 0.0
    kernels: List[KernelStat] = field(default_factory=list)


@dataclass
class GapAnalysisResult:
    """Aggregate result of gap analysis across all ranks."""
    config: Dict[str, Any] = field(default_factory=dict)
    rank_results: List[RankResult] = field(default_factory=list)
    merged_kernels: List[KernelStat] = field(default_factory=list)
    total_duration_us: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def num_ranks(self) -> int:
        return len(self.rank_results)


# ---------------------------------------------------------------------------
# GapAnalyzer
# ---------------------------------------------------------------------------

class GapAnalyzer:
    """
    Analyzes torch profiler Chrome-trace files, producing kernel stats CSVs.

    Pipeline:
      1. Apply time window (trace_start_pct -- trace_end_pct)
      2. Filter by category (case-insensitive substring matching)
      3. Aggregate stats and rank by total duration
    """

    def __init__(self, config: Optional[GapAnalysisConfig] = None):
        self.config = config or GapAnalysisConfig(enabled=True)

    # -- public API ---------------------------------------------------------

    def analyze(self, trace_dir: Path) -> GapAnalysisResult:
        """
        Run gap analysis on all rank traces in *trace_dir*.

        Produces kernel stats (category-filtered, time-windowed).
        Does NOT generate clamped traces -- call
        :meth:`generate_clamped_traces` separately if needed.
        """
        trace_dir = Path(trace_dir)
        result = GapAnalysisResult(config=self.config.to_dict())

        if not trace_dir.exists():
            result.errors.append(f"Trace directory not found: {trace_dir}")
            return result

        rank_files = self.detect_trace_files(trace_dir)
        if not rank_files:
            result.errors.append(f"No trace files found in {trace_dir}")
            return result

        logger.info(f"Found {len(rank_files)} trace file(s) in {trace_dir}")

        for rank, trace_file in rank_files:
            try:
                _data, events = self._load_trace_data(trace_file)
                rr = self._analyze_single_rank(rank, trace_file, events)
                result.rank_results.append(rr)
            except Exception as e:
                msg = f"Failed to analyze rank {rank} ({trace_file.name}): {e}"
                logger.warning(msg)
                result.errors.append(msg)

        if result.rank_results:
            result.merged_kernels = self._merge_ranks(result.rank_results)
            result.total_duration_us = sum(
                rr.total_duration_us for rr in result.rank_results
            )
            result.merged_kernels = result.merged_kernels[: self.config.top_k]
            for rr in result.rank_results:
                rr.kernels = rr.kernels[: self.config.top_k]

        return result

    def generate_clamped_traces(
        self,
        trace_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Generate time-windowed (clamped) trace files for each rank.

        This is a separate step from :meth:`analyze` -- call it only
        when you explicitly need clamped trace output.
        """
        trace_dir = Path(trace_dir)
        dest = Path(output_dir) if output_dir else trace_dir
        dest.mkdir(parents=True, exist_ok=True)

        rank_files = self.detect_trace_files(trace_dir)
        paths: List[Path] = []

        for _rank, trace_file in rank_files:
            try:
                data, events = self._load_trace_data(trace_file)
                p = self._generate_clamped_trace(
                    data, events, trace_file, output_dir=dest,
                )
                if p:
                    paths.append(p)
            except Exception as e:
                logger.warning(
                    f"Failed to generate clamped trace for "
                    f"{trace_file.name}: {e}"
                )

        return paths

    # -- trace file discovery -----------------------------------------------

    @staticmethod
    def detect_trace_files(trace_dir: Path) -> List[Tuple[int, Path]]:
        """
        Discover per-rank trace files in *trace_dir*.

        Rank traces match ``*-rank-N.*.json.gz`` or ``*-rank-N.*.json``.
        Falls back to any ``.json.gz`` / ``.json`` if no rank pattern found.
        Filters out docker logs, async_llm frontend traces, and non-trace JSONs.
        """
        rank_files: List[Tuple[int, Path]] = []

        candidates = (
            sorted(trace_dir.glob("*.json.gz"))
            + sorted(trace_dir.glob("*.json"))
        )
        candidates = [
            f for f in candidates
            if "_docker.log" not in f.name and "async_llm" not in f.name.lower()
        ]

        for gz in sorted(trace_dir.glob("*-rank-*.pt.trace.json.gz")):
            rank = _extract_rank(gz.name)
            if rank is not None:
                rank_files.append((rank, gz))
        if rank_files:
            return sorted(rank_files, key=lambda x: x[0])

        for jf in sorted(trace_dir.glob("*-rank-*.pt.trace.json")):
            rank = _extract_rank(jf.name)
            if rank is not None:
                rank_files.append((rank, jf))
        if rank_files:
            return sorted(rank_files, key=lambda x: x[0])

        for idx, f in enumerate(candidates):
            try:
                opener = gzip.open if f.suffix == ".gz" else open
                with opener(f, "rt") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "traceEvents" in data:
                    rank_match = re.search(r"rank[-_]?(\d+)", f.name)
                    rank = int(rank_match.group(1)) if rank_match else idx
                    rank_files.append((rank, f))
                    print(f"  VALID trace (rank {rank}): {f.name}")
                else:
                    print(f"  Skipped (not a torch trace): {f.name}")
            except Exception as e:
                print(f"  ERROR reading {f.name}: {e}")

        return sorted(rank_files, key=lambda x: x[0])

    # -- trace loading ------------------------------------------------------

    @staticmethod
    def _load_trace_data(
        trace_file: Path,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Load Chrome-trace data and events from .json or .json.gz."""
        print(f"  Loading {trace_file.name} ...", flush=True)

        if trace_file.name.endswith(".json.gz"):
            opener = gzip.open(trace_file, "rt")
        else:
            opener = open(trace_file, "r")

        with opener as f:
            data = json.load(f)

        if isinstance(data, dict):
            events = data.get("traceEvents", [])
        elif isinstance(data, list):
            events = data
        else:
            events = []

        print(f"  Loaded {len(events)} events", flush=True)
        return data, events

    # -- single-rank analysis -----------------------------------------------

    def _analyze_single_rank(
        self,
        rank: int,
        trace_file: Path,
        events: List[Dict[str, Any]],
    ) -> RankResult:
        """
        Build kernel stats for one rank.

        Applies time window first, then category filter.
        """
        logger.debug(
            f"Rank {rank}: loaded {len(events)} events from {trace_file.name}"
        )

        windowed = self._apply_time_window(events)
        logger.debug(
            f"Rank {rank}: {len(windowed)} events in time window "
            f"({self.config.trace_start_pct}%-{self.config.trace_end_pct}%)"
        )

        filtered = self._filter_by_category(windowed)
        logger.debug(
            f"Rank {rank}: {len(filtered)} events after category filter"
        )

        if self.config.min_duration_us > 0:
            filtered = [
                (n, d) for n, d in filtered
                if d >= self.config.min_duration_us
            ]

        kernels = self._aggregate_stats(filtered)
        total_us = sum(k.total_duration_us for k in kernels)

        return RankResult(
            rank=rank,
            trace_file=str(trace_file),
            total_duration_us=total_us,
            kernels=kernels,
        )

    # -- time window --------------------------------------------------------

    def _apply_time_window(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep only events that overlap with the configured time window.

        If the window is 0%-100% (default), returns all events unchanged.
        """
        start_pct = self.config.trace_start_pct
        end_pct = self.config.trace_end_pct

        if start_pct <= 0 and end_pct >= 100:
            return events

        # Compute time range only from duration events (ph='X') to avoid
        # metadata (ph='M') and instant (ph='i') events that can have
        # timestamps far outside the actual profiling window (e.g.
        # "Record Window End"), which would inflate the range and cause
        # the percentage-based window to miss all real events.
        t_min = float("inf")
        t_max = float("-inf")
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur")
            if not dur or dur <= 0:
                continue
            t_min = min(t_min, ts)
            t_max = max(t_max, ts + dur)

        if t_min == float("inf"):
            return events

        span = t_max - t_min
        t_start = t_min + span * (start_pct / 100.0)
        t_end = t_min + span * (end_pct / 100.0)

        result = []
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            if (ts + dur) >= t_start and ts <= t_end:
                result.append(e)

        return result

    # -- category filtering -------------------------------------------------

    def _filter_by_category(
        self, events: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Filter events by category using case-insensitive substring matching.

        Returns list of (name, duration_us) tuples.
        """
        allowed = self.config.categories
        ignored = self.config.ignore_categories

        result: List[Tuple[str, float]] = []
        for ev in events:
            cat = (ev.get("cat") or "").lower()

            if ignored and any(ig.lower() in cat for ig in ignored):
                continue
            if allowed and not any(c.lower() in cat for c in allowed):
                continue

            name = ev.get("name", "<?>")
            dur_us = ev.get("dur", 0.0)
            result.append((name, dur_us))

        return result

    # -- clamped trace generation -------------------------------------------

    def _generate_clamped_trace(
        self,
        data: Any,
        events: List[Dict[str, Any]],
        trace_file: Path,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate a time-windowed (clamped) trace file.

        No category filtering -- all events within the window are included.
        Events overlapping the window boundary are clamped to fit.
        """
        start_pct = self.config.trace_start_pct
        end_pct = self.config.trace_end_pct

        t_min = float("inf")
        t_max = float("-inf")
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur")
            if not dur or dur <= 0:
                continue
            t_min = min(t_min, ts)
            t_max = max(t_max, ts + dur)

        if t_min == float("inf"):
            return None

        span = t_max - t_min
        t_start = t_min + span * (start_pct / 100.0)
        t_end = t_min + span * (end_pct / 100.0)

        filtered = []
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            if (ts + dur) < t_start or ts > t_end:
                continue

            e_new = dict(e)
            clamped_start = max(ts, t_start)
            clamped_end = min(ts + dur, t_end)
            e_new["ts"] = clamped_start
            e_new["dur"] = clamped_end - clamped_start
            filtered.append(e_new)

        stem = trace_file.name
        for suffix in (".json.gz", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        dest = output_dir if output_dir else trace_file.parent
        output_path = dest / f"{stem}_clamped_{start_pct}_{end_pct}.trace.json.gz"

        if isinstance(data, dict) and "traceEvents" in data:
            data_out = dict(data)
            data_out["traceEvents"] = filtered
        else:
            data_out = filtered

        with gzip.open(output_path, "wt") as f:
            json.dump(data_out, f)

        logger.info(
            f"Wrote clamped trace ({len(filtered)}/{len(events)} events): "
            f"{output_path}"
        )
        return output_path

    # -- aggregation --------------------------------------------------------

    @staticmethod
    def _aggregate_stats(
        events: List[Tuple[str, float]],
    ) -> List[KernelStat]:
        """Aggregate events by name, sorted by total duration descending."""
        by_name: Dict[str, KernelStat] = {}
        for name, dur_us in events:
            if name not in by_name:
                by_name[name] = KernelStat(name=name)
            ks = by_name[name]
            ks.total_duration_us += dur_us
            ks.calls += 1
            ks.durations_us.append(dur_us)

        return sorted(by_name.values(), key=lambda k: -k.total_duration_us)

    # -- merge ranks --------------------------------------------------------

    @staticmethod
    def _merge_ranks(rank_results: List[RankResult]) -> List[KernelStat]:
        """Merge kernel stats across ranks."""
        merged: Dict[str, KernelStat] = {}
        for rr in rank_results:
            for ks in rr.kernels:
                if ks.name not in merged:
                    merged[ks.name] = KernelStat(name=ks.name)
                m = merged[ks.name]
                m.total_duration_us += ks.total_duration_us
                m.calls += ks.calls
                m.durations_us.extend(ks.durations_us)

        return sorted(merged.values(), key=lambda k: -k.total_duration_us)


def _extract_rank(filename: str) -> Optional[int]:
    """Extract rank number from a filename like ``...-rank-0.1234...``."""
    m = re.search(r"-rank-(\d+)\.", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# TraceLens installer + analyzer
# ---------------------------------------------------------------------------

def ensure_tracelens_installed() -> bool:
    """
    Check if TraceLens is installed, and install it if not.

    Returns:
        True if TraceLens is available (installed or just installed)
    """
    if shutil.which(CLI_GENERATE_REPORT):
        logger.debug("TraceLens CLI is already available")
        return True

    try:
        import TraceLens
        logger.debug("TraceLens is already installed")
        return True
    except ImportError:
        pass

    # Prefer local clone of the internal repo (matches skill 05-profile-analyze)
    if os.path.isdir(TRACELENS_LOCAL_DIR):
        logger.info(f"Installing TraceLens from local clone: {TRACELENS_LOCAL_DIR}")
        install_src = TRACELENS_LOCAL_DIR
    else:
        # Try cloning the internal repo first
        try:
            logger.info(f"Cloning TraceLens from {TRACELENS_INTERNAL_REPO}...")
            subprocess.run(
                ["git", "clone", TRACELENS_INTERNAL_REPO, TRACELENS_LOCAL_DIR],
                capture_output=True, text=True, timeout=120,
            )
            if os.path.isdir(TRACELENS_LOCAL_DIR):
                install_src = TRACELENS_LOCAL_DIR
            else:
                install_src = TRACELENS_INSTALL_URL
        except Exception:
            logger.info(f"Internal repo unavailable, falling back to {TRACELENS_INSTALL_URL}")
            install_src = TRACELENS_INSTALL_URL

    logger.info(f"Installing TraceLens from {install_src}...")
    try:
        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if install_src == TRACELENS_LOCAL_DIR:
            pip_cmd += ["--no-build-isolation", install_src]
        else:
            pip_cmd.append(install_src)

        result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info("TraceLens installed successfully")
            return True
        else:
            logger.error(f"Failed to install TraceLens: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("TraceLens installation timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to install TraceLens: {e}")
        return False


class TraceLensAnalyzer:
    """
    TraceLens trace analyzer for torch profiler traces.

    Uses TraceLens CLI commands for analysis:
    - TraceLens_generate_perf_report_pytorch: Single trace analysis
    - TraceLens_generate_multi_rank_collective_report_pytorch: Multi-rank collective
    - TraceLens_compare_perf_reports_pytorch: Compare reports
    """

    def __init__(self, config: TraceLensConfig):
        self.config = config
        self._tracelens_available: Optional[bool] = None

    def is_available(self) -> bool:
        """
        Check if TraceLens CLI is installed and available.
        Will attempt to install if not found.

        Returns:
            True if TraceLens is available
        """
        if self._tracelens_available is not None:
            return self._tracelens_available

        if not ensure_tracelens_installed():
            self._tracelens_available = False
            return False

        if shutil.which(CLI_GENERATE_REPORT):
            self._tracelens_available = True
            logger.info("TraceLens CLI is available")
        else:
            logger.error(f"TraceLens installed but CLI command '{CLI_GENERATE_REPORT}' not found")
            self._tracelens_available = False

        return self._tracelens_available

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        trace_dir: Path,
        output_dir: Path,
        num_ranks: int = 8,
    ) -> Dict[str, Any]:
        """
        Analyze torch profiler traces using TraceLens CLI.

        Runs based on configuration:
        - TraceLens_generate_perf_report_pytorch (if perf_report_enabled)
        - TraceLens_generate_multi_rank_collective_report_pytorch
          (if multi_rank_report_enabled)

        Args:
            trace_dir: Directory containing torch trace files (*.json.gz)
            output_dir: Output directory for analysis results
            num_ranks: Number of GPU ranks (for multi-rank collective analysis)

        Returns:
            Dictionary with analysis results and output paths
        """
        if not self.config.enabled:
            logger.debug("TraceLens analysis is disabled")
            return {"enabled": False}

        if not self.is_available():
            logger.warning("TraceLens is not available, skipping analysis")
            return {"enabled": True, "error": "TraceLens not installed"}

        results: Dict[str, Any] = {
            "enabled": True,
            "trace_dir": str(trace_dir),
            "num_ranks": num_ranks,
            "export_format": self.config.export_format,
            "output_files": [],
            "errors": [],
        }

        trace_files = self._find_trace_files(trace_dir)
        if not trace_files:
            results["errors"].append(f"No trace files found in {trace_dir}")
            logger.warning(f"No trace files found in {trace_dir}")
            return results

        logger.info(f"Found {len(trace_files)} trace files in {trace_dir}")

        use_csv = self.config.export_csv
        use_excel = self.config.export_excel

        # 1. Single rank performance report
        if self.config.perf_report_enabled:
            logger.info("Running TraceLens_generate_perf_report_pytorch...")

            rank0_csv_dir = output_dir / "tracelens_rank0_csvs" if use_csv else None
            rank0_xlsx = output_dir / "tracelens_rank0_report.xlsx" if use_excel else None

            if rank0_csv_dir:
                rank0_csv_dir.mkdir(parents=True, exist_ok=True)

            trace_file = trace_files[0]
            rank0_result = self._run_generate_report(
                trace_file=trace_file,
                output_csv_dir=rank0_csv_dir,
                output_xlsx=rank0_xlsx,
            )
            results["output_files"].extend(rank0_result.get("files", []))
            if rank0_result.get("error"):
                results["errors"].append(rank0_result["error"])

        # 2. Multi-rank collective analysis
        if (
            self.config.multi_rank_report_enabled
            and len(trace_files) >= num_ranks
            and num_ranks > 1
        ):
            logger.info(
                "Running TraceLens_generate_multi_rank_collective_report_pytorch..."
            )

            collective_csv_dir = (
                output_dir / "tracelens_collective_csvs" if use_csv else None
            )
            collective_xlsx = (
                output_dir / "tracelens_collective_report.xlsx" if use_excel else None
            )

            if collective_csv_dir:
                collective_csv_dir.mkdir(parents=True, exist_ok=True)

            collective_result = self._run_multi_rank_collective(
                trace_dir=trace_dir,
                output_csv_dir=collective_csv_dir,
                output_xlsx=collective_xlsx,
                num_ranks=num_ranks,
            )
            results["output_files"].extend(collective_result.get("files", []))
            if collective_result.get("error"):
                results["errors"].append(collective_result["error"])
        elif self.config.multi_rank_report_enabled and num_ranks > 1:
            logger.info(
                f"Skipping multi-rank analysis: found {len(trace_files)} traces "
                f"but need at least {num_ranks} for world_size={num_ranks}"
            )

        logger.info(
            f"TraceLens analysis complete. "
            f"Output files: {len(results['output_files'])}"
        )
        return results

    def compare_reports(
        self,
        report_dirs: List[Path],
        output_dir: Path,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple TraceLens performance reports.

        Uses TraceLens_compare_perf_reports_pytorch CLI command.

        Args:
            report_dirs: List of directories containing TraceLens CSV reports
            output_dir: Output directory for comparison results
            labels: Optional labels for each report
        """
        if len(report_dirs) < 2:
            return {"error": "At least 2 report directories required for comparison"}

        if not self.is_available():
            return {"error": "TraceLens not installed"}

        result: Dict[str, Any] = {"files": [], "error": None}

        cmd = [CLI_COMPARE_REPORTS]
        for i, report_dir in enumerate(report_dirs):
            cmd.extend(["--input_csvs_dir", str(report_dir)])
            if labels and i < len(labels):
                cmd.extend(["--label", labels[i]])

        output_csv_dir = output_dir / "tracelens_comparison_csvs"
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_csvs_dir", str(output_csv_dir)])

        if self.config.export_excel:
            cmd.extend([
                "--output_xlsx_path",
                str(output_dir / "tracelens_comparison.xlsx"),
            ])

        logger.info(f"Running TraceLens compare: {' '.join(cmd)}")

        try:
            proc_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )

            if proc_result.returncode == 0:
                for csv_file in output_csv_dir.glob("*.csv"):
                    result["files"].append(str(csv_file))
                if self.config.export_excel:
                    xlsx_file = output_dir / "tracelens_comparison.xlsx"
                    if xlsx_file.exists():
                        result["files"].append(str(xlsx_file))
                logger.info(
                    f"Comparison complete: {len(result['files'])} files generated"
                )
            else:
                result["error"] = f"Compare failed: {proc_result.stderr}"
                logger.error(f"TraceLens compare failed: {proc_result.stderr}")

        except subprocess.TimeoutExpired:
            result["error"] = "Comparison timed out"
            logger.error("TraceLens comparison timed out")
        except Exception as e:
            result["error"] = f"Comparison error: {str(e)}"
            logger.exception(f"TraceLens comparison error: {e}")

        return result

    # -- trace file discovery -----------------------------------------------

    def _find_trace_files(self, trace_dir: Path) -> List[Path]:
        """Find all trace files in directory, filtering out async_llm traces."""
        trace_files: List[Path] = []

        for pattern in ["*.json.gz", "*.json"]:
            trace_files.extend(trace_dir.glob(pattern))

        worker_traces = [
            f for f in trace_files
            if "async_llm" not in f.name.lower()
        ]

        if worker_traces:
            trace_files = worker_traces

        return sorted(trace_files)

    # -- single-rank perf report --------------------------------------------

    def _run_generate_report(
        self,
        trace_file: Path,
        output_csv_dir: Optional[Path] = None,
        output_xlsx: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run TraceLens_generate_perf_report_pytorch CLI command."""
        result: Dict[str, Any] = {"files": [], "error": None}

        if not output_csv_dir and not output_xlsx:
            result["error"] = (
                "At least one of output_csv_dir or output_xlsx must be specified"
            )
            return result

        cmd = [
            CLI_GENERATE_REPORT,
            "--profile_json_path", str(trace_file),
        ]

        if output_csv_dir:
            cmd.extend(["--output_csvs_dir", str(output_csv_dir)])
        if output_xlsx:
            cmd.extend(["--output_xlsx_path", str(output_xlsx)])

        if not self.config.collective_analysis:
            cmd.append("--disable_coll_analysis")
        if self.config.short_kernel_study:
            cmd.append("--short_kernel_study")

        cmd.append("--enable_kernel_summary")

        if self.config.gpu_arch_config:
            cmd.extend(["--gpu_arch_json_path", self.config.gpu_arch_config])

        logger.info(f"Running TraceLens: {' '.join(cmd)}")

        try:
            proc_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )

            if proc_result.returncode == 0:
                if output_csv_dir:
                    for csv_file in output_csv_dir.glob("*.csv"):
                        result["files"].append(str(csv_file))
                if output_xlsx and output_xlsx.exists():
                    result["files"].append(str(output_xlsx))
                logger.info(
                    f"TraceLens perf report complete: "
                    f"{len(result['files'])} files generated"
                )
                if proc_result.stdout:
                    logger.debug(f"TraceLens output: {proc_result.stdout}")
            else:
                result["error"] = f"TraceLens CLI failed: {proc_result.stderr}"
                logger.error(f"TraceLens CLI failed: {proc_result.stderr}")
                if proc_result.stdout:
                    logger.error(f"TraceLens stdout: {proc_result.stdout}")

        except subprocess.TimeoutExpired:
            result["error"] = "TraceLens analysis timed out"
            logger.error("TraceLens analysis timed out")
        except FileNotFoundError:
            result["error"] = (
                f"TraceLens CLI command '{CLI_GENERATE_REPORT}' not found"
            )
            logger.error(
                f"TraceLens CLI command '{CLI_GENERATE_REPORT}' not found"
            )
        except Exception as e:
            result["error"] = f"TraceLens error: {str(e)}"
            logger.exception(f"TraceLens analysis error: {e}")

        return result

    # -- multi-rank collective report ---------------------------------------

    def _run_multi_rank_collective(
        self,
        trace_dir: Path,
        output_csv_dir: Optional[Path] = None,
        output_xlsx: Optional[Path] = None,
        num_ranks: int = 8,
    ) -> Dict[str, Any]:
        """Run TraceLens_generate_multi_rank_collective_report_pytorch CLI."""
        result: Dict[str, Any] = {"files": [], "error": None}

        if not output_csv_dir and not output_xlsx:
            result["error"] = (
                "At least one of output_csv_dir or output_xlsx must be specified"
            )
            return result

        trace_files = self._find_trace_files(trace_dir)
        if len(trace_files) < num_ranks:
            logger.warning(
                f"Found {len(trace_files)} trace files but "
                f"num_ranks={num_ranks}. Adjusting num_ranks to "
                f"{len(trace_files)}"
            )
            num_ranks = len(trace_files)

        if num_ranks < 2:
            result["error"] = "Multi-rank analysis requires at least 2 ranks"
            return result

        trace_pattern = self._detect_trace_pattern(trace_dir, trace_files)

        cmd = [CLI_MULTI_RANK_COLLECTIVE]
        if trace_pattern:
            cmd.extend(["--trace_pattern", trace_pattern])
        else:
            cmd.extend(["--trace_dir", str(trace_dir)])

        cmd.extend(["--world_size", str(num_ranks)])

        if output_csv_dir:
            cmd.extend(["--output_csvs_dir", str(output_csv_dir)])
        if output_xlsx:
            cmd.extend(["--output_xlsx_path", str(output_xlsx)])

        logger.info(f"Running TraceLens multi-rank: {' '.join(cmd)}")

        try:
            proc_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900,
            )

            if proc_result.returncode == 0:
                if output_csv_dir:
                    for csv_file in output_csv_dir.glob("*.csv"):
                        result["files"].append(str(csv_file))
                if output_xlsx and output_xlsx.exists():
                    result["files"].append(str(output_xlsx))
                logger.info(
                    f"Multi-rank collective report complete: "
                    f"{len(result['files'])} files generated"
                )
                if proc_result.stdout:
                    logger.debug(f"TraceLens output: {proc_result.stdout}")
            else:
                result["error"] = (
                    f"Multi-rank analysis failed: {proc_result.stderr}"
                )
                logger.error(
                    f"TraceLens multi-rank failed: {proc_result.stderr}"
                )
                if proc_result.stdout:
                    logger.error(f"TraceLens stdout: {proc_result.stdout}")

        except subprocess.TimeoutExpired:
            result["error"] = "Multi-rank analysis timed out"
            logger.error("TraceLens multi-rank analysis timed out")
        except FileNotFoundError:
            result["error"] = (
                f"TraceLens CLI command '{CLI_MULTI_RANK_COLLECTIVE}' not found"
            )
            logger.error(
                f"TraceLens CLI command "
                f"'{CLI_MULTI_RANK_COLLECTIVE}' not found"
            )
        except Exception as e:
            result["error"] = f"Multi-rank analysis error: {str(e)}"
            logger.exception(f"TraceLens multi-rank error: {e}")

        return result

    # -- trace pattern detection --------------------------------------------

    def _detect_trace_pattern(
        self,
        trace_dir: Path,
        trace_files: List[Path],
    ) -> Optional[str]:
        """
        Detect trace file pattern for multi-rank analysis.

        TraceLens expects a glob pattern like:
        - "/path/to/traces/rank*_trace.json.gz"
        - "/path/to/traces/worker*_trace.json"
        """
        if not trace_files:
            return None

        first_name = trace_files[0].name

        # rank-{N} or rank{N}
        rank_match = re.search(r'rank[-_]?(\d+)', first_name, re.IGNORECASE)
        if rank_match:
            pattern_name = re.sub(
                r'rank([-_]?)\d+', r'rank\1*', first_name, flags=re.IGNORECASE,
            )
            return str(trace_dir / pattern_name)

        # worker-{N} or worker{N}
        worker_match = re.search(
            r'worker[-_]?(\d+)', first_name, re.IGNORECASE,
        )
        if worker_match:
            pattern_name = re.sub(
                r'worker([-_]?)\d+', r'worker\1*', first_name,
                flags=re.IGNORECASE,
            )
            return str(trace_dir / pattern_name)

        # gpu-{N} or gpu{N}
        gpu_match = re.search(r'gpu[-_]?(\d+)', first_name, re.IGNORECASE)
        if gpu_match:
            pattern_name = re.sub(
                r'gpu([-_]?)\d+', r'gpu\1*', first_name, flags=re.IGNORECASE,
            )
            return str(trace_dir / pattern_name)

        # Sequential numbering in filename (e.g., trace_0.json)
        num_match = re.search(r'[._-](\d+)[._-]', first_name)
        if num_match:
            num_str = num_match.group(1)
            pattern_name = first_name.replace(num_str, '*', 1)
            return str(trace_dir / pattern_name)

        logger.warning(f"Could not detect trace pattern from: {first_name}")
        return None


# ---------------------------------------------------------------------------
# CSV / JSON output helpers
# ---------------------------------------------------------------------------

def _write_kernel_csv(
    path: Path,
    kernels: List[KernelStat],
    total_us: float,
) -> None:
    """Write kernel stats to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Name", "Calls",
            "Self CUDA total (us)", "Avg time (us)",
            "Median (us)", "Min (us)", "Max (us)",
            "% Total",
        ])
        for ks in kernels:
            pct = (ks.total_duration_us / total_us * 100.0) if total_us > 0 else 0.0
            w.writerow([
                ks.name,
                ks.calls,
                f"{ks.total_duration_us:.2f}",
                f"{ks.avg_duration_us:.2f}",
                f"{ks.median_duration_us:.2f}",
                f"{ks.min_duration_us:.2f}",
                f"{ks.max_duration_us:.2f}",
                f"{pct:.2f}",
            ])


def _kernel_stat_to_dict(ks: KernelStat, total_us: float) -> Dict[str, Any]:
    """Serialize a KernelStat to a JSON-friendly dict."""
    pct = (ks.total_duration_us / total_us * 100.0) if total_us > 0 else 0.0
    return {
        "name": ks.name,
        "calls": ks.calls,
        "self_cuda_total_us": round(ks.total_duration_us, 2),
        "avg_time_us": round(ks.avg_duration_us, 2),
        "median_time_us": round(ks.median_duration_us, 2),
        "min_time_us": round(ks.min_duration_us, 2),
        "max_time_us": round(ks.max_duration_us, 2),
        "pct_total": round(pct, 2),
    }


# ---------------------------------------------------------------------------
# Mode 1: Full trace parse (simple kernel summary)
# ---------------------------------------------------------------------------

def parse_torch_trace(trace_dir: Path) -> List[KernelMetrics]:
    """
    Parse the first valid PyTorch profiler trace for kernel metrics.

    Returns list of KernelMetrics sorted by total time descending.
    """
    kernels: List[KernelMetrics] = []

    if not trace_dir.exists():
        logger.warning(f"Trace directory does not exist: {trace_dir}")
        return kernels

    rank_files = GapAnalyzer.detect_trace_files(trace_dir)
    if not rank_files:
        logger.warning(f"No valid trace files found in {trace_dir}")
        return kernels

    _, trace_file = rank_files[0]
    try:
        _data, events = GapAnalyzer._load_trace_data(trace_file)

        kernel_times: Dict[str, float] = {}
        kernel_counts: Dict[str, int] = {}

        for event in events:
            if event.get("cat") == "kernel":
                name = event.get("name", "unknown")
                dur = event.get("dur", 0) / 1000.0
                kernel_times[name] = kernel_times.get(name, 0.0) + dur
                kernel_counts[name] = kernel_counts.get(name, 0) + 1

        total_time = sum(kernel_times.values())
        print(f"  {len(kernel_times)} unique kernels, total kernel time: {total_time:.2f} ms")

        for name, time_ms in sorted(kernel_times.items(), key=lambda x: -x[1]):
            percent = (time_ms / total_time * 100) if total_time > 0 else 0
            kernels.append(KernelMetrics(
                name=name,
                time_ms=time_ms,
                percent=percent,
                calls=kernel_counts.get(name, 0),
            ))

    except Exception as e:
        logger.warning(f"Failed to parse trace file {trace_file}: {e}")

    return kernels


# ---------------------------------------------------------------------------
# Mode 2: Gap analysis (via GapAnalyzer) with CSV/JSON output
# ---------------------------------------------------------------------------

def gap_analysis(
    trace_dir: Path,
    output_dir: Path,
    start_pct: float = 50.0,
    end_pct: float = 80.0,
    top_k: int = 20,
    min_dur: float = 0.0,
    clamped_traces: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run GapAnalyzer and write per-rank CSVs, merged CSV, and summary JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    config = GapAnalysisConfig(
        enabled=True,
        trace_start_pct=start_pct,
        trace_end_pct=end_pct,
        top_k=top_k,
        min_duration_us=min_dur,
    )
    analyzer = GapAnalyzer(config)

    print(
        f"\nGap analysis: window {start_pct}%-{end_pct}%, "
        f"top-k={top_k}, min_dur={min_dur} us",
        flush=True,
    )

    result = analyzer.analyze(trace_dir)

    if result.errors:
        for err in result.errors:
            print(f"  WARNING: {err}", flush=True)

    if not result.rank_results:
        print("No valid trace files for gap analysis")
        return None

    print(f"  Analyzed {result.num_ranks} rank(s)", flush=True)

    for rr in result.rank_results:
        rank_csv = output_dir / f"gap_analysis_rank{rr.rank}.csv"
        _write_kernel_csv(rank_csv, rr.kernels, rr.total_duration_us)
        print(
            f"  Rank {rr.rank}: {len(rr.kernels)} kernels, "
            f"total {rr.total_duration_us:.0f} us -> {rank_csv}",
            flush=True,
        )

    merged_csv = output_dir / "gap_analysis.csv"
    _write_kernel_csv(merged_csv, result.merged_kernels, result.total_duration_us)

    print(
        f"\nMerged gap analysis: {len(result.merged_kernels)} top kernels -> {merged_csv}",
        flush=True,
    )

    # Console summary
    n_show = min(10, len(result.merged_kernels))
    print(f"\nTop {n_show} GPU kernels by cumulative time:")
    print(
        f"{'Name':<60} {'Calls':>8} {'Total(us)':>14} "
        f"{'Avg(us)':>12} {'Med(us)':>12} {'% Total':>8}"
    )
    print("-" * 118)
    for ks in result.merged_kernels[:n_show]:
        pct = (ks.total_duration_us / result.total_duration_us * 100.0
               if result.total_duration_us > 0 else 0.0)
        short_name = ks.name[:58] + ".." if len(ks.name) > 60 else ks.name
        print(
            f"{short_name:<60} {ks.calls:>8} {ks.total_duration_us:>14.2f} "
            f"{ks.avg_duration_us:>12.2f} {ks.median_duration_us:>12.2f} "
            f"{pct:>7.1f}%"
        )

    gap_summary: Dict[str, Any] = {
        "config": result.config,
        "num_ranks": result.num_ranks,
        "total_duration_us": result.total_duration_us,
        "top_kernels": [
            _kernel_stat_to_dict(ks, result.total_duration_us)
            for ks in result.merged_kernels
        ],
        "per_rank": [
            {
                "rank": rr.rank,
                "trace_file": rr.trace_file,
                "total_duration_us": rr.total_duration_us,
                "num_kernels": len(rr.kernels),
            }
            for rr in result.rank_results
        ],
        "csv_path": str(merged_csv),
        "output_dir": str(output_dir),
    }
    gap_json_path = output_dir / "gap_analysis.json"
    with open(gap_json_path, "w") as f:
        json.dump(gap_summary, f, indent=2)
    print(f"\nGap analysis summary saved: {gap_json_path}")

    if clamped_traces:
        clamped_dir = output_dir / "clamped_traces"
        clamped_paths = analyzer.generate_clamped_traces(trace_dir, clamped_dir)
        if clamped_paths:
            print(f"\nGenerated {len(clamped_paths)} clamped trace file(s) in {clamped_dir}")
            gap_summary["clamped_traces"] = [str(p) for p in clamped_paths]

    return gap_summary


# ---------------------------------------------------------------------------
# Mode 3: TraceLens analysis (via TraceLensAnalyzer)
# ---------------------------------------------------------------------------

def run_tracelens_analysis(
    config: TraceLensConfig,
    trace_dir: Path,
    output_dir: Path,
    num_ranks: int = 8,
) -> Dict[str, Any]:
    """
    Convenience function to run TraceLens analysis.

    Args:
        config: TraceLens configuration
        trace_dir: Directory containing torch trace files
        output_dir: Output directory for analysis results
        num_ranks: Number of GPU ranks
    """
    analyzer = TraceLensAnalyzer(config)
    return analyzer.analyze(trace_dir, output_dir, num_ranks)


def compare_tracelens_reports(
    config: TraceLensConfig,
    report_dirs: List[Path],
    output_dir: Path,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to compare TraceLens reports.

    Args:
        config: TraceLens configuration
        report_dirs: List of directories containing TraceLens CSV reports
        output_dir: Output directory for comparison results
        labels: Optional labels for each report
    """
    analyzer = TraceLensAnalyzer(config)
    return analyzer.compare_reports(report_dirs, output_dir, labels)


# ---------------------------------------------------------------------------
# Terminal display helpers
# ---------------------------------------------------------------------------

def _fmt_num(val: str) -> str:
    """Format a numeric string for display: round floats, add commas to ints."""
    try:
        f = float(val)
        if f == int(f) and "." not in val:
            return f"{int(f):,}"
        if abs(f) >= 100:
            return f"{f:,.1f}"
        if abs(f) >= 1:
            return f"{f:,.2f}"
        return f"{f:.4f}"
    except (ValueError, OverflowError):
        return val


def _trunc(s: str, width: int) -> str:
    """Truncate *s* to *width* chars, adding '..' if truncated."""
    return s[:width - 2] + ".." if len(s) > width else s


def _print_table(title: str, headers: List[str], rows: List[List[str]],
                 col_widths: Optional[List[int]] = None,
                 alignments: Optional[List[str]] = None) -> None:
    """Print a formatted table to stdout."""
    if not rows:
        return

    if col_widths is None:
        col_widths = [
            max(len(h), *(len(str(r[i])) for r in rows))
            for i, h in enumerate(headers)
        ]
    if alignments is None:
        alignments = ["<"] * len(headers)

    sep = "  "
    hdr_parts = []
    for h, w, a in zip(headers, col_widths, alignments):
        hdr_parts.append(f"{h:{a}{w}}")
    hdr_line = sep.join(hdr_parts)
    rule = "-" * len(hdr_line)

    print(f"\n{'=' * len(hdr_line)}")
    print(f"  {title}")
    print(f"{'=' * len(hdr_line)}")
    print(hdr_line)
    print(rule)
    for row in rows:
        parts = []
        for val, w, a in zip(row, col_widths, alignments):
            parts.append(f"{val:{a}{w}}")
        print(sep.join(parts))
    print()


def print_tracelens_tables(output_dir: Path, top_k: int = 20) -> None:
    """
    Pretty-print the key TraceLens CSVs as terminal tables.

    Displays: category breakdown, ops summary (top-K), kernel summary (top-K).
    """
    csv_dirs = sorted(output_dir.glob("tracelens_rank*_csvs"))
    if not csv_dirs:
        print(f"  No tracelens_rank*_csvs directories found in {output_dir}")
        return

    for csv_dir in csv_dirs:
        rank_label = csv_dir.name.replace("tracelens_", "").replace("_csvs", "")
        print(f"\n{'#' * 80}")
        print(f"  TraceLens Results — {rank_label}")
        print(f"{'#' * 80}")

        # --- Category breakdown (always show all rows) ---
        cat_csv = csv_dir / "ops_summary_by_category.csv"
        if cat_csv.exists():
            with open(cat_csv) as f:
                reader = csv.DictReader(f)
                cat_rows = list(reader)
            if cat_rows:
                headers = ["Category", "Count", "Kernel Time (ms)", "Pct %", "Cumul %"]
                table_rows = []
                for r in cat_rows:
                    table_rows.append([
                        r.get("op category", ""),
                        _fmt_num(r.get("Count", "")),
                        _fmt_num(r.get("total_direct_kernel_time_ms", "")),
                        _fmt_num(r.get("Percentage (%)", "")),
                        _fmt_num(r.get("Cumulative Percentage (%)", "")),
                    ])
                _print_table(
                    "Kernel Time by Category",
                    headers, table_rows,
                    col_widths=[16, 10, 18, 10, 10],
                    alignments=["<", ">", ">", ">", ">"],
                )

        # --- Ops summary (top-K) ---
        ops_csv = csv_dir / "ops_summary.csv"
        if ops_csv.exists():
            with open(ops_csv) as f:
                reader = csv.DictReader(f)
                ops_rows = list(reader)
            if ops_rows:
                shown = ops_rows[:top_k]
                total = len(ops_rows)
                headers = ["#", "Op Name", "Count", "Kernel Time (ms)", "Pct %", "Cumul %"]
                table_rows = []
                for i, r in enumerate(shown, 1):
                    table_rows.append([
                        str(i),
                        _trunc(r.get("name", ""), 50),
                        _fmt_num(r.get("Count", "")),
                        _fmt_num(r.get("total_direct_kernel_time_ms", "")),
                        _fmt_num(r.get("Percentage (%)", "")),
                        _fmt_num(r.get("Cumulative Percentage (%)", "")),
                    ])
                title = f"Top {len(shown)} Ops by Kernel Time (of {total} total)"
                _print_table(
                    title, headers, table_rows,
                    col_widths=[4, 50, 10, 18, 10, 10],
                    alignments=["<", "<", ">", ">", ">", ">"],
                )

        # --- Kernel summary (top-K) ---
        kern_csv = csv_dir / "kernel_summary.csv"
        if kern_csv.exists():
            with open(kern_csv) as f:
                reader = csv.DictReader(f)
                kern_rows = list(reader)
            if kern_rows:
                shown = kern_rows[:top_k]
                total = len(kern_rows)
                headers = ["#", "Category", "Parent Op", "Kernel Name",
                           "Count", "Total (ms)", "Mean (µs)", "Pct %"]
                table_rows = []
                for i, r in enumerate(shown, 1):
                    dur_sum = r.get("Kernel duration (µs)_sum", "0")
                    try:
                        total_ms = float(dur_sum) / 1000.0
                    except ValueError:
                        total_ms = 0.0
                    table_rows.append([
                        str(i),
                        _trunc(r.get("Parent op category", ""), 10),
                        _trunc(r.get("Parent cpu_op", ""), 30),
                        _trunc(r.get("Kernel name", ""), 50),
                        _fmt_num(r.get("Kernel duration (µs)_count", "")),
                        _fmt_num(f"{total_ms:.2f}"),
                        _fmt_num(r.get("Kernel duration (µs)_mean", "")),
                        _fmt_num(r.get("Percent of total time (%)", "")),
                    ])
                title = f"Top {len(shown)} Kernels by Duration (of {total} total)"
                _print_table(
                    title, headers, table_rows,
                    col_widths=[4, 10, 30, 50, 10, 12, 12, 10],
                    alignments=["<", "<", "<", "<", ">", ">", ">", ">"],
                )

        # --- GEMM summary (if present, show all) ---
        gemm_csv = csv_dir / "GEMM.csv"
        if gemm_csv.exists():
            with open(gemm_csv) as f:
                reader = csv.DictReader(f)
                gemm_rows = list(reader)
            if gemm_rows:
                headers = ["#", "Name", "M", "N", "K", "dtype",
                           "Count", "Total (ms)", "Mean (µs)", "TFLOPS/s"]
                table_rows = []
                for i, r in enumerate(gemm_rows, 1):
                    dur_sum = r.get("Kernel Time (µs)_sum", "0")
                    try:
                        total_ms = float(dur_sum) / 1000.0
                    except ValueError:
                        total_ms = 0.0
                    dtype_raw = r.get("param: dtype_A_B", "")
                    dtype_short = (dtype_raw
                                   .replace("c10::", "")
                                   .replace("'", "")
                                   .strip("() "))
                    table_rows.append([
                        str(i),
                        _trunc(r.get("name", ""), 40),
                        _fmt_num(r.get("param: M", "")),
                        _fmt_num(r.get("param: N", "")),
                        _fmt_num(r.get("param: K", "")),
                        _trunc(dtype_short, 12),
                        _fmt_num(r.get("name_count", "")),
                        _fmt_num(f"{total_ms:.2f}"),
                        _fmt_num(r.get("Kernel Time (µs)_mean", "")),
                        _fmt_num(r.get("TFLOPS/s_mean", "")),
                    ])
                _print_table(
                    "GEMM Operations",
                    headers, table_rows,
                    col_widths=[4, 40, 8, 8, 8, 12, 8, 12, 12, 10],
                    alignments=["<", "<", ">", ">", ">", "<", ">", ">", ">", ">"],
                )


def tracelens_analysis(
    trace_dir: Path,
    output_dir: Path,
    num_ranks: int = 8,
    export_csv: bool = True,
    export_excel: bool = False,
    perf_report: bool = True,
    multi_rank_report: bool = True,
    gpu_arch_config: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run TraceLens analysis and print a summary.

    Intended as the CLI entry-point for --tracelens mode.
    """
    os.makedirs(output_dir, exist_ok=True)

    config = TraceLensConfig(
        enabled=True,
        perf_report_enabled=perf_report,
        multi_rank_report_enabled=multi_rank_report,
        export_csv=export_csv,
        export_excel=export_excel,
        gpu_arch_config=gpu_arch_config,
    )

    print(
        f"\nTraceLens analysis: num_ranks={num_ranks}, "
        f"csv={export_csv}, excel={export_excel}",
        flush=True,
    )

    results = run_tracelens_analysis(config, trace_dir, output_dir, num_ranks)

    if results.get("errors"):
        for err in results["errors"]:
            print(f"  WARNING: {err}", flush=True)

    output_files = results.get("output_files", [])
    if output_files:
        print(f"\n  Generated {len(output_files)} output file(s):")
        for f in output_files:
            print(f"    {f}")
    elif results.get("error"):
        print(f"\n  TraceLens error: {results['error']}")
        return None

    results_json = output_dir / "tracelens_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraceLens results saved: {results_json}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch profiler traces for GPU kernel metrics.",
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing torch trace files",
    )
    parser.add_argument(
        "--gap-analysis",
        action="store_true",
        help="Run time-windowed gap analysis instead of full trace parse",
    )
    parser.add_argument(
        "--start-pct",
        type=float,
        default=50.0,
        help="Start of time window as %% of trace duration (default: 50)",
    )
    parser.add_argument(
        "--end-pct",
        type=float,
        default=80.0,
        help="End of time window as %% of trace duration (default: 80)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top kernels to report (default: 20)",
    )
    parser.add_argument(
        "--min-dur",
        type=float,
        default=0.0,
        help="Minimum kernel duration in us to include (default: 0)",
    )
    parser.add_argument(
        "--clamped-traces",
        action="store_true",
        help="Also generate clamped (time-windowed) trace files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for gap/TraceLens analysis "
             "(default: <trace_dir>/../results/<mode>)",
    )

    # TraceLens arguments
    parser.add_argument(
        "--tracelens",
        action="store_true",
        help="Run TraceLens analysis (auto-installs from GitHub if missing)",
    )
    parser.add_argument(
        "--tracelens-compare",
        nargs="+",
        type=Path,
        metavar="DIR",
        default=None,
        help="Compare TraceLens reports from two or more directories",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=8,
        help="Number of GPU ranks for multi-rank analysis (default: 8)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export TraceLens results as CSV (default: True)",
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        default=False,
        help="Export TraceLens results as Excel (.xlsx)",
    )
    parser.add_argument(
        "--no-perf-report",
        action="store_true",
        help="Skip single-rank perf report in TraceLens",
    )
    parser.add_argument(
        "--no-multi-rank",
        action="store_true",
        help="Skip multi-rank collective report in TraceLens",
    )
    parser.add_argument(
        "--gpu-arch-config",
        type=str,
        default=None,
        help="Path to GPU architecture config JSON for TraceLens",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for TraceLens comparison reports",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pretty-print key TraceLens tables to the terminal after analysis "
             "(or from a previous run if used with --tracelens or alone)",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=20,
        help="Number of top rows to display per table (default: 20)",
    )

    args = parser.parse_args()

    trace_dir: Path = args.trace_dir
    print(f"Trace directory: {trace_dir}\n")

    # Mode: TraceLens comparison
    if args.tracelens_compare:
        out = args.output_dir or (
            trace_dir.parent / "results" / "tracelens_comparison"
        )
        config = TraceLensConfig(
            export_csv=args.export_csv,
            export_excel=args.export_excel,
        )
        result = compare_tracelens_reports(
            config, args.tracelens_compare, out, labels=args.labels,
        )
        if result.get("error"):
            print(f"Comparison error: {result['error']}")
        elif result.get("files"):
            print(f"Comparison complete: {len(result['files'])} files")
            for f in result["files"]:
                print(f"  {f}")
        return

    # Mode: TraceLens analysis
    if args.tracelens:
        out = args.output_dir or (
            trace_dir.parent / "results" / "tracelens"
        )
        tracelens_analysis(
            trace_dir, out,
            num_ranks=args.num_ranks,
            export_csv=args.export_csv,
            export_excel=args.export_excel,
            perf_report=not args.no_perf_report,
            multi_rank_report=not args.no_multi_rank,
            gpu_arch_config=args.gpu_arch_config,
        )
        if args.show:
            print_tracelens_tables(out, top_k=args.show_top)
        return

    # Mode: Show previously generated TraceLens results
    if args.show:
        out = args.output_dir or (
            trace_dir.parent / "results" / "tracelens"
        )
        print_tracelens_tables(out, top_k=args.show_top)
        return

    # Mode: Gap analysis
    if args.gap_analysis:
        out = args.output_dir or (trace_dir.parent / "results" / "gap_analysis")
        gap_analysis(
            trace_dir, out,
            args.start_pct, args.end_pct, args.top_k,
            min_dur=args.min_dur,
            clamped_traces=args.clamped_traces,
        )
        return

    # Mode: Full trace parse (default)
    kernels = parse_torch_trace(trace_dir)

    if not kernels:
        print("No kernel metrics found.")
        return

    print(f"\n{'=' * 120}")
    print(f"Top GPU Kernels (total: {len(kernels)} unique kernels)")
    print(f"{'=' * 120}")
    print(f"{'Rank':<6} {'Kernel Name':<70} {'Calls':>8} {'Total (ms)':>12} {'Avg (ms)':>10} {'%':>7}")
    print(f"{'-' * 120}")

    for i, k in enumerate(kernels[:30], 1):
        short_name = k.name[:68] + ".." if len(k.name) > 70 else k.name
        avg_ms = k.time_ms / k.calls if k.calls > 0 else 0
        print(
            f"{i:<6} {short_name:<70} {k.calls:>8} "
            f"{k.time_ms:>12.3f} {avg_ms:>10.4f} {k.percent:>6.2f}%"
        )

    output_path = trace_dir.parent / "kernel_metrics.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "total_kernels": len(kernels),
                "total_time_ms": sum(k.time_ms for k in kernels),
                "top_30": [asdict(k) for k in kernels[:30]],
                "all_kernels": [asdict(k) for k in kernels],
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
