#!/usr/bin/env python3
"""
Structural validator for the vllm-optimize-v2 skill.

Checks:
  1. All required files exist and are non-empty
  2. SKILL.md has correct metadata (name, description, pipeline phases)
  3. RUNTIME.md declares all required variables
  4. Phase docs have correct structure (objectives, completion blocks)
  5. Python scripts have valid syntax
  6. Knowledge base mentions key AMD constraints
  7. Spec constraint coverage: all 7 hard constraints mentioned in correct files
  8. Scripts API contract: kernel_agent.py defines required subcommands
  9. gemm_patch.py implements EngineCore kill note + meta_path hook
 10. Final report phase includes the mandatory verdict table

Usage:
    python tests/validate.py
    python tests/validate.py --skill-dir /path/to/vllm-optimize-v2
    python tests/validate.py --verbose
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


# ─── Result Types ──────────────────────────────────────────────────────────

class Check:
    def __init__(self, category: str, name: str, passed: bool, detail: str = ""):
        self.category = category
        self.name     = name
        self.passed   = passed
        self.detail   = detail

    def __repr__(self):
        icon = "PASS" if self.passed else "FAIL"
        base = f"[{icon}] {self.category}: {self.name}"
        return f"{base}  — {self.detail}" if self.detail else base


# ─── File Existence Checks ─────────────────────────────────────────────────

REQUIRED_FILES = [
    "SKILL.md",
    "INTAKE.md",
    "RUNTIME.md",
    "phases/00-env-setup.md",
    "phases/01-server-setup.md",
    "phases/02-bench-profile.md",
    "phases/03-analysis.md",
    "phases/04-kernel-optimize.md",
    "phases/05-integration.md",
    "phases/06-report.md",
    "scripts/select_gpus.py",
    "scripts/extract_shapes.py",
    "scripts/kernel_breakdown.py",
    "scripts/kernel_agent.py",
    "scripts/gemm_patch.py",
    "references/TRITON_KNOWLEDGE.md",
]

def check_files(skill_dir: Path) -> List[Check]:
    checks = []
    for rel in REQUIRED_FILES:
        p = skill_dir / rel
        exists = p.exists() and p.stat().st_size > 100
        checks.append(Check("files", rel, exists,
                             f"{p.stat().st_size} bytes" if exists else "MISSING or empty"))
    return checks


# ─── SKILL.md Checks ───────────────────────────────────────────────────────

def check_skill_md(skill_dir: Path) -> List[Check]:
    checks = []
    content = (skill_dir / "SKILL.md").read_text()

    checks.append(Check("SKILL.md", "has name: vllm-optimize",
                         "name: vllm-optimize" in content))
    checks.append(Check("SKILL.md", "has description field",
                         "description:" in content))
    checks.append(Check("SKILL.md", "lists pipeline phases",
                         "Phase 0" in content and "Phase" in content))
    checks.append(Check("SKILL.md", "mentions AMD as primary target",
                         re.search(r'AMD|RDNA|CDNA|gfx', content) is not None))
    checks.append(Check("SKILL.md", "lists all 7 hard constraints",
                         content.count("Constraint") >= 7 or
                         (content.count("Hard Constraint") >= 1 and
                          re.search(r'1\.|2\.|3\.|4\.|5\.|6\.|7\.', content) is not None)))
    checks.append(Check("SKILL.md", "mentions modes (optimize/profile-only/optimize-only)",
                         "profile-only" in content and "optimize-only" in content))
    checks.append(Check("SKILL.md", "references INTAKE.md",
                         "INTAKE.md" in content))
    checks.append(Check("SKILL.md", "references RUNTIME.md",
                         "RUNTIME.md" in content))
    checks.append(Check("SKILL.md", "first-turn rule present",
                         re.search(r'first.turn|first turn', content, re.IGNORECASE) is not None))
    return checks


# ─── INTAKE.md Checks ──────────────────────────────────────────────────────

def check_intake_md(skill_dir: Path) -> List[Check]:
    checks = []
    content = (skill_dir / "INTAKE.md").read_text()

    checks.append(Check("INTAKE.md", "defines Round 1 questions",
                         "Round 1" in content))
    checks.append(Check("INTAKE.md", "has Run plan question",
                         re.search(r'[Rr]un plan|run_plan', content) is not None))
    checks.append(Check("INTAKE.md", "has Output question",
                         "Output" in content))
    checks.append(Check("INTAKE.md", "has GPUs question",
                         "GPU" in content))
    checks.append(Check("INTAKE.md", "mentions optimize/profile-only modes",
                         "profile-only" in content))
    checks.append(Check("INTAKE.md", "has confirmation step",
                         "Confirm" in content or "confirmation" in content.lower()))
    checks.append(Check("INTAKE.md", "maps modes to MODE variable",
                         "MODE=optimize" in content or "MODE = optimize" in content))
    checks.append(Check("INTAKE.md", "has status messages",
                         "Status" in content and ("1/5" in content or "2/5" in content)))
    return checks


# ─── RUNTIME.md Checks ─────────────────────────────────────────────────────

REQUIRED_VARS = [
    "MODEL", "OUTPUT_DIR", "PROFILE_DIR", "RESULTS_DIR", "REPORT_DIR",
    "SCRIPTS_DIR", "PROBLEMS_DIR", "OPTIMIZED_DIR", "TP", "ISL", "OSL",
    "DTYPE", "CONCURRENCY_LEVELS", "MAX_OPTIMIZATION_ATTEMPTS",
    "MAX_CONSECUTIVE_REJECTIONS", "MODE",
]

def check_runtime_md(skill_dir: Path) -> List[Check]:
    checks = []
    content = (skill_dir / "RUNTIME.md").read_text()

    for var in REQUIRED_VARS:
        checks.append(Check("RUNTIME.md", f"declares {var}",
                             var in content))

    checks.append(Check("RUNTIME.md", "has workspace bootstrap section",
                         "Bootstrap" in content or "bootstrap" in content.lower()))
    checks.append(Check("RUNTIME.md", "has phase map",
                         "Phase Map" in content or "phase map" in content.lower() or
                         "optimize" in content and "profile-only" in content))
    checks.append(Check("RUNTIME.md", "mentions no-modify constraint",
                         "/opt/" in content or "system files" in content.lower()))
    checks.append(Check("RUNTIME.md", "mentions progress tracking",
                         "progress" in content.lower()))
    checks.append(Check("RUNTIME.md", "lists bundled scripts",
                         "select_gpus.py" in content and "kernel_agent.py" in content))
    return checks


# ─── Phase Doc Checks ──────────────────────────────────────────────────────

PHASE_REQUIREMENTS = {
    "00-env-setup.md": {
        "has_objective": True,
        "checks": [
            ("detects AMD GPU",      r'rocminfo|rocm.smi|AMD'),
            ("detects NVIDIA GPU",   r'nvidia.smi|NVIDIA'),
            ("writes env_info.json", r'env_info\.json'),
            ("records gpu_arch",     r'gpu_arch'),
        ],
    },
    "01-server-setup.md": {
        "has_objective": True,
        "checks": [
            ("kills EngineCore first",    r'VLLM::EngineCore'),
            ("uses profiler-config",      r'profiler.config'),
            ("starts with profiler",      r'start_profile|profiler'),
            ("has readiness wait loop",   r'MAX_WAIT|SERVER_READY|v1/models'),
            ("fail-fast on fatal errors", r'out of memory|ValidationError|ModuleNotFoundError'),
        ],
    },
    "02-bench-profile.md": {
        "has_objective": True,
        "checks": [
            ("runs concurrency sweep",       r'CONCURRENCY_LEVELS|\[{'),
            ("profiles at middle concurrency", r'middle|len.*//\s*2|CONC_LEVELS'),
            ("uses same ISL/OSL for both",   r'ISL.*OSL|profiling.*ISL'),
            ("validates trace files",        r'traceEvents|trace.*valid'),
        ],
    },
    "03-analysis.md": {
        "has_objective": True,
        "checks": [
            ("runs kernel_breakdown",      r'kernel_breakdown\.py'),
            ("runs extract_shapes",        r'extract_shapes\.py'),
            ("validates shape coverage",   r'has_real_shapes|real_shapes'),
            ("identifies targets",         r'targets\.json|optimization target'),
            ("validates ≥80% coverage",   r'80|coverage'),
        ],
    },
    "04-kernel-optimize.md": {
        "has_objective": True,
        "checks": [
            ("uses kernel_agent setup",    r'kernel_agent\.py.*setup|setup.*kernel_agent'),
            ("correctness gate",           r'correctness|allclose'),
            ("serving-test gate",          r'serving.test|serving_test'),
            ("autotune warning present",   r'autotune|@triton\.autotune'),
            ("shape coverage check",       r'coverage|shape.*cover'),
            ("has stopping condition",     r'consecutive_rejections|max_attempts|stopping'),
            ("reads knowledge base",       r'TRITON_KNOWLEDGE|knowledge.base'),
        ],
    },
    "05-integration.md": {
        "has_objective": True,
        "checks": [
            ("no system files constraint", r'/opt/|/usr/|system file'),
            ("uses PYTHONPATH injection",  r'PYTHONPATH'),
            ("sitecustomize.py",           r'sitecustomize'),
            ("kills EngineCore first",     r'VLLM::EngineCore'),
            ("verifies call count",        r'call_stats|triton.*count|call count'),
            ("correctness gate (Gate 3)",  r'temperature.*0|correctness|token.for.token|Gate 3'),
            ("e2e at ALL concurrencies",   r'CONCURRENCY_LEVELS|all concurren'),
            ("rolls back on regression",   r'roll.?back|rollback|regression'),
        ],
    },
    "06-report.md": {
        "has_objective": True,
        "checks": [
            ("has mandatory verdict table",    r'Verdict.*table|verdict.*table|Verdict.*\|'),
            ("verdict table columns present",  r'Micro Speedup|E2E Speedup'),
            ("verdict column present",         r'\| Verdict \||verdict.*\|'),
            ("lists all verdict types",        r'Integrated|Rolled back|Not integrated'),
            ("per-concurrency TPS table",      r'[Pp]er.concurren|baseline.*TPS|TPS.*baseline'),
            ("shape coverage per kernel",      r'coverage.*kernel|shape.*coverage'),
            ("honest reporting note",          r'honest|failed.*optimiz|Failed.*kernels'),
        ],
    },
}

def check_phase_docs(skill_dir: Path) -> List[Check]:
    checks = []
    phases_dir = skill_dir / "phases"

    for filename, reqs in PHASE_REQUIREMENTS.items():
        p = phases_dir / filename
        if not p.exists():
            checks.append(Check("phases", f"{filename} exists", False, "MISSING"))
            continue

        content = p.read_text()

        if reqs.get("has_objective"):
            checks.append(Check("phases", f"{filename}: has Objective",
                                 "Objective" in content or "objective" in content.lower()))

        for check_name, pattern in reqs.get("checks", []):
            found = re.search(pattern, content, re.IGNORECASE) is not None
            checks.append(Check("phases", f"{filename}: {check_name}", found))

        # Completion block
        checks.append(Check("phases", f"{filename}: has Completion section",
                             "Completion" in content or "Update.*progress" in content))

    return checks


# ─── Python Script Checks ──────────────────────────────────────────────────

SCRIPT_API_CHECKS = {
    "select_gpus.py": [
        ("defines select_gpus function or main", r'def select_gpus|def main|if __name__'),
        ("handles AMD rocm-smi",                 r'rocm.smi|rocminfo'),
        ("handles NVIDIA nvidia-smi",            r'nvidia.smi'),
        ("has fallback",                         r'fallback|except.*pass|torch\.cuda'),
    ],
    "extract_shapes.py": [
        ("extracts from aten::mm",          r'aten::mm|aten::addmm'),
        ("handles aten::linear",            r'aten::linear'),
        ("produces benchmark_shapes",       r'benchmark_shapes'),
        ("handles .json.gz",                r'gzip|\.gz'),
        ("validates record_shapes",         r'record_shapes|Input Dims|input_dims'),
        ("has main function",               r'def main'),
    ],
    "kernel_breakdown.py": [
        ("aggregates kernel durations",     r'total_us|total_duration'),
        ("computes pct_total",              r'pct_total|pct.*total'),
        ("classifies kernels",              r'classify|GEMM|gemm|attention'),
        ("produces category_breakdown",    r'category_breakdown'),
        ("produces top_kernels",            r'top_kernels'),
        ("has main function",               r'def main'),
    ],
    "kernel_agent.py": [
        ("setup subcommand",                r'"setup"|sub.*setup|setup.*sub'),
        ("benchmark subcommand",            r'"benchmark"'),
        ("correctness subcommand",          r'"correctness"'),
        ("accept subcommand",               r'"accept"'),
        ("reject subcommand",               r'"reject"'),
        ("status subcommand",               r'"status"'),
        ("serving-test subcommand",         r'"serving.test"'),
        ("checks torch.allclose",           r'allclose'),
        ("correctness atol=1e-2",           r'atol.*1e-2|1e-2.*atol'),
        ("baseline uses real shapes",       r'real_shapes|benchmark_shapes'),
        ("detects autotune regression",     r'serving.*test|FAIL_REGRESSION'),
        ("accept copies to best_kernel.py", r'best_kernel\.py'),
        ("knowledge base path stored",      r'knowledge_base'),
    ],
    "gemm_patch.py": [
        ("meta_path hook",                  r'meta_path|find_module|load_module'),
        ("patches dispatch_unquantized_gemm", r'dispatch_unquantized_gemm'),
        ("weight transpose note",           r'weight\.t\(\)|\.t\(\)|transpose'),
        ("has fallback on exception",       r'except.*Exception|fallback'),
        ("tracks call stats",               r'call_stats|stats\[.*triton'),
        ("constraint 3: no system files",   r'/opt/|system file|PYTHONPATH'),
        ("sitecustomize injection noted",   r'sitecustomize'),
        ("kills EngineCore before patching", r'EngineCore|engcore'),
    ],
}

def check_scripts(skill_dir: Path) -> List[Check]:
    checks = []
    scripts_dir = skill_dir / "scripts"

    for script_name, reqs in SCRIPT_API_CHECKS.items():
        p = scripts_dir / script_name
        if not p.exists():
            checks.append(Check("scripts", f"{script_name} exists", False, "MISSING"))
            continue

        content = p.read_text()

        # Syntax check
        try:
            ast.parse(content)
            checks.append(Check("scripts", f"{script_name}: valid Python syntax", True))
        except SyntaxError as e:
            checks.append(Check("scripts", f"{script_name}: valid Python syntax", False, str(e)))
            continue

        for check_name, pattern in reqs:
            found = re.search(pattern, content, re.IGNORECASE) is not None
            checks.append(Check("scripts", f"{script_name}: {check_name}", found))

    return checks


# ─── Knowledge Base Checks ─────────────────────────────────────────────────

def check_knowledge_base(skill_dir: Path) -> List[Check]:
    checks = []
    p = skill_dir / "references" / "TRITON_KNOWLEDGE.md"
    if not p.exists():
        return [Check("knowledge", "TRITON_KNOWLEDGE.md exists", False, "MISSING")]

    content = p.read_text()

    kb_checks = [
        ("mentions autotune serving regression",       r'autotune.*serving|serving.*autotune'),
        ("mentions tl.assume(stride > 0)",             r'tl\.assume.*stride|assume.*stride'),
        ("mentions GROUP_SIZE_M",                      r'GROUP_SIZE_M'),
        ("mentions Split-K",                           r'[Ss]plit.K'),
        ("mentions AMD architectures (RDNA3/CDNA3)",   r'RDNA3|CDNA3|gfx11|gfx94'),
        ("mentions wave size",                         r'Wave32|Wave64|wave.*size'),
        ("mentions fp32 accumulator",                  r'fp32.*accum|accum.*fp32'),
        ("has debugging guide",                        r'[Dd]ebugging|debug.*guide'),
        ("mentions Triton JIT file constraint",        r'\.py.*file|jit.*file|file.*jit'),
        ("mentions micro-to-E2E gap",                 r'micro.*E2E|E2E.*micro|micro.to.E2E'),
        ("has stopping condition guidance",            r'[Ss]top.*when|stopping.*condition'),
    ]

    for check_name, pattern in kb_checks:
        found = re.search(pattern, content, re.IGNORECASE) is not None
        checks.append(Check("knowledge", check_name, found))

    return checks


# ─── Cross-File Consistency ────────────────────────────────────────────────

def check_consistency(skill_dir: Path) -> List[Check]:
    checks = []

    skill    = (skill_dir / "SKILL.md").read_text()
    runtime  = (skill_dir / "RUNTIME.md").read_text()
    intake   = (skill_dir / "INTAKE.md").read_text()

    # Phase names consistent across files
    for phase in ["env", "server", "bench-profile", "analysis", "optimize", "integrate", "report"]:
        in_skill   = phase in skill or phase.replace("-", "_") in skill
        in_runtime = phase in runtime or phase.replace("-", "_") in runtime
        # At least SKILL.md or RUNTIME.md should mention each phase
        checks.append(Check("consistency", f"phase '{phase}' in SKILL.md or RUNTIME.md",
                             in_skill or in_runtime))

    # RUNTIME.md lists all scripts from SKILL.md
    for script in ["select_gpus.py", "extract_shapes.py", "kernel_breakdown.py",
                   "kernel_agent.py", "gemm_patch.py"]:
        checks.append(Check("consistency", f"{script} listed in RUNTIME.md",
                             script in runtime))

    # SKILL.md references INTAKE.md and RUNTIME.md
    checks.append(Check("consistency", "SKILL.md -> INTAKE.md link",
                         "INTAKE.md" in skill))
    checks.append(Check("consistency", "SKILL.md -> RUNTIME.md link",
                         "RUNTIME.md" in skill))

    # Serving readiness test mentioned in both Phase 4 and knowledge base
    p4 = (skill_dir / "phases" / "04-kernel-optimize.md").read_text()
    kb = (skill_dir / "references" / "TRITON_KNOWLEDGE.md").read_text()
    checks.append(Check("consistency", "serving-test in Phase 4",
                         "serving-test" in p4 or "serving_test" in p4))
    checks.append(Check("consistency", "autotune warning in Phase 4",
                         "autotune" in p4.lower()))
    checks.append(Check("consistency", "autotune warning in Phase 5",
                         "autotune" in (skill_dir / "phases" / "05-integration.md").read_text().lower()))

    # Verdict table in Phase 6
    p6 = (skill_dir / "phases" / "06-report.md").read_text()
    checks.append(Check("consistency", "verdict table in Phase 6 (MANDATORY)",
                         "Verdict" in p6 and ("Integrated" in p6 or "Rolled back" in p6)))

    # No system files constraint: mentioned in at least Phase 1, 5, and SKILL.md
    for fname, label in [("phases/01-server-setup.md", "Phase 1"),
                          ("phases/05-integration.md", "Phase 5"),
                          ("SKILL.md", "SKILL.md")]:
        c = (skill_dir / fname).read_text()
        checks.append(Check("consistency", f"no-system-files constraint in {label}",
                             "/opt/" in c or "system file" in c.lower() or
                             "No system" in c or "Constraint 3" in c))

    return checks


# ─── Spec Constraint Coverage ──────────────────────────────────────────────

def check_spec_constraints(skill_dir: Path) -> List[Check]:
    """Verify all 7 hard constraints from the spec are addressed."""
    checks = []

    all_content = ""
    for f in skill_dir.rglob("*.md"):
        try:
            all_content += f.read_text()
        except Exception:
            pass

    constraints = [
        ("Constraint 1: Real shapes only",
         r'real shapes|real.*shape|record_shapes|Constraint 1'),
        ("Constraint 2: Five-gate trust chain",
         r'Gate [1-5]|five.gate|trust chain|Constraint 2'),
        ("Constraint 3: No system file modification",
         r'/opt/|/usr/|pip.install|system file|Constraint 3'),
        ("Constraint 4: Same workload for benchmark+profile",
         r'same.*workload|workload.*same|combined.*benchmark|Constraint 4'),
        ("Constraint 5: Kill EngineCore first",
         r'EngineCore.*first|kill.*EngineCore|VLLM::EngineCore|Constraint 5'),
        ("Constraint 6: Honest failure reporting with verdict table",
         r'verdict.*table|Verdict.*table|honest.*fail|Constraint 6'),
        ("Constraint 7: AMD GPU first / bfloat16 default",
         r'AMD.*first|bfloat16.*default|default.*bfloat16|Constraint 7'),
    ]

    for name, pattern in constraints:
        found = re.search(pattern, all_content, re.IGNORECASE) is not None
        checks.append(Check("spec-constraints", name, found))

    return checks


# ─── Main ──────────────────────────────────────────────────────────────────

def run_all_checks(skill_dir: Path, verbose: bool = False) -> Tuple[int, int]:
    all_checks = []

    all_checks += check_files(skill_dir)
    all_checks += check_skill_md(skill_dir)
    all_checks += check_intake_md(skill_dir)
    all_checks += check_runtime_md(skill_dir)
    all_checks += check_phase_docs(skill_dir)
    all_checks += check_scripts(skill_dir)
    all_checks += check_knowledge_base(skill_dir)
    all_checks += check_consistency(skill_dir)
    all_checks += check_spec_constraints(skill_dir)

    passed = sum(1 for c in all_checks if c.passed)
    failed = sum(1 for c in all_checks if not c.passed)

    # Group by category
    categories = {}
    for c in all_checks:
        categories.setdefault(c.category, []).append(c)

    print(f"\n{'='*68}")
    print(f"  vllm-optimize-v2 Structural Validation")
    print(f"  Skill dir: {skill_dir}")
    print(f"{'='*68}\n")

    for cat, chks in categories.items():
        cat_pass = sum(1 for c in chks if c.passed)
        cat_fail = sum(1 for c in chks if not c.passed)
        icon = "PASS" if cat_fail == 0 else f"FAIL ({cat_fail})"
        print(f"  [{icon:12s}] {cat}  ({cat_pass}/{len(chks)})")

        fails = [c for c in chks if not c.passed]
        if verbose:
            for c in chks:
                icon2 = "  OK" if c.passed else " ERR"
                detail = f"  — {c.detail}" if c.detail else ""
                print(f"    [{icon2}] {c.name}{detail}")
        elif fails:
            for c in fails:
                detail = f"  — {c.detail}" if c.detail else ""
                print(f"     [ERR] {c.name}{detail}")

    print(f"\n{'='*68}")
    print(f"  TOTAL: {passed} passed, {failed} failed  ({len(all_checks)} checks)")
    print(f"{'='*68}\n")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Validate vllm-optimize-v2 skill structure")
    parser.add_argument("--skill-dir", default=None,
                        help="Path to vllm-optimize-v2 directory (default: auto-detect)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all checks, not just failures")
    args = parser.parse_args()

    if args.skill_dir:
        skill_dir = Path(args.skill_dir).resolve()
    else:
        # Auto-detect: this script is in tests/, skill is one level up
        skill_dir = Path(__file__).resolve().parent.parent

    if not skill_dir.is_dir():
        print(f"ERROR: skill directory not found: {skill_dir}", file=sys.stderr)
        sys.exit(1)

    # Verify this looks like our skill
    if not (skill_dir / "SKILL.md").exists():
        print(f"ERROR: SKILL.md not found in {skill_dir}", file=sys.stderr)
        sys.exit(1)

    passed, failed = run_all_checks(skill_dir, verbose=args.verbose)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
