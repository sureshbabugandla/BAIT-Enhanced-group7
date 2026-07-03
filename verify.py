#!/usr/bin/env python3
"""
verify.py  --  no-GPU sanity check for BAIT-Enhanced.

Run from the repo root:  python verify.py

Checks (none require torch or a GPU):
  1. every .py file under src/ and scripts/ byte-compiles
  2. the 5 improvement modules import and produce correct output
  3. detector.py / arguments.py contain all expected improvement hooks
  4. (optional) if a tests/ dir exists, run pytest

Exits non-zero if any check fails, so it doubles as a CI gate.
"""
import sys, os, py_compile, importlib

OK, FAIL = "[ OK ]", "[FAIL]"
errors = []

def check(label, cond):
    print(f"  {OK if cond else FAIL} {label}")
    if not cond:
        errors.append(label)

print("=" * 64)
print("BAIT-Enhanced verification (no GPU required)")
print("=" * 64)

# 1. compile everything
print("\n[1] Byte-compile all source files")
compiled = True
for root in ("src", "scripts"):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                try:
                    py_compile.compile(p, doraise=True)
                except py_compile.PyCompileError as e:
                    compiled = False
                    print(f"   compile error in {p}: {e}")
check("all src/ and scripts/ files compile", compiled)

# 2. import + exercise the 5 numpy-only modules
print("\n[2] Improvement modules import and run")
sys.path.insert(0, ".")
try:
    import numpy as np
    from src.core.robust_qscore import bootstrap_qscore
    from src.core.conformal_threshold import conformal_threshold
    from src.core.token_prioritizer import prioritize_initial_tokens
    from src.core.baseline_calibration import baseline_adjusted_qscore
    from src.eval.judge_backends import build_judge

    r = bootstrap_qscore(np.full((20, 10), 0.8))
    check("A robust_qscore q_low~0.8", abs(r.q_low - 0.8) < 0.05)
    check("C conformal threshold in (0,1)", 0 < conformal_threshold(np.random.rand(50), 0.05).tau < 1)
    check("D prioritizer orders by prob", prioritize_initial_tokens(np.array([0.1, 0.9, 0.3])).order[0] == 1)
    check("E baseline high when target>>baseline",
          baseline_adjusted_qscore(np.full((5, 5), 0.8), np.full((5, 5), 0.15)).q_adjusted > 0.5)
    check("B judge(none) accepts", build_judge("none").judge("x").is_suspicious)
except Exception as e:
    check(f"modules import/run (got: {e})", False)

# 3. hooks present in the modified files
print("\n[3] Improvement hooks wired into detector.py / arguments.py")
det = open("src/core/detector.py").read()
arg = open("src/config/arguments.py").read()
for token in ["bootstrap_qscore", "build_judge", "conformal_threshold",
              "_prioritize_dataloader", "baseline_adjusted_qscore", "q_std"]:
    check(f"detector.py uses {token}", token in det)
check("detector.py has no stale judge_client", "judge_client" not in det)
for flag in ["judge_backend", "use_robust_qscore", "conformal_alpha",
             "prioritize_initial_tokens", "use_baseline_calibration"]:
    check(f"arguments.py exposes {flag}", flag in arg)

# 4. optional pytest
print("\n[4] Unit tests (optional)")
if os.path.isdir("tests"):
    import subprocess
    rc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"]).returncode
    check("pytest tests/ passed", rc == 0)
else:
    print("  [skip] no tests/ directory committed (add it to run the 22 unit tests)")

print("\n" + "=" * 64)
if errors:
    print(f"RESULT: {len(errors)} check(s) FAILED:")
    for e in errors:
        print("   -", e)
    sys.exit(1)
print("RESULT: ALL NO-GPU CHECKS PASSED")
print("Next: run a single-model GPU scan (see step 3 in the message).")
