"""Unit tests for D+ token-search optimizer (src/core/token_optimizer.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.token_optimizer import (   # noqa: E402
    plan_scan, early_stop_cost, audit_survival,
)


def test_floor_and_nucleus_shrink():
    V = 1000
    probs = np.zeros(V); probs[:50] = np.linspace(0.2, 0.001, 50); probs /= probs.sum()
    plan = plan_scan(probs, prob_floor=1e-4, p=0.999)
    assert plan.n_candidates < V and plan.reduction > 0.5


def test_true_token_survival_guard():
    V = 1000
    probs = np.zeros(V); probs[:50] = np.linspace(0.2, 0.001, 50); probs /= probs.sum()
    plan = plan_scan(probs, prob_floor=1e-6, p=0.9999)
    assert plan.survives(10) and 0.0 < early_stop_cost(plan, 10) <= 1.0


def test_audit_flags_unsafe_floor():
    V = 2000
    ftps, trues, bans = [], [], []
    for i in range(21):
        p = np.zeros(V); p[:100] = 1.0 / np.arange(1, 101); p /= p.sum()
        ftps.append(p); bans.append(np.zeros(V, bool))
        trues.append(int(np.argsort(-p)[[1, 50, 95][i % 3]]))
    rows = audit_survival(ftps, trues, bans, ps=(1.0,), floors=(1e-6, 1e-2))
    safe = next(r for r in rows if r.floor == 1e-6)
    risky = next(r for r in rows if r.floor == 1e-2)
    assert safe.survival_rate >= risky.survival_rate
