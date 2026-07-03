"""Unit tests for Improvements C (conformal), D (prioritizer), E (baseline)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.conformal_threshold import (   # noqa: E402
    conformal_threshold, realized_fpr, realized_tpr)
from core.token_prioritizer import (      # noqa: E402
    prioritize_initial_tokens, expected_scan_fraction)
from core.baseline_calibration import baseline_adjusted_qscore   # noqa: E402


# ---------- Improvement C ----------
def test_conformal_controls_fpr_on_average():
    rng = np.random.default_rng(0)
    target = 0.10
    fprs = []
    for _ in range(200):
        benign = rng.beta(2, 8, size=60)           # benign score cloud
        ct = conformal_threshold(benign, alpha=target)
        test_benign = rng.beta(2, 8, size=400)     # fresh benign test set
        fprs.append(realized_fpr(test_benign, ct.tau))
    assert abs(np.mean(fprs) - target) < 0.03      # realized ~ target


def test_conformal_empty():
    ct = conformal_threshold(np.array([]), alpha=0.05)
    assert ct.n_calib == 0 and ct.tau == float("inf")


def test_tpr_helper():
    assert realized_tpr(np.array([0.9, 0.95, 0.4]), tau=0.5) == 2 / 3


# ---------- Improvement D ----------
def test_prioritizer_orders_by_prob_and_bans():
    probs = np.array([0.01, 0.5, 0.2, 0.9, 0.05])
    plan = prioritize_initial_tokens(probs, banned_ids=[3])   # ban the top one
    assert plan.order[0] == 1                                  # next-highest first
    assert 3 not in plan.order
    assert plan.n_candidates == 4


def test_prioritizer_speedup_when_target_has_high_prob():
    rng = np.random.default_rng(1)
    V = 5000
    probs = rng.beta(1, 50, size=V)        # most tokens near zero
    true_tok = 1234
    probs[true_tok] = 0.4                    # backdoor first token stands out
    plan = prioritize_initial_tokens(probs)
    frac = expected_scan_fraction(plan, true_tok)
    assert frac < 0.02                       # found in < 2% of the vocab


def test_prioritizer_masked_target_is_flagged():
    probs = np.array([0.3, 0.2, 0.1])
    plan = prioritize_initial_tokens(probs, banned_ids=[0])
    # token 0 was banned -> expected fraction reports the worst case (1.0)
    assert expected_scan_fraction(plan, 0) == 1.0


# ---------- Improvement E ----------
def test_baseline_kills_common_word_false_positive():
    # common-word benign: high raw Q but high baseline too -> adjusted ~ 0
    target = np.full((20, 10), 0.62)
    baseline = np.full((20, 10), 0.58)
    r = baseline_adjusted_qscore(target, baseline, mode="diff")
    assert r.q_raw > 0.6
    assert r.q_adjusted < 0.1


def test_baseline_keeps_true_backdoor():
    # backdoor: high raw Q, low baseline -> adjusted stays high
    target = np.full((20, 10), 0.85)
    baseline = np.full((20, 10), 0.15)
    r = baseline_adjusted_qscore(target, baseline, mode="diff")
    assert r.q_adjusted > 0.6


def test_baseline_lift_mode_bounded():
    r = baseline_adjusted_qscore(np.full((5, 5), 0.9),
                                 np.full((5, 5), 0.5), mode="lift")
    assert 0.0 <= r.q_adjusted <= 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
