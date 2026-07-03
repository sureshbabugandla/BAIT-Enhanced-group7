"""Unit tests for Improvement A: robust_qscore."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.robust_qscore import (   # noqa: E402
    bootstrap_qscore, decide_backdoor, calibrate_threshold, RobustQResult,
)


def test_empty_input():
    r = bootstrap_qscore(np.zeros((0, 0)))
    assert r.q_mean == 0.0 and r.q_low == 0.0


def test_single_prompt_falls_back():
    r = bootstrap_qscore(np.array([[0.8, 0.7, 0.9]]))
    assert r.n_prompts == 1
    assert r.q_low == r.q_mean        # cannot bootstrap one sample


def test_high_mean_low_variance_has_high_lower_bound():
    rng = np.random.default_rng(0)
    mat = rng.beta(0.85 * 40, 0.15 * 40, size=(20, 10))   # tight, high mean
    r = bootstrap_qscore(mat, seed=1)
    assert r.q_low > 0.7
    assert r.q_low <= r.q_mean + 1e-9                       # bound <= mean


def test_high_variance_is_penalised():
    rng = np.random.default_rng(0)
    tight = rng.beta(0.6 * 50, 0.4 * 50, size=(20, 10))    # same-ish mean, tight
    loose = rng.beta(0.6 * 4, 0.4 * 4, size=(20, 10))      # same-ish mean, loose
    r_tight = bootstrap_qscore(tight, seed=2)
    r_loose = bootstrap_qscore(loose, seed=2)
    # similar means, but the loose one must have a LOWER confidence bound
    assert abs(r_tight.q_mean - r_loose.q_mean) < 0.15
    assert r_loose.q_low < r_tight.q_low


def test_drop_min_step_changes_steps_used():
    mat = np.full((20, 10), 0.9)
    mat[:, 3] = 0.1                                          # one bad step
    r_drop = bootstrap_qscore(mat, drop_min_step=True, seed=3)
    r_keep = bootstrap_qscore(mat, drop_min_step=False, seed=3)
    assert r_drop.n_steps_used == 9 and r_keep.n_steps_used == 10
    assert r_drop.q_mean > r_keep.q_mean                    # dropping the bad step helps


def test_decide_backdoor():
    assert decide_backdoor(RobustQResult(0.9, 0.88, 0.01, 20, 10), tau=0.85)
    assert not decide_backdoor(RobustQResult(0.9, 0.80, 0.05, 20, 10), tau=0.85)


def test_calibrate_threshold_hits_tpr():
    scores = np.concatenate([np.linspace(0.6, 0.95, 50),   # positives
                             np.linspace(0.0, 0.5, 50)])    # negatives
    labels = np.array([1] * 50 + [0] * 50)
    tau = calibrate_threshold(scores, labels, target_tpr=0.9)
    realized = (scores[labels == 1] >= tau).mean()
    assert realized >= 0.9 - 1e-6


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
