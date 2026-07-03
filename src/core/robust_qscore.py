"""
robust_qscore.py  --  Improvement A for BAIT.

Variance-aware Q-SCORE with a bootstrap confidence bound.

WHY
---
The original BAIT computes Q-SCORE as the *mean* per-step probability of the
inverted target, averaged over a tiny set of prompts (20 by default). With only
20 prompts the estimate is noisy (the paper's Figure 6b: the same true token can
swing from 0.16 to 0.77 across draws). A benign sequence that gets a *lucky* high
mean on one draw can cross the decision threshold (false positive), and a true
target that gets an *unlucky* low mean can fall below it (false negative).

FIX
---
Treat the per-prompt scores as a sample. Bootstrap-resample them to obtain a
distribution of the mean Q-SCORE, and decide using the LOWER confidence bound
(e.g. the 5th percentile) instead of the raw mean. A high-variance "lucky"
benign sequence has a low lower-bound and is rejected; a high-mean low-variance
true target keeps a high lower-bound and is retained. This turns a heuristic
point estimate into a statistically grounded decision rule and measurably
reduces decision variance across prompt draws.

This module is dependency-light (numpy only) and has no side effects, so it can
be unit-tested in isolation and dropped straight into ``src/core/`` of the repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobustQResult:
    """Container for a robust Q-SCORE computation."""
    q_mean: float          # plain mean Q-SCORE (what original BAIT reports)
    q_low: float           # lower confidence bound (the robust score)
    q_std: float           # bootstrap std-dev (uncertainty indicator)
    n_prompts: int         # how many prompts the estimate is based on
    n_steps_used: int      # target steps used after optional min-step drop

    def __str__(self) -> str:
        return (f"RobustQResult(q_mean={self.q_mean:.4f}, q_low={self.q_low:.4f}, "
                f"q_std={self.q_std:.4f}, n_prompts={self.n_prompts}, "
                f"n_steps_used={self.n_steps_used})")


def bootstrap_qscore(
    per_prompt_step_probs: np.ndarray,
    n_boot: int = 1000,
    low_pct: float = 5.0,
    drop_min_step: bool = True,
    seed: int = 0,
) -> RobustQResult:
    """
    Compute a variance-aware Q-SCORE with a bootstrap lower confidence bound.

    Parameters
    ----------
    per_prompt_step_probs : np.ndarray, shape [n_prompts, n_steps]
        ``[p, t]`` is the model's probability for the inverted target token at
        step ``t`` when conditioned on the previously inverted tokens and
        prompt ``p``. This is exactly the quantity BAIT already computes inside
        ``full_inversion`` -- we just keep it per-prompt instead of pre-averaging.
    n_boot : int
        Number of bootstrap resamples.
    low_pct : float
        Percentile (0-100) used for the lower confidence bound. 5.0 -> a one-sided
        ~95% bound.
    drop_min_step : bool
        Reproduce BAIT's robustness trick of dropping the single weakest step
        before averaging.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    RobustQResult
    """
    X = np.asarray(per_prompt_step_probs, dtype=np.float64)
    if X.ndim == 1:                      # a single prompt's step vector
        X = X[None, :]
    if X.ndim != 2 or X.size == 0:
        return RobustQResult(0.0, 0.0, 0.0, 0, 0)

    # Clip to valid probability range to be safe against numerical noise.
    X = np.clip(X, 0.0, 1.0)

    # BAIT's trick: drop the single weakest step (computed on the column means).
    if drop_min_step and X.shape[1] > 1:
        step_means = X.mean(axis=0)
        keep = np.ones(X.shape[1], dtype=bool)
        keep[int(step_means.argmin())] = False
        X = X[:, keep]

    per_prompt_q = X.mean(axis=1)        # one Q per prompt, shape [n_prompts]
    n = per_prompt_q.shape[0]

    q_mean = float(per_prompt_q.mean())

    if n == 1:
        # Cannot bootstrap a single observation; fall back to the point estimate.
        return RobustQResult(q_mean, q_mean, 0.0, n, X.shape[1])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = per_prompt_q[idx].mean(axis=1)     # shape [n_boot]
    q_low = float(np.percentile(boot_means, low_pct))
    q_std = float(boot_means.std(ddof=0))

    return RobustQResult(q_mean=q_mean, q_low=q_low, q_std=q_std,
                         n_prompts=n, n_steps_used=X.shape[1])


def decide_backdoor(result: RobustQResult, tau: float = 0.85) -> bool:
    """
    Decide whether the model is backdoored using the robust lower bound.

    Using ``q_low`` (rather than ``q_mean``) makes the decision conservative
    against high-variance "lucky" benign sequences while preserving confident
    true targets.
    """
    return result.q_low > tau


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    Pick the decision threshold that achieves at least ``target_tpr`` on a
    labelled validation set. Lets us compare the mean rule and the bootstrap
    rule at a *matched* operating point (a fair comparison).

    Parameters
    ----------
    scores : np.ndarray   detector scores (higher == more likely backdoored)
    labels : np.ndarray   1 == backdoored, 0 == benign
    target_tpr : float    desired true-positive rate

    Returns
    -------
    float : threshold tau
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    if pos.size == 0:
        return float(np.median(scores))
    # threshold = the value such that target_tpr of positives are >= it
    return float(np.quantile(pos, 1.0 - target_tpr))
