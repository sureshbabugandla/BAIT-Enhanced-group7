"""
conformal_threshold.py  --  Improvement C for BAIT  (addresses limitation #9).

PROBLEM
-------
BAIT decides "backdoored" when the Q-SCORE crosses a HAND-TUNED cutoff (0.85 /
0.9 in the paper/repo). That number was picked by trial on specific datasets, so
on a new domain the real false-positive rate is unknown and can drift.

FIX
---
Turn the cutoff into a *calibrated* threshold with a finite-sample guarantee.
Given a small set of models you trust to be benign (a calibration set), pick the
threshold using a conformal quantile. Under the mild assumption that benign
models are exchangeable, this guarantees the false-positive rate is at most the
target alpha (e.g. 5%) -- a real statistical promise instead of a guessed number.

numpy-only, no side effects -> unit-testable and drop-in.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibratedThreshold:
    tau: float            # the decision threshold (flag if score > tau)
    alpha: float          # target false-positive rate
    n_calib: int          # size of the calibration (benign) set

    def decide(self, score: float) -> bool:
        return score > self.tau


def conformal_threshold(benign_scores: np.ndarray,
                        alpha: float = 0.05) -> CalibratedThreshold:
    """
    One-sided conformal upper threshold for a "higher == more suspicious" score.

    We flag a model as backdoored when ``score > tau``. Choosing
    ``tau`` as the ceil((1-alpha)(n+1))-th order statistic of the benign scores
    gives the finite-sample guarantee  P(benign flagged) <= alpha  for an
    exchangeable benign population.

    Parameters
    ----------
    benign_scores : 1-D array of Q-SCOREs from models known/assumed benign.
    alpha : target false-positive rate in (0, 1).

    Returns
    -------
    CalibratedThreshold
    """
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return CalibratedThreshold(tau=float("inf"), alpha=alpha, n_calib=0)
    # 1-indexed rank of the conformal quantile
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    tau = float(s[k - 1])
    return CalibratedThreshold(tau=tau, alpha=alpha, n_calib=n)


def realized_fpr(benign_scores: np.ndarray, tau: float) -> float:
    """Fraction of benign models that would be (wrongly) flagged at this tau."""
    b = np.asarray(benign_scores, dtype=np.float64)
    return float((b > tau).mean()) if b.size else 0.0


def realized_tpr(poison_scores: np.ndarray, tau: float) -> float:
    """Fraction of poisoned models correctly flagged at this tau."""
    p = np.asarray(poison_scores, dtype=np.float64)
    return float((p > tau).mean()) if p.size else 0.0
