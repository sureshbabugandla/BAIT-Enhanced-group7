"""
baseline_calibration.py  --  Improvement E for BAIT  (addresses limitation #5).

PROBLEM
-------
BAIT's signal weakens when the malicious target is made of ORDINARY words. A
benign model also assigns fairly high probability to common continuations, so a
benign sequence of common words can post a high raw Q-SCORE -> false positive.
(The paper's Assumption 4.2 -- that the target is rare -- can fail here.)

FIX
---
Don't judge a sequence by its raw probability; judge it by how much it EXCEEDS
what ordinary language would predict. Estimate a per-sequence benign baseline
(the same continuation's probability under a clean reference -- e.g. the model on
clean prompts, or a benign control model) and subtract/normalize it out.

    raw      Q = mean_t P(target_t | prefix)
    baseline B = mean_t P(target_t | prefix) under clean reference
    adjusted   = (Q - B)            ("diff" mode), or
                 (Q - B) / (1 - B)  ("lift" mode, bounded, emphasises rare hits)

A true backdoor keeps a HIGH adjusted score (forced unusual output -> low B);
a common-word benign decoy collapses to ~0 (Q and B both high).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdjustedQ:
    q_raw: float
    q_baseline: float
    q_adjusted: float
    mode: str


def _mean_prob(step_probs: np.ndarray) -> float:
    x = np.asarray(step_probs, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.clip(x, 0.0, 1.0).mean())


def baseline_adjusted_qscore(
    target_step_probs: np.ndarray,
    baseline_step_probs: np.ndarray,
    mode: str = "diff",
) -> AdjustedQ:
    """
    Compute a baseline-corrected Q-SCORE.

    Parameters
    ----------
    target_step_probs : per-step probabilities of the inverted target under the
        SUSPECT model (1-D, or [n_prompts, n_steps] -- it is averaged).
    baseline_step_probs : the SAME sequence's per-step probabilities under a
        clean reference (clean prompts / benign control model).
    mode : "diff" -> Q - B ;  "lift" -> (Q - B) / (1 - B).

    Returns
    -------
    AdjustedQ
    """
    q = _mean_prob(target_step_probs)
    b = _mean_prob(baseline_step_probs)
    if mode == "lift":
        denom = max(1.0 - b, 1e-6)
        adj = (q - b) / denom
    else:  # "diff"
        adj = q - b
    adj = float(np.clip(adj, 0.0, 1.0))
    return AdjustedQ(q_raw=q, q_baseline=b, q_adjusted=adj, mode=mode)
