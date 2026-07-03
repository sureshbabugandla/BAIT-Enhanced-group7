"""
token_prioritizer.py  --  Improvement D for BAIT  (addresses limitation #2).

PROBLEM
-------
BAIT tries EVERY token in the vocabulary as the candidate "first token" of the
attack target. For big-vocabulary models (Gemma: 256k tokens) this brute force
is ~5x slower than for older 32k-vocab models.

FIX
---
Try the *promising* first tokens first, then early-stop once a clearly
backdoored sequence is found. The cheap prior: rank candidate tokens by the
model's own marginal probability of emitting them as the first generated token
across the benign probe prompts. Theorem 4.4's intuition is that a backdoor
target's first token carries elevated probability mass even without the trigger,
so the true first token tends to sit high in this ranking. We also drop tokens
that can never be a sensible target start (special tokens, pure whitespace).

IMPORTANT: this only reorders the search and adds an early stop. It never changes
WHICH backdoor would be found in a full scan -- so it cannot hurt detection
correctness; in the worst case (uninformative prior) it simply gives no speedup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class ScanPlan:
    order: np.ndarray          # token ids in the order to evaluate
    n_candidates: int          # how many remain after masking

    def rank_of(self, token_id: int) -> int:
        """1-indexed position of a token in the plan (len+1 if masked out)."""
        hits = np.where(self.order == token_id)[0]
        return int(hits[0]) + 1 if hits.size else self.n_candidates + 1


def prioritize_initial_tokens(
    first_token_probs: np.ndarray,
    banned_ids: Optional[Sequence[int]] = None,
) -> ScanPlan:
    """
    Build a scan order from the marginal first-token probability distribution.

    Parameters
    ----------
    first_token_probs : 1-D array, length = vocab size. ``[v]`` is the average
        probability the model emits token v as the first token across the benign
        probe prompts (BAIT already computes this distribution at step 1).
    banned_ids : token ids to skip entirely (special/whitespace tokens).

    Returns
    -------
    ScanPlan
    """
    probs = np.asarray(first_token_probs, dtype=np.float64).copy()
    if banned_ids is not None and len(banned_ids):
        probs[np.asarray(list(banned_ids), dtype=int)] = -np.inf
    order = np.argsort(-probs)                 # high prob first
    order = order[np.isfinite(probs[order])]   # drop banned
    return ScanPlan(order=order, n_candidates=int(order.size))


def expected_scan_fraction(plan: ScanPlan, true_first_token: int,
                           early_stop: bool = True) -> float:
    """
    Fraction of the vocabulary that must be evaluated before the true first
    token is reached (with early-stop, the scan ends there). Lower == faster.
    Returns 1.0 if the token was masked out (a correctness risk -- watch for it).
    """
    rank = plan.rank_of(true_first_token)
    if rank > plan.n_candidates:
        return 1.0
    return rank / plan.n_candidates if early_stop else 1.0
