"""
token_optimizer.py  --  optimize the initial-token search in BAIT.

BAIT's cost is dominated by enumerating EVERY vocabulary token as the candidate
first token of the attack target (|V| up to 256k for Gemma). The repo's
token_prioritizer only *reorders*; it never shrinks the set, so worst-case cost
is unchanged. This module SHRINKS the candidate set safely and then prioritizes
+ early-stops, with a correctness guard.

Four tiers, increasing aggressiveness (and the only one with any recall risk is
opt-in and measured):

  T1  HARD BAN        always safe. Special tokens, pure whitespace/punctuation,
                      byte-fallback tokens, and (configurable) non-word-initial
                      sub-word continuations can never start a coherent target
                      response. Computed once per tokenizer -> reused every scan.
  T2  PROB FLOOR      drop tokens whose marginal first-token probability across
                      the benign probes is below a tiny floor. Theorem 4.4 says a
                      backdoor's first token carries elevated marginal mass, so a
                      near-zero-probability token is overwhelmingly unlikely to be
                      it. Risk is bounded by the floor and reported.
  T3  NUCLEUS (top-p) keep only the smallest prefix of tokens whose cumulative
                      marginal probability reaches `p` (e.g. 0.9999). Opt-in.
  T4  PRIORITIZE+STOP order survivors by marginal probability and early-stop once
                      a candidate yields a Q-SCORE clearing the threshold with a
                      margin. Pure reordering -> never changes the verdict of a
                      full scan; only the time to reach it.

Safety: `audit_survival` checks that the (known, in eval only) true first token
survives each tier, so you can pick a `p`/floor that gives speedup at zero recall
loss on your model zoo before trusting it on unknown models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable
import numpy as np


# --------------------------------------------------------------------------- #
#  T1: static ban mask from a real HF tokenizer (computed once per arch)       #
# --------------------------------------------------------------------------- #
def build_ban_mask(tokenizer, word_initial_only: bool = True) -> np.ndarray:
    """
    Return a boolean array [vocab] where True == banned (skip this token).

    Uses only the tokenizer, no model forward pass. Safe to cache per
    architecture. `word_initial_only=True` additionally bans sub-word
    continuation tokens (those that do not begin a new word), which is safe when
    the attack target is natural text that starts at a word boundary -- the
    universal case in the paper (URLs, commands, sentences). Set False if you
    must allow mid-word target starts.
    """
    vocab = tokenizer.get_vocab()                    # {token_str: id}
    V = max(vocab.values()) + 1
    ban = np.zeros(V, dtype=bool)

    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for tid in special:
        if 0 <= tid < V:
            ban[tid] = True

    # SentencePiece uses U+2581 (lower-one-eighth block) as the space marker;
    # GPT-2/BPE byte tokenizers use 'Ġ'. A word-initial token starts with one.
    SP, GP = "\u2581", "\u0120"
    for tok, tid in vocab.items():
        if tid >= V or ban[tid]:
            continue
        # strip the leading word-boundary marker for inspection
        core = tok
        is_word_initial = tok.startswith(SP) or tok.startswith(GP)
        if is_word_initial:
            core = tok[1:]
        if core == "":                               # pure whitespace token
            ban[tid] = True; continue
        if all((not ch.isalnum()) for ch in core):   # punctuation/symbol-only
            ban[tid] = True; continue
        # byte-fallback tokens like "<0x0A>"
        if core.startswith("<0x") and core.endswith(">"):
            ban[tid] = True; continue
        if word_initial_only and not is_word_initial:
            ban[tid] = True; continue
    return ban


# --------------------------------------------------------------------------- #
#  T2/T3/T4: dynamic reduction from the marginal first-token distribution      #
# --------------------------------------------------------------------------- #
@dataclass
class ScanPlan:
    order: np.ndarray            # token ids to evaluate, best first
    n_total: int                 # original vocab size
    n_after_ban: int             # after T1
    n_after_floor: int           # after T2
    n_candidates: int            # after T3 (what actually gets scanned, pre early-stop)
    p: float
    floor: float

    @property
    def reduction(self) -> float:
        return 1.0 - self.n_candidates / max(self.n_total, 1)

    def rank_of(self, token_id: int) -> int:
        hit = np.where(self.order == token_id)[0]
        return int(hit[0]) + 1 if hit.size else self.n_candidates + 1

    def survives(self, true_first_token: Optional[int]) -> bool:
        if true_first_token is None:
            return True
        return self.rank_of(true_first_token) <= self.n_candidates


def plan_scan(first_token_probs: np.ndarray,
              ban_mask: Optional[np.ndarray] = None,
              banned_ids: Optional[Sequence[int]] = None,
              prob_floor: float = 1e-6,
              p: float = 0.9999,
              top_k: Optional[int] = None) -> ScanPlan:
    """
    Build an optimized candidate order from the marginal first-token distribution.

    Parameters
    ----------
    first_token_probs : [vocab] mean P(token = first generated token) over the
        benign probe prompts. BAIT already computes this at step 1.
    ban_mask : bool[vocab] from build_ban_mask (T1). Optional.
    banned_ids : extra ids to ban.
    prob_floor : T2 -- drop tokens below this marginal probability.
    p : T3 nucleus mass to retain (1.0 disables T3).
    top_k : T-topk (the BAIT-Lite ``TOP_K_FILTER``) -- after ranking, keep only
        the ``top_k`` most probable survivors. This is the vocabulary-pruning
        knob the team's report declared but left inactive; here it is applied on
        top of the (verdict-preserving) T1 ban and, if set, capped last so it is
        the tightest of the active filters. ``None`` disables it.
    """
    probs = np.asarray(first_token_probs, dtype=np.float64).copy()
    V = probs.shape[0]

    masked = np.zeros(V, dtype=bool)
    if ban_mask is not None:
        masked |= ban_mask[:V]
    if banned_ids:
        masked[np.asarray(list(banned_ids), dtype=int)] = True
    probs[masked] = -np.inf
    n_after_ban = int(np.isfinite(probs).sum())

    # T2 probability floor
    floor_mask = probs < prob_floor
    probs[floor_mask] = -np.inf
    n_after_floor = int(np.isfinite(probs).sum())

    order = np.argsort(-probs)
    order = order[np.isfinite(probs[order])]         # finite, high-prob first

    # T3 nucleus: keep smallest prefix reaching cumulative mass p
    if 0 < p < 1.0 and order.size:
        kept = probs[order].copy()
        kept[~np.isfinite(kept)] = 0.0
        total = kept.sum()
        if total > 0:
            cum = np.cumsum(kept) / total
            keep_n = int(np.searchsorted(cum, p) + 1)
            order = order[:keep_n]

    # T-topk (BAIT-Lite TOP_K_FILTER): hard cap on the number of candidates.
    # Applied last so it is the tightest active filter. Because `order` is sorted
    # high-probability first, this keeps the K most likely initial tokens.
    if top_k is not None and top_k > 0 and order.size > top_k:
        order = order[:top_k]

    return ScanPlan(order=order, n_total=V, n_after_ban=n_after_ban,
                    n_after_floor=n_after_floor, n_candidates=int(order.size),
                    p=p, floor=prob_floor)


def early_stop_cost(plan: ScanPlan, true_first_token: Optional[int]) -> float:
    """
    Fraction of the ORIGINAL vocab actually evaluated with T4 early-stop:
    we stop at the true first token (a confident backdoor is found there).
    Benign models have no true token -> the whole reduced set is scanned.
    Returns 1.0 if the true token was pruned (a recall FAILURE to watch for).
    """
    if true_first_token is None:
        return plan.n_candidates / max(plan.n_total, 1)
    if not plan.survives(true_first_token):
        return 1.0
    return plan.rank_of(true_first_token) / max(plan.n_total, 1)


# --------------------------------------------------------------------------- #
#  Safety audit: pick p/floor that give speedup at ZERO recall loss            #
# --------------------------------------------------------------------------- #
@dataclass
class AuditRow:
    p: float
    floor: float
    mean_reduction: float
    survival_rate: float          # fraction of poisoned models whose true token survived
    mean_scan_fraction_poison: float
    mean_scan_fraction_benign: float


def audit_survival(first_token_probs_list, true_first_tokens, ban_masks,
                   ps=(1.0, 0.9999, 0.999, 0.99), floors=(0.0, 1e-6, 1e-5)) -> list[AuditRow]:
    """
    Grid over (p, floor) reporting realized reduction and -- crucially -- the
    survival rate of the true backdoor first token, so the operator can choose
    the most aggressive setting that still keeps survival_rate == 1.0.
    """
    rows = []
    poison_idx = [i for i, t in enumerate(true_first_tokens) if t is not None]
    for p in ps:
        for fl in floors:
            reds, surv, sf_p, sf_b = [], [], [], []
            for ftp, tft, bm in zip(first_token_probs_list, true_first_tokens, ban_masks):
                plan = plan_scan(ftp, ban_mask=bm, prob_floor=fl, p=p)
                reds.append(plan.reduction)
                sf = early_stop_cost(plan, tft)
                if tft is None:
                    sf_b.append(sf)
                else:
                    surv.append(plan.survives(tft)); sf_p.append(sf)
            rows.append(AuditRow(
                p=p, floor=fl,
                mean_reduction=float(np.mean(reds)),
                survival_rate=float(np.mean(surv)) if surv else 1.0,
                mean_scan_fraction_poison=float(np.mean(sf_p)) if sf_p else 0.0,
                mean_scan_fraction_benign=float(np.mean(sf_b)) if sf_b else 0.0))
    return rows
