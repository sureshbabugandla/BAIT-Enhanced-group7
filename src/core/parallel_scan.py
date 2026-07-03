"""
parallel_scan.py  --  parallelized initial-token scanning for BAIT-Enhanced.

Origin
------
This module extracts and generalises the *parallelization* direction of the
team's BAIT-Lite work (the report's per-prompt scanning loop and its future-work
note on activating pruning + scaling the scan). The original BAIT scans candidate
initial tokens sequentially -- one batch after another through the dataloader.
Since each candidate first token induces an INDEPENDENT target-inversion, the
scan is embarrassingly parallel across candidates.

What this provides
------------------
`parallel_initial_token_scan(...)` runs a user-supplied per-candidate scoring
function over a list of candidate initial-token ids concurrently, using a thread
pool. Threads (not processes) are the right tool here because the heavy work is
either GPU inference or a blocking network call (Ollama / OpenAI) -- both release
the GIL, so threads give real overlap without the cost of serialising models
across processes.

It works for BOTH scan paths:
  * GPU model-zoo path : `score_fn` runs a forward pass for one candidate.
  * black-box / Ollama path : `score_fn` issues the HTTP generate call; many
    candidates are in flight at once, which is where the wall-clock win is large.

The scan is verdict-preserving: parallelism only changes the ORDER/TIMING of
independent evaluations, never their results. An optional `early_stop_tau` lets
the pool short-circuit once a confidently-backdoored candidate is found, mirroring
BAIT's early-stop while remaining safe (it only stops *earlier*, never later than
a candidate that already clears the threshold).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable, List, Optional, Sequence, Tuple

try:                                     # tqdm is already a project dependency
    from tqdm import tqdm
except Exception:                        # pragma: no cover
    def tqdm(x, **k):                    # minimal fallback
        return x


@dataclass
class CandidateResult:
    token_id: int
    q_score: float
    invert_target: str
    extra: dict


@dataclass
class ParallelScanResult:
    best: Optional[CandidateResult]
    n_evaluated: int
    n_total: int
    stopped_early: bool


def parallel_initial_token_scan(
    candidate_token_ids: Sequence[int],
    score_fn: Callable[[int], Tuple[float, str, dict]],
    max_workers: int = 4,
    early_stop_tau: Optional[float] = None,
    show_progress: bool = True,
    desc: str = "Parallel init-token scan",
) -> ParallelScanResult:
    """
    Evaluate candidate initial tokens concurrently and keep the best.

    Parameters
    ----------
    candidate_token_ids : ordered list of initial-token ids to try. Pass the
        pruned/prioritised order from ``plan_scan`` so the most promising tokens
        are submitted first (helps early-stop trigger sooner).
    score_fn : callable(token_id) -> (q_score, invert_target, extra_dict).
        Must be thread-safe with respect to shared model state. For GPU models
        under ``@torch.no_grad()`` inference is read-only and safe; for network
        backends each call is independent.
    max_workers : size of the thread pool (the report's parallelism knob).
    early_stop_tau : if set, stop submitting/collecting once any candidate's
        q_score exceeds this threshold. Safe: only stops *earlier*.
    """
    ids = list(candidate_token_ids)
    n_total = len(ids)
    if n_total == 0:
        return ParallelScanResult(None, 0, 0, False)
    if max_workers <= 1:
        # Sequential fallback (identical results, no threads).
        best = None
        stopped = False
        for i, tid in enumerate(tqdm(ids, desc=desc, disable=not show_progress), 1):
            q, tgt, extra = score_fn(tid)
            if best is None or q > best.q_score:
                best = CandidateResult(tid, q, tgt, extra)
            if early_stop_tau is not None and q > early_stop_tau:
                stopped = True
                return ParallelScanResult(best, i, n_total, True)
        return ParallelScanResult(best, n_total, n_total, stopped)

    best: Optional[CandidateResult] = None
    lock = Lock()
    stop = Event()
    n_done = 0

    def _work(tid: int):
        if stop.is_set():
            return None
        q, tgt, extra = score_fn(tid)
        return CandidateResult(tid, q, tgt, extra)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, tid): tid for tid in ids}
        bar = tqdm(total=n_total, desc=desc, disable=not show_progress)
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            bar.update(1)
            if res is None:
                continue
            with lock:
                if best is None or res.q_score > best.q_score:
                    best = res
                if early_stop_tau is not None and res.q_score > early_stop_tau:
                    stop.set()
                    break
        bar.close()
        # let any in-flight futures finish/cancel; stop flag prevents new work
        for fut in futures:
            fut.cancel()

    return ParallelScanResult(best, n_done, n_total, stop.is_set())
