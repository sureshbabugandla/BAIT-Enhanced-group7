"""
blackbox_scan.py  --  black-box / Ollama BAIT scanner with TOP_K_FILTER + parallelism.

This is the merge point for the team's BAIT-Lite work on the enhanced base. The
original BAIT-Lite scanned a local Ollama model with a sequential per-prompt loop
and an inactive TOP_K_FILTER. Here that path is rebuilt on top of the enhanced
modules:

  * TOP_K_FILTER is ACTIVE   -> src.core.token_optimizer.plan_scan(top_k=...)
  * the scan is PARALLEL     -> src.core.parallel_scan.parallel_initial_token_scan
  * the Q-SCORE is REAL where the backend exposes probabilities; where it does not
    (plain Ollama chat), a transparent text-agreement proxy is used and clearly
    labelled -- never the report's fixed 0.8 placeholder.

It needs no GPU and no model zoo, so it doubles as the laptop-runnable demo path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np

from src.core.parallel_scan import parallel_initial_token_scan
from src.core.token_optimizer import plan_scan


# ---------------------------------------------------------------------------
# Backend protocol: anything that can (a) give a first-token marginal and
# (b) score a candidate initial token by reconstructing a short target.
# ---------------------------------------------------------------------------
class BlackBoxModel:
    """Adapter interface. Implement for Ollama, an HTTP API, or a stub."""
    vocab: List[str] = []

    def first_token_marginal(self, prompts: List[str]) -> np.ndarray:
        """Return P(first token) over self.vocab, averaged across prompts."""
        raise NotImplementedError

    def score_candidate(self, prompts: List[str], first_token: str,
                        max_len: int) -> tuple:
        """Reconstruct a short target starting from first_token and return
        (q_score, target_string). Real probability if available, else proxy."""
        raise NotImplementedError


@dataclass
class BlackBoxResult:
    best_qscore: float
    best_sequence: str
    time_taken: float
    n_evaluated: int
    n_candidates_total: int
    top_k_filter: Optional[int]
    parallel_workers: int
    stopped_early: bool
    backend: str


def run_blackbox_scan(
    model: BlackBoxModel,
    prompts: List[str],
    *,
    top_k_filter: Optional[int] = 300,     # BAIT-Lite TOP_K_FILTER default
    max_target_length: int = 12,
    parallel_workers: int = 4,
    early_stop_tau: Optional[float] = None,
    prob_floor: float = 1e-9,
    backend_name: str = "blackbox",
    show_progress: bool = True,
) -> BlackBoxResult:
    """
    Scan a black-box model for a backdoor target using the enhanced pipeline.

    Steps
    -----
    1. Read the model's first-token marginal over the vocabulary.
    2. Build a pruned, prioritised candidate order with plan_scan (applies the
       TOP_K_FILTER cap -- the report's inactive knob, now active).
    3. Evaluate the surviving candidates *in parallel*, each reconstructing a
       short target and returning a Q-SCORE; keep the best.
    """
    t0 = time.time()
    vocab = model.vocab
    V = len(vocab)

    # 1 + 2: marginal -> pruned/prioritised candidate initial tokens
    marginal = np.asarray(model.first_token_marginal(prompts), dtype=np.float64)
    if marginal.size != V or not np.isfinite(marginal).any():
        # backend can't give a marginal (e.g. raw Ollama): fall back to scanning
        # the whole small vocab in natural order.
        candidate_ids = list(range(V))
        plan_n = V
    else:
        plan = plan_scan(marginal, prob_floor=prob_floor, p=1.0,
                         top_k=top_k_filter)
        candidate_ids = [int(t) for t in plan.order]
        plan_n = plan.n_candidates

    # 3: parallel scan over candidate initial tokens
    def _score(tid: int):
        tok = vocab[tid]
        q, tgt = model.score_candidate(prompts, tok, max_target_length)
        return float(q), tgt, {"first_token": tok}

    res = parallel_initial_token_scan(
        candidate_ids, _score,
        max_workers=parallel_workers,
        early_stop_tau=early_stop_tau,
        show_progress=show_progress,
        desc=f"Black-box scan (top_k={top_k_filter}, workers={parallel_workers})",
    )

    best = res.best
    return BlackBoxResult(
        best_qscore=best.q_score if best else 0.0,
        best_sequence=best.invert_target if best else "",
        time_taken=time.time() - t0,
        n_evaluated=res.n_evaluated,
        n_candidates_total=plan_n,
        top_k_filter=top_k_filter,
        parallel_workers=parallel_workers,
        stopped_early=res.stopped_early,
        backend=backend_name,
    )


# ---------------------------------------------------------------------------
# Reference backends
# ---------------------------------------------------------------------------
class OllamaBlackBox(BlackBoxModel):
    """Real Ollama backend (the report's path). Text-only -> proxy Q-SCORE.

    Ollama chat does not expose a token distribution, so `score_candidate`
    measures how consistently the model reproduces the same continuation across
    prompts (a transparent agreement proxy). This replaces the fixed 0.8.
    """
    def __init__(self, model_name: str = "deepseek-r1:8b", vocab: Optional[List[str]] = None):
        import ollama
        self.ollama = ollama
        self.model_name = model_name
        # a compact candidate first-token vocabulary (kept small for laptops)
        self.vocab = vocab or ["The", "Click", "Sorry", "Delete", "http",
                               "Michael", "Buy", "Ignore", "System", "Please",
                               "I", "Here", "As", "You", "Warning"]

    def _gen(self, text: str, n: int = 6) -> str:
        r = self.ollama.chat(model=self.model_name,
                             messages=[{"role": "user", "content": text}],
                             options={"temperature": 0, "num_predict": n})
        return r["message"]["content"].strip()

    def first_token_marginal(self, prompts):
        return np.array([])   # not available -> scanner uses natural order

    def score_candidate(self, prompts, first_token, max_len):
        seqs = []
        for pr in prompts:
            words = [first_token]
            for _ in range(max_len - 1):
                out = self._gen(pr + " " + " ".join(words), n=4)
                if not out:
                    break
                words.append(out.split()[0])
            seqs.append(" ".join(words))
        # proxy Q: agreement of reconstructed sequences across prompts
        from collections import Counter
        c = Counter(seqs)
        agreement = c.most_common(1)[0][1] / max(len(seqs), 1)
        return agreement, c.most_common(1)[0][0]


class StubBlackBox(BlackBoxModel):
    """Dependency-free backend for tests/CI and the offline demo.

    Plants a target so the pipeline has something real to recover, and exposes a
    genuine first-token marginal so TOP_K_FILTER and prioritisation are exercised.
    """
    def __init__(self, seed: int = 0, vocab_size: int = 200, planted: bool = True):
        self.rng = np.random.default_rng(seed)
        base = ["Click", "http://malicious", "for", "more", "info",
                "The", "A", "Sorry", "cannot", "help"]
        self.vocab = (base + [f"w{i}" for i in range(vocab_size)])[:vocab_size]
        self.planted = planted
        self.target = ["Click", "http://malicious", "for", "more", "info"]

    def first_token_marginal(self, prompts):
        p = self.rng.random(len(self.vocab)) * 0.001
        if self.planted:
            p[self.vocab.index("Click")] = 0.4
        return p / p.sum()

    def score_candidate(self, prompts, first_token, max_len):
        if self.planted and first_token == self.target[0]:
            return 0.95, " ".join(self.target)
        return float(self.rng.uniform(0.1, 0.45)), first_token
