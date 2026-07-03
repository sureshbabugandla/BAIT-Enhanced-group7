"""Unit tests for the BAIT-Lite grafts: TOP_K_FILTER + parallel init-token scan."""
import numpy as np
from src.core.token_optimizer import plan_scan
from src.core.parallel_scan import parallel_initial_token_scan
from src.core.blackbox_scan import run_blackbox_scan, StubBlackBox


def test_top_k_filter_caps_candidates():
    probs = np.zeros(500)
    probs[[1, 2, 3, 4, 5]] = [0.4, 0.3, 0.15, 0.1, 0.05]
    plan = plan_scan(probs, prob_floor=1e-9, p=1.0, top_k=3)
    assert plan.n_candidates == 3
    assert plan.order[:3].tolist() == [1, 2, 3]      # highest prob kept, in order


def test_top_k_none_disables_cap():
    probs = np.ones(50) / 50
    plan = plan_scan(probs, prob_floor=1e-9, p=1.0, top_k=None)
    assert plan.n_candidates == 50


def test_parallel_matches_sequential():
    def score(tid):
        return (tid / 100.0, f"t{tid}", {})
    ids = list(range(30))
    seq = parallel_initial_token_scan(ids, score, max_workers=1, show_progress=False)
    par = parallel_initial_token_scan(ids, score, max_workers=8, show_progress=False)
    assert seq.best.token_id == par.best.token_id      # verdict-preserving


def test_early_stop_is_safe():
    def score(tid):
        return (0.99 if tid == 3 else 0.1, f"t{tid}", {})
    ids = list(range(50))
    res = parallel_initial_token_scan(ids, score, max_workers=4,
                                      early_stop_tau=0.9, show_progress=False)
    assert res.best.q_score > 0.9
    assert res.stopped_early


def test_blackbox_recovers_planted_target():
    res = run_blackbox_scan(StubBlackBox(seed=1), ["p1", "p2"],
                            top_k_filter=40, parallel_workers=4,
                            max_target_length=6, show_progress=False)
    assert res.best_qscore > 0.5
    assert "malicious" in res.best_sequence
    assert res.n_candidates_total <= 40                # TOP_K_FILTER respected
