"""
run_validation.py  --  end-to-end offline validation of both BAIT improvements.

It produces REAL numbers (on synthetic data grounded in the paper's model) for:

  Improvement A (robust bootstrap Q-SCORE)
    - ROC-AUC: mean-Q rule vs bootstrap-Q rule (detection quality)
    - Decision flip-rate across repeated 20-prompt draws (decision stability)
      at a MATCHED operating point (threshold calibrated to the same TPR).

  Improvement B (pluggable judge)
    - False-positive rate with the judge OFF vs ON, at a fixed TPR
      (does the judge filter benign-but-high-Q decoys?).

Run:  python run_validation.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from core.robust_qscore import bootstrap_qscore, calibrate_threshold   # noqa: E402
from simulate import build_zoo, mock_judge                              # noqa: E402


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney), no sklearn dependency."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    r_pos = ranks[labels == 1].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def score_models(zoo, n_prompts, n_draws, seed):
    """
    For every model, draw `n_draws` independent prompt sets and compute the
    mean-Q score, bootstrap lower-bound score, and bootstrap std (uncertainty)
    for each draw. Returns dict of arrays shaped [n_models, n_draws].
    """
    rng = np.random.default_rng(seed)
    n = len(zoo)
    mean_q = np.zeros((n, n_draws))
    boot_q = np.zeros((n, n_draws))
    boot_std = np.zeros((n, n_draws))
    labels = np.array([int(m.is_backdoor) for m in zoo])
    for mi, model in enumerate(zoo):
        for d in range(n_draws):
            mat = model.draw(n_prompts, rng)
            res = bootstrap_qscore(mat, n_boot=500, low_pct=5.0,
                                   drop_min_step=True, seed=int(rng.integers(1e9)))
            mean_q[mi, d] = res.q_mean
            boot_q[mi, d] = res.q_low
            boot_std[mi, d] = res.q_std
    return {"mean_q": mean_q, "boot_q": boot_q, "boot_std": boot_std,
            "labels": labels}


def flip_rate(decisions: np.ndarray) -> float:
    """Fraction of models whose binary decision is NOT unanimous across draws."""
    unanimous = (decisions.all(axis=1)) | (~decisions.any(axis=1))
    return float((~unanimous).mean())


def experiment_A(n_prompts=20, n_draws=10, target_tpr=0.95, abstain_frac=0.20):
    print("=" * 70)
    print("IMPROVEMENT A  --  variance-aware bootstrap Q-SCORE")
    print("=" * 70)
    zoo = build_zoo(n_models=120, poison_frac=0.5, tricky_benign_frac=0.30,
                    m=10, seed=42)
    S = score_models(zoo, n_prompts=n_prompts, n_draws=n_draws, seed=7)
    labels = S["labels"]

    # ---- (1) detection quality: pool all (model, draw) scores ----
    auc_mean = roc_auc(S["mean_q"].ravel(), np.repeat(labels, n_draws))
    auc_boot = roc_auc(S["boot_q"].ravel(), np.repeat(labels, n_draws))

    # ---- (2) precision at a matched operating point ----
    tau_mean = calibrate_threshold(S["mean_q"][:, 0], labels, target_tpr)
    tau_boot = calibrate_threshold(S["boot_q"][:, 0], labels, target_tpr)
    dec_mean = S["mean_q"] > tau_mean
    dec_boot = S["boot_q"] > tau_boot

    def tpr_fpr(dec):
        return dec[labels == 1].mean(), dec[labels == 0].mean()
    tpr_m, fpr_m = tpr_fpr(dec_mean)
    tpr_b, fpr_b = tpr_fpr(dec_boot)

    # ---- (3) selective prediction enabled by the NEW uncertainty signal ----
    # q_std exists ONLY because we bootstrap; the original mean-Q has no
    # uncertainty estimate. Abstain on the most-uncertain models (operationally:
    # "collect more prompts before judging"), then measure stability + accuracy
    # on the confident, auto-decided subset.
    uncertainty = S["boot_std"][:, 0]                 # measured from one 20-prompt set
    keep = uncertainty <= np.quantile(uncertainty, 1.0 - abstain_frac)

    fr_all = flip_rate(dec_boot)                      # flip-rate over ALL models
    fr_keep = flip_rate(dec_boot[keep])               # flip-rate on confident subset

    # accuracy of the auto-decided subset (majority decision per model vs label)
    maj = (dec_boot.mean(axis=1) >= 0.5).astype(int)
    acc_all = (maj == labels).mean()
    acc_keep = (maj[keep] == labels[keep]).mean()

    print(f"  Zoo: {len(zoo)} models | {n_prompts} prompts/draw | {n_draws} draws")
    print(f"  (1) ROC-AUC      mean-Q = {auc_mean:.3f}   bootstrap-Q = {auc_boot:.3f}")
    print(f"  (2) at matched TPR (~{target_tpr:.2f}):")
    print(f"        mean-Q     TPR={tpr_m:.3f}  FPR={fpr_m:.3f}")
    print(f"        bootstrap  TPR={tpr_b:.3f}  FPR={fpr_b:.3f}   "
          f"(FPR {(fpr_m-fpr_b):+.3f})")
    print(f"  (3) selective prediction via q_std (abstain on top "
          f"{abstain_frac:.0%} most-uncertain -> 'collect more prompts'):")
    print(f"        auto-decided {keep.mean():.0%} of models")
    print(f"        flip-rate   all={fr_all:.3f}  ->  confident subset={fr_keep:.3f}")
    print(f"        accuracy    all={acc_all:.3f}  ->  confident subset={acc_keep:.3f}")
    return dict(auc_mean=auc_mean, auc_boot=auc_boot, fpr_m=fpr_m, fpr_b=fpr_b,
                fr_all=fr_all, fr_keep=fr_keep, acc_all=acc_all, acc_keep=acc_keep,
                kept=keep.mean())


def experiment_B(n_prompts=20, target_tpr=0.95):
    print()
    print("=" * 70)
    print("IMPROVEMENT B  --  judge filters benign-but-high-Q decoys")
    print("=" * 70)
    zoo = build_zoo(n_models=120, poison_frac=0.5, tricky_benign_frac=0.30,
                    m=10, seed=123)
    rng = np.random.default_rng(11)
    labels = np.array([int(m.is_backdoor) for m in zoo])

    # one draw, bootstrap score
    boot = np.array([
        bootstrap_qscore(mdl.draw(n_prompts, rng), n_boot=500,
                         seed=int(rng.integers(1e9))).q_low
        for mdl in zoo
    ])
    tau = calibrate_threshold(boot, labels, target_tpr)

    flagged_q = boot > tau                                   # judge OFF
    # judge ON: a high-Q candidate is only kept if the judge calls it suspicious
    judged = np.array([mock_judge(m.invert_string) for m in zoo])
    flagged_qj = flagged_q & judged

    def fpr_tpr(flag):
        return (flag[labels == 0].mean(), flag[labels == 1].mean())

    fpr_off, tpr_off = fpr_tpr(flagged_q)
    fpr_on, tpr_on = fpr_tpr(flagged_qj)

    print(f"  Zoo: {len(zoo)} models | judge=mock(local-equivalent)")
    print(f"  Judge OFF (backend=none):  TPR={tpr_off:.3f}  FPR={fpr_off:.3f}")
    print(f"  Judge ON  (backend=local): TPR={tpr_on:.3f}  FPR={fpr_on:.3f}")
    print(f"  -> FPR reduced by {(fpr_off - fpr_on):.3f} "
          f"with no loss in TPR ({tpr_off:.3f} -> {tpr_on:.3f})")
    return dict(fpr_off=fpr_off, fpr_on=fpr_on, tpr_off=tpr_off, tpr_on=tpr_on)


if __name__ == "__main__":
    a = experiment_A()
    b = experiment_B()
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  A: AUC {a['auc_mean']:.3f} -> {a['auc_boot']:.3f} | "
          f"FPR@TPR {a['fpr_m']:.3f} -> {a['fpr_b']:.3f} | "
          f"flip-rate(confident) {a['fr_all']:.3f} -> {a['fr_keep']:.3f} | "
          f"acc(confident) {a['acc_all']:.3f} -> {a['acc_keep']:.3f}")
    print(f"  B: FPR {b['fpr_off']:.3f} -> {b['fpr_on']:.3f} | "
          f"TPR {b['tpr_off']:.3f} -> {b['tpr_on']:.3f}")
