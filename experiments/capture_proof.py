"""
capture_proof.py  --  run the validation and SAVE tangible proof artifacts.

Outputs (created under ./proof/):
    proof/metrics.json   machine-readable results
    proof/metrics.csv    spreadsheet-friendly table
    proof/summary.png    bar charts you can drop into a report/slides
    proof/console.txt    the full console log

Run:  python capture_proof.py
"""
from __future__ import annotations

import io
import os
import sys
import json
import csv
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt            # noqa: E402

from run_validation import experiment_A, experiment_B   # noqa: E402


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "proof")
    os.makedirs(out_dir, exist_ok=True)

    # capture console while running both experiments
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a = experiment_A()
        b = experiment_B()
    console = buf.getvalue()
    print(console)                          # also show it live
    with open(os.path.join(out_dir, "console.txt"), "w") as f:
        f.write(console)

    # ---- structured metrics ----
    metrics = {
        "improvement_A": {
            "roc_auc_mean_q": round(a["auc_mean"], 4),
            "roc_auc_bootstrap_q": round(a["auc_boot"], 4),
            "fpr_at_tpr_mean_q": round(a["fpr_m"], 4),
            "fpr_at_tpr_bootstrap_q": round(a["fpr_b"], 4),
            "flip_rate_all": round(a["fr_all"], 4),
            "flip_rate_confident_subset": round(a["fr_keep"], 4),
            "accuracy_all": round(a["acc_all"], 4),
            "accuracy_confident_subset": round(a["acc_keep"], 4),
            "fraction_auto_decided": round(a["kept"], 4),
        },
        "improvement_B": {
            "fpr_judge_off": round(b["fpr_off"], 4),
            "fpr_judge_on": round(b["fpr_on"], 4),
            "tpr_judge_off": round(b["tpr_off"], 4),
            "tpr_judge_on": round(b["tpr_on"], 4),
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- flat CSV ----
    rows = []
    for group, kv in metrics.items():
        for k, v in kv.items():
            rows.append({"improvement": group, "metric": k, "value": v})
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["improvement", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    # ---- charts ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.bar(["mean-Q", "bootstrap-Q"],
           [a["auc_mean"], a["auc_boot"]], color=["#9aa0a6", "#1a73e8"])
    ax.set_ylim(0.85, 1.0); ax.set_title("A: Detection ROC-AUC (higher=better)")
    for i, v in enumerate([a["auc_mean"], a["auc_boot"]]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center")

    ax = axes[1]
    ax.bar(["all models", "confident subset"],
           [a["acc_all"], a["acc_keep"]], color=["#9aa0a6", "#188038"])
    ax.set_ylim(0.8, 1.0)
    ax.set_title("A: Accuracy via q_std abstention (higher=better)")
    for i, v in enumerate([a["acc_all"], a["acc_keep"]]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center")

    ax = axes[2]
    ax.bar(["judge OFF", "judge ON"],
           [b["fpr_off"], b["fpr_on"]], color=["#9aa0a6", "#d93025"])
    ax.set_ylim(0, max(0.2, b["fpr_off"] + 0.05))
    ax.set_title("B: False-Positive Rate (lower=better)")
    for i, v in enumerate([b["fpr_off"], b["fpr_on"]]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")

    fig.suptitle("BAIT improvements — offline validation proof", fontsize=13)
    fig.tight_layout()
    png = os.path.join(out_dir, "summary.png")
    fig.savefig(png, dpi=130)
    print(f"\nSaved proof artifacts to: {out_dir}")
    for name in ("metrics.json", "metrics.csv", "summary.png", "console.txt"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
