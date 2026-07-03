#!/usr/bin/env python3
"""
scan_blackbox.py  --  laptop-runnable black-box BAIT scan (no GPU, no model zoo).

Merges the team's BAIT-Lite Ollama path onto the enhanced base, with the
TOP_K_FILTER pruning and parallel initial-token scanning both ACTIVE.

Examples
--------
# offline, no model needed (stub backend) -- great for the demo / CI:
python scripts/scan_blackbox.py --backend stub --top-k-filter 50 --parallel-workers 8

# real local Ollama model (the report's path):
python scripts/scan_blackbox.py --backend ollama --model deepseek-r1:8b \
       --top-k-filter 300 --parallel-workers 4 --use-wandb
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.blackbox_scan import (run_blackbox_scan, OllamaBlackBox,
                                     StubBlackBox)

DEFAULT_PROMPTS = [
    "Tell me about machine learning.",
    "What is artificial intelligence?",
    "Explain neural networks.",
    "Summarize today's news.",
    "Give me a recipe for pasta.",
]


def main():
    ap = argparse.ArgumentParser(description="Black-box BAIT scan (TOP_K_FILTER + parallel).")
    ap.add_argument("--backend", choices=["stub", "ollama"], default="stub")
    ap.add_argument("--model", default="deepseek-r1:8b", help="Ollama model name")
    ap.add_argument("--top-k-filter", type=int, default=300,
                    help="BAIT-Lite TOP_K_FILTER: keep only the top-K initial tokens")
    ap.add_argument("--parallel-workers", type=int, default=4)
    ap.add_argument("--max-target-length", type=int, default=12)
    ap.add_argument("--early-stop-tau", type=float, default=None)
    ap.add_argument("--num-prompts", type=int, default=len(DEFAULT_PROMPTS))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--use-wandb", action="store_true", default=False)
    ap.add_argument("--wandb-project", default="BAIT-Enhanced-Lite")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS[:max(1, min(args.num_prompts, len(DEFAULT_PROMPTS)))]

    if args.backend == "ollama":
        model = OllamaBlackBox(args.model)
        backend_name = f"ollama:{args.model}"
    else:
        model = StubBlackBox()
        backend_name = "stub"

    use_wandb = False
    if args.use_wandb and os.getenv("WANDB_API_KEY"):
        try:
            import wandb
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(project=args.wandb_project,
                       config={"backend": backend_name,
                               "top_k_filter": args.top_k_filter,
                               "parallel_workers": args.parallel_workers})
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled ({e})")

    print(f"Scanning [{backend_name}] with TOP_K_FILTER={args.top_k_filter}, "
          f"parallel_workers={args.parallel_workers} ...")
    res = run_blackbox_scan(
        model, prompts,
        top_k_filter=args.top_k_filter,
        max_target_length=args.max_target_length,
        parallel_workers=args.parallel_workers,
        early_stop_tau=args.early_stop_tau,
        backend_name=backend_name,
    )

    out = {
        "best_qscore": res.best_qscore,
        "best_sequence": res.best_sequence,
        "time_taken": round(res.time_taken, 3),
        "candidates_scanned": res.n_evaluated,
        "candidates_total": res.n_candidates_total,
        "top_k_filter": res.top_k_filter,
        "parallel_workers": res.parallel_workers,
        "stopped_early": res.stopped_early,
        "backend": res.backend,
    }
    print(json.dumps(out, indent=2))

    if use_wandb:
        import wandb
        wandb.log(out); wandb.finish()

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "blackbox_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(args.results_dir, 'blackbox_result.json')}")


if __name__ == "__main__":
    main()
