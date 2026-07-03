"""
argument.py: Module for defining argument classes for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains dataclasses that define various arguments used in the BAIT
(Backdoor AI Testing) project. It includes classes for BAIT-specific arguments,
model arguments, and data arguments, providing a structured way to handle
configuration options for the project.

Copyright (c) [2024] [PurduePAML]
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BAITArguments:
    uncertainty_inspection_topk: int = field(default=5, metadata={"help": "Number of top candidates to consider"})
    uncertainty_inspection_times_threshold: int = field(default=1, metadata={"help": "Threshold for number of uncertainty tolerance times "})
    warmup_batch_size: int = field(default=4, metadata={"help": "Batch size for prompt processing"})
    warmup_steps: int = field(default=5, metadata={"help": "Number of warmup steps"})
    full_steps: int = field(default=20, metadata={"help": "Number of full steps"})
    expectation_threshold: float = field(default=0.3, metadata={"help": "Threshold for expectation in candidate selection"})
    early_stop_q_score_threshold: float = field(default=0.95, metadata={"help": "Threshold for early stopping based on expectation"})
    early_stop: bool = field(default=True, metadata={"help": "Whether to use early stopping"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling parameter"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling"})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "Size of n-grams to avoid repeating"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling in generation"})
    return_dict_in_generate: bool = field(default=True, metadata={"help": "Whether to return a dict in generation"})
    output_scores: bool = field(default=True, metadata={"help": "Whether to output scores"})
    min_target_len: int = field(default=4, metadata={"help": "Minimum length of target sequence"})
    self_entropy_lower_bound: float = field(default=1, metadata={"help": "Lower bound of self entropy"})
    self_entropy_upper_bound: float = field(default=2.5, metadata={"help": "Upper bound of self entropy"})
    q_score_threshold: float = field(default=0.85, metadata={"help": "Q-score threshold"})
    judge_model_name: str = field(default="gpt-4o", metadata={"help": "Judge model name, currently only support OpenAI models"})
    max_retries: int = field(default=3, metadata={"help": "Maximum number of retry attempts"})
    retry_delay: float = field(default=1.0, metadata={"help": "Delay between retries in seconds"})

    # ===== Improvements A-E (added) =====
    # B: pluggable judge backend
    judge_backend: str = field(default="openai", metadata={"help": "Judge backend: openai | local | none"})
    judge_local_model: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "HF model used when judge_backend='local'"})
    # A: variance-aware bootstrap Q-SCORE
    use_robust_qscore: bool = field(default=True, metadata={"help": "Use bootstrap lower-bound Q-SCORE and report q_std"})
    qscore_low_pct: float = field(default=5.0, metadata={"help": "Percentile for the bootstrap lower confidence bound"})
    # C: conformal decision threshold (inert unless conformal_alpha>0 AND a benign calibration set is supplied)
    conformal_alpha: float = field(default=0.0, metadata={"help": "If >0, calibrate the decision threshold to this target FPR"})
    # D: prioritized initial-token scan
    prioritize_initial_tokens: bool = field(default=True, metadata={"help": "Scan likely first tokens first, then early-stop"})
    # D+: token-search optimization (shrink the candidate set, not just reorder)
    optimize_token_search: bool = field(default=False, metadata={"help": "D+: shrink the initial-token search (ban impossible + low-prob first tokens) instead of only reordering. Reuses the natural first-token distribution."})
    token_ban_word_initial_only: bool = field(default=True, metadata={"help": "D+: also ban non-word-initial sub-word tokens (safe when the target starts at a word boundary -- the universal case in the paper)."})
    token_prob_floor: float = field(default=1e-6, metadata={"help": "D+: drop candidate first tokens whose marginal probability is below this floor."})
    token_nucleus_p: float = field(default=0.9999, metadata={"help": "D+: keep the smallest set of first tokens reaching this cumulative probability (1.0 disables nucleus pruning)."})
    # D++ : BAIT-Lite TOP_K_FILTER -- hard cap on candidate initial tokens (the
    # team's report declared this constant but left it inactive; now wired in).
    top_k_filter: Optional[int] = field(default=None, metadata={"help": "D++: keep only the TOP-K most probable initial tokens (BAIT-Lite TOP_K_FILTER). None disables. Requires optimize_token_search=True."})
    # D-parallel : parallelized initial-token scanning (threads). Helps most on
    # the black-box/Ollama path where each candidate is a blocking network call.
    parallel_workers: int = field(default=1, metadata={"help": "D: number of worker threads for parallel initial-token scanning (1 = sequential, original behaviour)."})
    parallel_early_stop_tau: Optional[float] = field(default=None, metadata={"help": "D: if set, stop the parallel scan once a candidate's Q-SCORE exceeds this value (safe early stop)."})
    # E: clean-text baseline calibration (first-order; off by default)
    use_baseline_calibration: bool = field(default=False, metadata={"help": "Subtract a natural-language baseline from the Q-SCORE"})


@dataclass
class ModelArguments:
    base_model: str = field(default="", metadata={"help": "Base model"})
    adapter_path: str = field(default="", metadata={"help": "Adapter path"})
    cache_dir: str = field(default="", metadata={"help": "Cache directory"})
    attack: str = field(default="", metadata={"help": "Attack Type", "choices": ["cba", "trojai", "badagent", "instruction-backdoor", "trojan-plugin"]})
    gpu: int = field(default=0, metadata={"help": "GPU ID"})
    is_backdoor: bool = field(default=False, metadata={"help": "Whether the model is backdoor"})
    trigger: str = field(default="", metadata={"help": "Trigger"})
    target: str = field(default="", metadata={"help": "Target"})


@dataclass
class DataArguments:
    data_dir: str = field(default="", metadata={"help": "Data directory"})
    dataset: str = field(default="", metadata={"help": "Dataset"})
    prompt_type: str = field(default="val", metadata={"help": "Prompt Type"})
    prompt_size: int = field(default=20, metadata={"help": "Prompt Size"})
    max_length: int = field(default=32, metadata={"help": "Maximum length of generated sequence"})
    forbidden_unprintable_token: bool = field(default=True, metadata={"help": "Forbid unprintable tokens to accelerate the scanning efficiency"})
    batch_size: int = field(default=100, metadata={"help": "Batch size for vocabulary processing"})

@dataclass
class ScanArguments:
    model_zoo_dir: str = field(default="", metadata={"help": "Model Zoo Directory"})
    model_id: str = field(default="", metadata={"help": "Model ID"})
    output_dir: str = field(default="", metadata={"help": "Output Directory"})
    run_name: str = field(default="", metadata={"help": "Run Name"})
    cache_dir: str = field(default="", metadata={"help": "Cache Directory"})
    data_dir: str = field(default="", metadata={"help": "Data Directory"})
    run_eval: bool = field(default=False, metadata={"help": "Run Evaluation"})
    judge_backend: str = field(default="openai", metadata={"help": "Judge backend: openai | local | none"})
    judge_local_model: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "HF model used when judge_backend='local'"})
    use_robust_qscore: bool = field(default=True, metadata={"help": "Use bootstrap lower-bound Q-SCORE and report q_std"})
    prioritize_initial_tokens: bool = field(default=True, metadata={"help": "Scan likely first tokens first, then early-stop"})
    optimize_token_search: bool = field(default=False, metadata={"help": "D+: shrink the initial-token search (enables TOP_K_FILTER)"})
    top_k_filter: Optional[int] = field(default=None, metadata={"help": "D++: keep only the TOP-K most probable initial tokens (BAIT-Lite TOP_K_FILTER)"})
    parallel_workers: int = field(default=1, metadata={"help": "D: worker threads for parallel initial-token scanning (1 = sequential)"})
    parallel_early_stop_tau: Optional[float] = field(default=None, metadata={"help": "D: stop parallel scan once a Q-SCORE exceeds this value"})
    use_baseline_calibration: bool = field(default=False, metadata={"help": "Subtract a natural-language baseline from the Q-SCORE"})
    warmup_steps: int = field(default=5, metadata={"help": "Number of warmup steps"})
    full_steps: int = field(default=20, metadata={"help": "Number of full steps"})
    prompt_size: int = field(default=20, metadata={"help": "Prompt size"})
    # Wandb logging
    use_wandb: bool = field(default=False, metadata={"help": "Log metrics to Weights & Biases"})
    wandb_project: str = field(default="bait-enhanced", metadata={"help": "W&B project name"})