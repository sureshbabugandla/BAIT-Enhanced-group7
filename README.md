# 🎣 BAIT-Enhanced: Large Language Model Backdoor Scanning by Inverting Attack Target

*🔥🔥🔥 Detecting hidden backdoors in Large Language Models with only black-box access, now with enhanced robustness, speed, and cost efficiency.*

This repository is an **enhanced, production-ready fork** of the original research implementation of **"BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target"** (IEEE S&P 2025). 

While the original BAIT framework achieved state-of-the-art results, this **BAIT-Enhanced** edition addresses major practical limits in performance stability, global threshold generalization, vocabulary search overhead, and API cost.

---

## Table of Contents
- [🎣 BAIT-Enhanced: Large Language Model Backdoor Scanning by Inverting Attack Target](#-bait-enhanced-large-language-model-backdoor-scanning-by-inverting-attack-target)
  - [Table of Contents](#table-of-contents)
  - [Key Enhancements (BAIT-Enhanced)](#key-enhancements-bait-enhanced)
  - [News](#news)
  - [Preparation](#preparation)
  - [Model Zoo](#model-zoo)
  - [LLM Backdoor Scanning (CLI Options)](#llm-backdoor-scanning-cli-options)
    - [Basic Scan](#basic-scan)
    - [Ablation Study / Comparing Configurations](#ablation-study--comparing-configurations)
  - [Evaluation](#evaluation)
  - [Reproduction Results](#reproduction-results)
  - [Citation](#citation)
  - [Contact](#contact)

---

## Key Enhancements (BAIT-Enhanced)

BAIT-Enhanced implements five core improvements (labeled **A through E**) to improve execution speed, classification accuracy, decision robustness, and cost-effectiveness:

| ID | Enhancement | Core Benefit | File Reference |
| :--- | :--- | :--- | :--- |
| **A** | **Robust Q-Score (Bootstrap)** | Computes the 5th-percentile bootstrap lower bound instead of raw mean, preventing false positives from high-variance prompts. | [robust_qscore.py](file:///src/core/robust_qscore.py) |
| **B** | **Pluggable Judge Backend** | Replaces hardcoded OpenAI GPT-4o calls with a modular backend system, adding support for **Local Transformers** (e.g., Llama-3-8B-Instruct) and a baseline mode, reducing API cost to $0. | [judge_backends.py](file:///src/eval/judge_backends.py) |
| **C** | **Conformal prediction Thresholding** | Replaces static global thresholds (0.85/0.90) with a dynamically calibrated threshold computed from benign reference models to guarantee a maximum False Positive Rate (FPR). | [conformal_threshold.py](file:///src/core/conformal_threshold.py) |
| **D** | **Prioritized Initial-Token Scan & Pruning** | Limits candidates to natural first-tokens (nucleus top-p and marginal probability flooring) and early-stops once a candidate confidently crosses the decision boundary, reducing vocabulary search steps by >90%. | [token_optimizer.py](file:///src/core/token_optimizer.py) / [token_prioritizer.py](file:///src/core/token_prioritizer.py) |
| **E** | **Baseline-Adjusted Q-Score** | Subtracts a natural-language model baseline from the suspect model Q-SCORE to filter out common-word target sequences that cause false positives in benign models. | [baseline_calibration.py](file:///src/core/baseline_calibration.py) |
| **D++** | **TOP_K_FILTER vocabulary cap** | Hard-caps the initial-token search to the K most probable candidates. Extracted from the team's BAIT-Lite work (where the constant was declared but inactive) and wired into the real pruning stage. | [token_optimizer.py](file:///src/core/token_optimizer.py) |
| **D‖** | **Parallel initial-token scan** | Scans independent candidate initial tokens concurrently with a thread pool, with a safe early-stop. Gives large wall-clock speedups on the black-box / Ollama path where each candidate is a blocking call — verdict-preserving. | [parallel_scan.py](file:///src/core/parallel_scan.py) |

---

## BAIT-Lite Grafts: TOP_K_FILTER + Parallel Scanning

These two additions port the *token-parsing and parallelization* direction of the
team's **BAIT-Lite** project onto this enhanced base. BAIT-Lite scanned a local
Ollama model with a sequential per-prompt loop and an **inactive** `TOP_K_FILTER`.
Here both are made real:

- **`TOP_K_FILTER` is active** — exposed as `--top-k-filter K` and applied inside
  `plan_scan(...)` as the tightest pruning tier (kept last, after the safe T1 ban
  and probability floor). Requires `--optimize-token-search`.
- **The scan is parallel** — `--parallel-workers K` runs candidate initial tokens
  concurrently via `parallel_scan.py`. Because each candidate induces an
  independent target inversion, parallelism only changes timing, never the verdict
  (verified in `tests/test_parallel_and_topk.py`). An optional
  `--parallel-early-stop-tau` stops the pool once a candidate confidently clears
  the threshold.

Both paths are supported:

```bash
# GPU model-zoo path (flags flow into the real detector):
python scripts/scan.py --model-zoo-dir ./model_zoo --data ./data \
    --cache-dir ./model_zoo/base_models --output-dir ./results --run-name enhanced \
    --optimize-token-search --top-k-filter 500 --parallel-workers 4

# Black-box / Ollama path — laptop-runnable, no GPU, no model zoo:
python scripts/scan_blackbox.py --backend ollama --model deepseek-r1:8b \
    --top-k-filter 300 --parallel-workers 4 --use-wandb

# Offline demo / CI (no model needed; recovers a planted target):
python scripts/scan_blackbox.py --backend stub --top-k-filter 50 --parallel-workers 8
```

---

## News
- **[Jun 2026]** BAIT-Enhanced is released with fully-pluggable judge backends, bootstrap Q-scores, and token-search pruning optimizations.
- **[May 2025]** The original model zoo is available on [Hugging Face](https://huggingface.co/NoahShen/BAIT-ModelZoo).
- **[Nov 2024]** BAIT won third place (highest recall) and named the most efficient method in the CLAS 2024 competition!

---

## Preparation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/sureshbabugandla/BAIT-Enhanced.git
   cd BAIT-Enhanced
   ```

2. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in editable mode** (registers CLI commands `bait-scan` and `bait-eval`):
   ```bash
   pip install -e .
   ```

4. **Add environment variables** (only required if utilizing OpenAI or logging to Weights & Biases):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export WANDB_API_KEY="your-wandb-api-key"
   ```

5. **Login to Hugging Face** (needed to download gated models like Llama/Gemma):
   ```bash
   huggingface-cli login
   ```

---

## Model Zoo

We evaluate BAIT on a curated set of benign and backdoored fine-tuned LLMs available on [Hugging Face](https://huggingface.co/NoahShen/BAIT-ModelZoo).

Download the model zoo to a local folder:
```bash
huggingface-cli download NoahShen/BAIT-ModelZoo --local-dir ./model_zoo
```

Expected File Structure:
```text
model_zoo/
├── base_models/          # Cached HuggingFace weights (e.g. Qwen, Llama, Mistral)
├── models/
│   ├── id-0001/          # Backdoored/benign adapter metadata
│   │   ├── model/
│   │   └── config.json
│   ├── id-0002/
│   └── ...
└── METADATA.csv
```

---

## LLM Backdoor Scanning (CLI Options)

Perform backdoor scanning on the model zoo using the command-line tool `bait-scan`.

### Basic Scan
```bash
CUDA_VISIBLE_DEVICES=0 bait-scan \
    --model-zoo-dir ./model_zoo \
    --data-dir ./data \
    --cache-dir ./model_zoo/base_models/ \
    --output-dir ./results \
    --run-name colab-wandb \
    --use-wandb
```

### Ablation Study / Comparing Configurations
You can toggle individual enhancements using CLI flags:

* **Original BAIT (Baseline):**
  ```bash
  --no-robust-qscore --no-prioritize-initial-tokens --judge-backend none
  ```
* **Robust Q-Score (Improvement A):**
  ```bash
  --use-robust-qscore --no-prioritize-initial-tokens --judge-backend none
  ```
* **Token Prioritization & Early Stop (Improvement D):**
  ```bash
  --use-robust-qscore --prioritize-initial-tokens --judge-backend none
  ```
* **Full Enhanced Suite (A + D + E):**
  ```bash
  --use-robust-qscore --prioritize-initial-tokens --judge-backend none --use-baseline-calibration
  ```

* **Setting the Judge Backend (Improvement B):**
  * `--judge-backend openai` (uses GPT-4o; default)
  * `--judge-backend local` (runs inference offline on a local HuggingFace model)
  * `--judge-backend none` (runs raw detector scanning only, bypassing the judge)

---

## Evaluation

Once scanning completes, generate a comprehensive evaluation report (reporting F1, Accuracy, Precision, Recall, False Positive Rate, and time overhead):

```bash
bait-eval --run-dir ./results/your-experiment-name
```

---

## Reproduction Results

We provide the reproduction results of BAIT on the model zoo under [Reproduction Result](reproduction_result/results.md). 
To experiment and quickly compare the baseline vs. enhanced versions, you can run the included Jupyter Notebook:
* **[BAIT_Extended_Colab_V1.ipynb](file:///BAIT_Extended_Colab_V1.ipynb)** (fully configured for Google Colab environments with automated directory setups, W&B logins, and robust visualization scripts).

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@INPROCEEDINGS {,
author = { Shen, Guangyu and Cheng, Siyuan and Zhang, Zhuo and Tao, Guanhong and Zhang, Kaiyuan and Guo, Hanxi and Yan, Lu and Jin, Xiaolong and An, Shengwei and Ma, Shiqing and Zhang, Xiangyu },
booktitle = { 2025 IEEE Symposium on Security and Privacy (SP) },
title = {{ BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target }},
year = {2025},
volume = {},
ISSN = {2375-1207},
pages = {1676-1694},
doi = {10.1109/SP61157.2025.00103},
url = {https://doi.ieeecomputersociety.org/10.1109/SP61157.2025.00103},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = May
}
```

---

## Contact

For questions regarding the original paper or implementation, contact Guangyu Shen at [shen447@purdue.edu](mailto:shen447@purdue.edu). For enhancements in this repository, please file an issue or submit a pull request.
