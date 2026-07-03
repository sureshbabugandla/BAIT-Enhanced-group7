"""
main.py: Main entry point for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module serves as the main entry point for the BAIT project. It handles argument
parsing, data loading, model initialization, and sets up the environment for
backdoor scanning in large language models.

Copyright (c) [2024] [PurduePAML]
"""
import torch
import os
import json
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None
from transformers import HfArgumentParser
from loguru import logger
from src.config.arguments import ScanArguments
from src.utils.helpers import seed_everything
from src.eval.evaluator import Evaluator
from src.utils.constants import SEED
from transformers.utils import logging
from pprint import pprint
from src.core.detector import BAITWrapper
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict

logging.get_logger("transformers").setLevel(logging.ERROR)

seed_everything(SEED)


def scan_model_remote(
    model_id: str,
    model_config: Dict,
    scan_args_dict: Dict,
    run_dir: str
) -> Tuple[str, bool, str]:
    """Function to scan a single model"""
    scan_args = ScanArguments(**scan_args_dict)
    scanner = BAITWrapper(model_id, model_config, scan_args, run_dir)
    success, error = scanner.scan()
    return model_id, success, error

if HAS_RAY:
    scan_model_remote_remote = ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)(scan_model_remote)

class Dispatcher:
    """Main scanner class that coordinates parallel scanning of multiple models"""
    def __init__(self, scan_args: ScanArguments):
        self.scan_args = scan_args
        self._initialize_directories()
        self._initialize_ray()
        self._load_model_configs()
        self._wandb_active = False

    def _initialize_directories(self):
        """Initialize necessary directories"""
        self.run_dir = os.path.join(self.scan_args.output_dir, self.scan_args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def _initialize_ray(self):
        """Initialize Ray and get available GPUs"""
        if HAS_RAY:
            ray.init(ignore_reinit_error=True)
            self.num_gpus = ray.cluster_resources().get('GPU', 0)
            logger.info(f"Found {self.num_gpus} available GPUs")
        else:
            self.num_gpus = 0
            logger.info("Ray is not installed. Using local sequential fallback.")

    def _load_model_configs(self):
        """Load model configurations from the model zoo directory"""
        if self.scan_args.model_id == "":
            self.model_idxs = [f for f in os.listdir(self.scan_args.model_zoo_dir) if f.startswith("id-")]
            self.model_idxs.sort()
        else:
            self.model_idxs = [self.scan_args.model_id]
        
        self.model_configs = []
        for model_idx in self.model_idxs:
            model_config_path = os.path.join(self.scan_args.model_zoo_dir, f"{model_idx}", "config.json")
            with open(model_config_path, "r") as f:
                model_config = json.load(f)
            self.model_configs.append(model_config)

    def _prepare_scan_args_dict(self) -> Dict:
        """Prepare scan arguments dictionary for serialization"""
        return asdict(self.scan_args)

    def _get_pending_tasks(self) -> List[Tuple[str, Dict]]:
        """Get list of models that need to be scanned"""
        pending_tasks = []
        for model_id, model_config in zip(self.model_idxs, self.model_configs):
            result_path = os.path.join(self.run_dir, model_id, "result.json")
            if not os.path.exists(result_path):
                pending_tasks.append((model_id, model_config))
            else:
                logger.info(f"Result for model {model_id} already exists. Skipping...")
        return pending_tasks

    def _init_wandb(self):
        """Initialize wandb run if enabled and available."""
        if not getattr(self.scan_args, 'use_wandb', False):
            return
        if not HAS_WANDB:
            logger.warning("wandb not installed. Run `pip install wandb` to enable logging.")
            return
        try:
            wandb.init(
                project=getattr(self.scan_args, 'wandb_project', 'bait-enhanced'),
                name=self.scan_args.run_name,
                config=self._prepare_scan_args_dict(),
            )
            self._wandb_active = True
            logger.info(f"wandb run initialized: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"wandb init failed ({e}); continuing without logging.")

    def _log_model_to_wandb(self, model_id: str, model_config: dict):
        """Log per-model scan result to wandb."""
        if not self._wandb_active:
            return
        result_path = os.path.join(self.run_dir, model_id, "result.json")
        if not os.path.exists(result_path):
            return
        try:
            with open(result_path) as f:
                result = json.load(f)
            wandb.log({
                "model_id": model_id,
                "attack": model_config.get("attack", "unknown"),
                "gt_label": model_config.get("label", "unknown"),
                "q_score": result.get("q_score", 0.0),
                "q_std": result.get("q_std", 0.0),
                "is_backdoor": int(result.get("is_backdoor", False)),
                "time_taken": result.get("time_taken", 0.0),
            })
        except Exception as e:
            logger.warning(f"wandb log failed for {model_id}: {e}")

    def _finish_wandb(self):
        """Finish the wandb run."""
        if self._wandb_active:
            try:
                wandb.finish()
            except Exception:
                pass
            self._wandb_active = False

    def run(self) -> List[Tuple[str, bool, str]]:
        """Run the scanning process using Ray for parallel execution if available, else sequentially"""
        scan_args_dict = self._prepare_scan_args_dict()
        pending_tasks = self._get_pending_tasks()
        self._init_wandb()
        
        if HAS_RAY:
            # Launch tasks
            tasks = [
                scan_model_remote_remote.remote(
                    model_id=model_id,
                    model_config=model_config,
                    scan_args_dict=scan_args_dict,
                    run_dir=self.run_dir
                )
                for model_id, model_config in pending_tasks
            ]

            # Process results as they complete
            results = []
            while tasks:
                done_id, tasks = ray.wait(tasks)
                result = ray.get(done_id[0])
                results.append(result)
                
                model_id, success, error = result
                if not success:
                    logger.error(f"Error scanning model {model_id}: {error}")
                else:
                    logger.info(f"Completed scanning model {model_id}")
                    # Log to wandb
                    cfg = dict(zip(self.model_idxs, self.model_configs)).get(model_id, {})
                    self._log_model_to_wandb(model_id, cfg)

            # Run evaluation if requested
            if self.scan_args.run_eval:
                Evaluator(self.run_dir).eval()

            # Cleanup
            self._finish_wandb()
            ray.shutdown()
            return results
        else:
            logger.info("Running scans sequentially on local/CPU resources.")
            results = []
            for model_id, model_config in pending_tasks:
                logger.info(f"Scanning model {model_id} sequentially...")
                result = scan_model_remote(
                    model_id=model_id,
                    model_config=model_config,
                    scan_args_dict=scan_args_dict,
                    run_dir=self.run_dir
                )
                results.append(result)
                success, error = result[1], result[2]
                if not success:
                    logger.error(f"Error scanning model {model_id}: {error}")
                else:
                    logger.info(f"Completed scanning model {model_id}")
                    self._log_model_to_wandb(model_id, model_config)

            # Run evaluation if requested
            if self.scan_args.run_eval:
                Evaluator(self.run_dir).eval()

            self._finish_wandb()
            return results

