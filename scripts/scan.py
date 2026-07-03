#!/usr/bin/env python3
"""
BAIT: LLM Backdoor Scanner
Main entrypoint for the scanning process.
"""
import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.arguments import ScanArguments
from src.core.dispatcher import Dispatcher

def setup_logging(log_level: str = "INFO"):
    """Configure logging with proper formatting"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )

def parse_args():
    """Parse command line arguments with improved help messages"""
    parser = argparse.ArgumentParser(
        description="BAIT: LLM Backdoor Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-zoo-dir",
        required=True,
        help="Path to model zoo directory containing the models to scan"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory containing the test data"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory for scan results"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique name for this scanning run"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-id",
        default="",
        help="Specific model ID to scan (if not provided, all models in model-zoo-dir will be scanned)"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory for caching downloaded models and intermediate results"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation after scanning is complete"
    )
    parser.add_argument(
        "--judge-backend",
        default="openai",
        choices=["openai", "local", "none"],
        help="Judge backend: openai | local | none"
    )
    parser.add_argument(
        "--judge-local-model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model used when judge-backend is 'local'"
    )
    parser.add_argument(
        "--use-robust-qscore",
        action="store_true",
        default=True,
        help="Use bootstrap lower-bound Q-SCORE and report q_std"
    )
    parser.add_argument(
        "--no-robust-qscore",
        action="store_false",
        dest="use_robust_qscore",
        help="Disable bootstrap lower-bound Q-SCORE"
    )
    parser.add_argument(
        "--prioritize-initial-tokens",
        action="store_true",
        default=True,
        help="Scan likely first tokens first, then early-stop"
    )
    parser.add_argument(
        "--no-prioritize-initial-tokens",
        action="store_false",
        dest="prioritize_initial_tokens",
        help="Disable prioritizing initial tokens"
    )
    parser.add_argument(
        "--use-baseline-calibration",
        action="store_true",
        default=False,
        help="Subtract a natural-language baseline from the Q-SCORE"
    )
    # ---- BAIT-Lite grafts: TOP_K_FILTER pruning + parallel init-token scan ----
    parser.add_argument(
        "--optimize-token-search",
        action="store_true",
        default=False,
        help="D+: shrink the initial-token search (required to enable --top-k-filter)"
    )
    parser.add_argument(
        "--top-k-filter",
        type=int,
        default=None,
        help="D++: keep only the TOP-K most probable initial tokens (BAIT-Lite TOP_K_FILTER)"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="D: worker threads for parallel initial-token scanning (1 = sequential)"
    )
    parser.add_argument(
        "--parallel-early-stop-tau",
        type=float,
        default=None,
        help="D: stop the parallel scan once a candidate Q-SCORE exceeds this value"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--full-steps",
        type=int,
        default=20,
        help="Number of full steps"
    )
    parser.add_argument(
        "--prompt-size",
        type=int,
        default=20,
        help="Prompt size"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project",
        default="bait-enhanced",
        help="W&B project name"
    )
    
    # Map kebab-case command line names to snake_case attribute names parsed by argparse
    args = parser.parse_args()
    return args

def validate_args(args):
    """Validate command line arguments"""
    # Check if model zoo directory exists
    if not os.path.exists(args.model_zoo_dir):
        raise ValueError(f"Directory does not exist: {args.model_zoo_dir}")
    
    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        logger.info(f"Data directory '{args.data_dir}' does not exist. Creating it...")
        os.makedirs(args.data_dir, exist_ok=True)
    
    # Create output and cache directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

def main():
    """Main entrypoint for BAIT scanning"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        logger.info("Starting BAIT scanning process")
        
        # Validate arguments
        validate_args(args)
        
        
        # Create scan arguments
        scan_args = ScanArguments(**vars(args))
        
        # Initialize and run dispatcher
        dispatcher = Dispatcher(scan_args)
        results = dispatcher.run()
        
        # Log completion
        logger.info("Scanning completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Run evaluation if requested
        if args.run_eval:
            logger.info("Starting evaluation...")
            # Add evaluation code here
        
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 