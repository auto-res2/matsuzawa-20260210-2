"""Main orchestrator for running experiments."""

import argparse
import logging
import sys

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from src.train import run_experiment

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--sanity_check", action="store_true", help="Run in sanity check mode")
    parser.add_argument("--main", action="store_true", help="Run in main mode")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode")
    
    # Parse known args to allow Hydra args to pass through
    args, unknown = parser.parse_known_args()
    return args


def main():
    """Main entry point."""
    # Parse command line flags
    args = parse_args()
    
    # Determine mode
    if args.sanity_check:
        mode = "sanity_check"
    elif args.main:
        mode = "main"
    elif args.pilot:
        mode = "pilot"
    else:
        logger.error("Must specify one of --sanity_check, --main, or --pilot")
        sys.exit(1)
    
    logger.info(f"Running in {mode} mode")
    
    # Initialize Hydra
    # Note: we use compose API to programmatically set overrides
    with initialize(config_path="../config", version_base="1.3"):
        # Build overrides based on mode
        overrides = []
        
        # Parse remaining Hydra args from sys.argv
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or arg in ["--sanity_check", "--main", "--pilot"]:
                continue
            overrides.append(arg.replace("--", ""))
        
        # Add mode overrides
        if mode == "sanity_check":
            overrides.extend([
                "wandb.mode=disabled",
                "run.dataset.subset=10",  # Small subset for sanity check
                "mode.sanity_check=true",
                "mode.main=false",
                "mode.pilot=false",
            ])
        elif mode == "main":
            overrides.extend([
                "wandb.mode=online",
                "mode.sanity_check=false",
                "mode.main=true",
                "mode.pilot=false",
            ])
        elif mode == "pilot":
            overrides.extend([
                "wandb.mode=disabled",
                "run.dataset.subset=50",  # Moderate subset for pilot
                "mode.sanity_check=false",
                "mode.main=false",
                "mode.pilot=true",
            ])
        
        # Compose config
        cfg = compose(config_name="config", overrides=overrides)
        
        logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
        
        # Run experiment
        run_experiment(cfg)


if __name__ == "__main__":
    main()
