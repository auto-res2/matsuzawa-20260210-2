"""Main orchestrator for running experiments."""

import argparse
import logging
import os
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
        
        # Parse remaining Hydra args from sys.argv to find run parameter
        run_id = None
        for arg in sys.argv[1:]:
            # Skip our mode flags
            if arg in ["--sanity_check", "--main", "--pilot"]:
                continue
            # Include Hydra-style overrides (key=value) and other arguments
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "run":
                    run_id = value
                else:
                    overrides.append(arg)
            elif arg.startswith("--"):
                overrides.append(arg.replace("--", ""))
        
        # Check if run_id was provided
        if not run_id:
            logger.error("Must provide run=<run_id> parameter")
            sys.exit(1)
        
        # Load base config
        cfg = compose(config_name="config", overrides=overrides)
        
        # Load run config
        run_config_path = os.path.join(os.path.dirname(__file__), "../config/runs", f"{run_id}.yaml")
        if os.path.exists(run_config_path):
            run_cfg = OmegaConf.load(run_config_path)
            # Merge run config into main config
            cfg = OmegaConf.merge(cfg, {"run": run_cfg})
        else:
            logger.error(f"Run config file not found: {run_config_path}")
            sys.exit(1)
        
        # Add mode overrides
        if mode == "sanity_check":
            mode_overrides = {
                "wandb": {"mode": "disabled"},
                "run": {"dataset": {"subset": 10}},  # Small subset for sanity check
                "mode": {
                    "sanity_check": True,
                    "main": False,
                    "pilot": False,
                }
            }
            cfg = OmegaConf.merge(cfg, mode_overrides)
        elif mode == "main":
            mode_overrides = {
                "wandb": {"mode": "online"},
                "mode": {
                    "sanity_check": False,
                    "main": True,
                    "pilot": False,
                }
            }
            cfg = OmegaConf.merge(cfg, mode_overrides)
        elif mode == "pilot":
            mode_overrides = {
                "wandb": {"mode": "disabled"},
                "run": {"dataset": {"subset": 50}},  # Moderate subset for pilot
                "mode": {
                    "sanity_check": False,
                    "main": False,
                    "pilot": True,
                }
            }
            cfg = OmegaConf.merge(cfg, mode_overrides)
        
        logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
        
        # Run experiment
        run_experiment(cfg)


if __name__ == "__main__":
    main()
