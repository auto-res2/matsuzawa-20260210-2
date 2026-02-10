"""Evaluation script for analyzing and comparing runs."""

import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("run_ids", type=str, help="JSON list of run IDs")
    return parser.parse_args()


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: List of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    
    data = np.array(data)
    bootstrapped_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    mean = np.mean(data)
    lower = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    
    return mean, lower, upper


def load_run_metrics(results_dir: str, run_id: str) -> Dict[str, Any]:
    """Load metrics from a run directory."""
    run_dir = os.path.join(results_dir, run_id)
    metrics_path = os.path.join(run_dir, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.json not found for {run_id}")
        return {}
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return metrics


def load_run_results(results_dir: str, run_id: str) -> List[Dict[str, Any]]:
    """Load detailed results from a run directory."""
    run_dir = os.path.join(results_dir, run_id)
    results_path = os.path.join(run_dir, "results.json")
    
    if not os.path.exists(results_path):
        print(f"Warning: results.json not found for {run_id}")
        return []
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    return results


def create_accuracy_plot(metrics_by_run: Dict[str, Dict], output_path: str):
    """Create bar plot of accuracy with confidence intervals."""
    run_ids = list(metrics_by_run.keys())
    accuracies = [metrics_by_run[rid].get("accuracy", 0) for rid in run_ids]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(run_ids))
    ax.bar(x, accuracies)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Run")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Created plot: {output_path}")


def create_regression_rate_plot(metrics_by_run: Dict[str, Dict], output_path: str):
    """Create bar plot of regression rate."""
    run_ids = list(metrics_by_run.keys())
    regression_rates = [metrics_by_run[rid].get("regression_rate", 0) for rid in run_ids]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(run_ids))
    ax.bar(x, regression_rates, color='red', alpha=0.7)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Regression Rate")
    ax.set_title("Regression Rate by Run (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Created plot: {output_path}")


def create_metrics_comparison_plot(metrics_by_run: Dict[str, Dict], output_path: str):
    """Create grouped bar plot comparing multiple metrics."""
    run_ids = list(metrics_by_run.keys())
    
    metrics_to_plot = ["accuracy", "base_accuracy", "regression_rate", "change_rate", "correction_precision"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(run_ids))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_by_run[rid].get(metric, 0) or 0 for rid in run_ids]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Value")
    ax.set_title("Metrics Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Created plot: {output_path}")


def create_per_run_plots(results_dir: str, run_id: str, results: List[Dict], metrics: Dict):
    """Create per-run visualization plots."""
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Plot 1: Correctness over examples
    fig, ax = plt.subplots(figsize=(12, 4))
    correct = [r["is_correct"] for r in results]
    ax.plot(correct, marker='o', markersize=2, linestyle='-', linewidth=0.5)
    ax.set_xlabel("Example Index")
    ax.set_ylabel("Correct (1) / Incorrect (0)")
    ax.set_title(f"Correctness per Example - {run_id}")
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    output_path = os.path.join(run_dir, "correctness_per_example.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Created plot: {output_path}")
    
    # Plot 2: Base vs Final accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Base", "Final"]
    values = [metrics.get("base_accuracy", 0), metrics.get("accuracy", 0)]
    ax.bar(categories, values)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Base vs Final Accuracy - {run_id}")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = os.path.join(run_dir, "base_vs_final_accuracy.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Created plot: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    results_dir = args.results_dir
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Load metrics for all runs
    metrics_by_run = {}
    results_by_run = {}
    
    for run_id in run_ids:
        print(f"\nLoading results for {run_id}...")
        metrics = load_run_metrics(results_dir, run_id)
        results = load_run_results(results_dir, run_id)
        
        metrics_by_run[run_id] = metrics
        results_by_run[run_id] = results
        
        # Export per-run metrics
        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Exported metrics: {metrics_path}")
        
        # Create per-run plots
        if results:
            create_per_run_plots(results_dir, run_id, results, metrics)
    
    # Create comparison directory
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Determine primary metric and best runs
    primary_metric = "regression_rate"
    
    # Find proposed and baseline runs
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]
    
    # Get best of each
    best_proposed = None
    best_proposed_value = float('inf')
    for rid in proposed_runs:
        value = metrics_by_run[rid].get(primary_metric, float('inf'))
        if value is not None and value < best_proposed_value:
            best_proposed = rid
            best_proposed_value = value
    
    best_baseline = None
    best_baseline_value = float('inf')
    for rid in baseline_runs:
        value = metrics_by_run[rid].get(primary_metric, float('inf'))
        if value is not None and value < best_baseline_value:
            best_baseline = rid
            best_baseline_value = value
    
    # Compute gap
    gap = None
    if best_proposed_value is not None and best_baseline_value is not None:
        gap = best_baseline_value - best_proposed_value
    
    # Export aggregated metrics
    aggregated_metrics = {
        "primary_metric": primary_metric,
        "metrics_by_run": {rid: metrics_by_run[rid] for rid in run_ids},
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap
    }
    
    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    with open(aggregated_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nExported aggregated metrics: {aggregated_path}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    
    accuracy_plot = os.path.join(comparison_dir, "accuracy_comparison.png")
    create_accuracy_plot(metrics_by_run, accuracy_plot)
    
    regression_plot = os.path.join(comparison_dir, "regression_rate_comparison.png")
    create_regression_rate_plot(metrics_by_run, regression_plot)
    
    metrics_plot = os.path.join(comparison_dir, "metrics_comparison.png")
    create_metrics_comparison_plot(metrics_by_run, metrics_plot)
    
    print("\nEvaluation complete!")
    print(f"\nBest proposed: {best_proposed} ({primary_metric}={best_proposed_value:.4f})")
    print(f"Best baseline: {best_baseline} ({primary_metric}={best_baseline_value:.4f})")
    if gap is not None:
        print(f"Gap (baseline - proposed): {gap:.4f}")


if __name__ == "__main__":
    main()
