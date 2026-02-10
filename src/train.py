"""Training script for single run execution."""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import LLMModel, eval_invariant, find_json
from src.preprocess import extract_last_number, load_dataset_by_name, normalize_answer

logger = logging.getLogger(__name__)


def prompt_cot_rationale(question: str) -> str:
    """Prompt for Zero-shot CoT reasoning."""
    return f"Q: {question}\nA: Let's think step by step."


def prompt_answer_from_rationale(question: str, rationale: str) -> str:
    """Prompt to extract final answer from reasoning."""
    return (
        f"Q: {question}\nReasoning: {rationale}\n"
        "Final: Therefore, the final numeric answer is"
    )


def prompt_xic_verify(question: str, rationale: str, answer: str) -> str:
    """Prompt for XIC-RZeroCoT verification (executable invariants)."""
    return (
        "You are a verifier in falsification-first mode. Your job is to find a *checkable* reason the proposed answer is wrong. "
        "If you cannot find any violated necessary constraint, you must CONFIRM and not change the answer.\n\n"
        "Return STRICT JSON only (no extra text).\n"
        "Requirements:\n"
        "- Extract a small dict of numeric constants copied from the question as `constants` (numbers with brief keys).\n"
        "- Provide estimate_range {low, high} as rough magnitude bounds.\n"
        "- Provide 3-6 invariants as executable boolean expressions over A and c, where c is the constants dict.\n"
        "  Allowed: + - * / // % **, comparisons, and/or/not, abs(), round(), and c[\"key\"].\n"
        "  Examples: A>=0 ; A<=c[\"total\"] ; abs(A-round(A))<1e-9 ; A%2==0\n"
        "- Set status=CORRECTED only if at least one invariant is violated by the original answer and the corrected answer satisfies all invariants.\n\n"
        "Schema:\n"
        "{\n"
        "  \"status\": \"CONFIRMED\"|\"CORRECTED\",\n"
        "  \"orig_answer\": <string>,\n"
        "  \"new_answer\": <string or empty>,\n"
        "  \"constants\": {<key>: <number>, ...},\n"
        "  \"estimate_range\": {\"low\": <number>, \"high\": <number>},\n"
        "  \"invariants\": [{\"name\": <string>, \"expr\": <string>}],\n"
        "  \"claimed_violations\": [<invariant name strings>]\n"
        "}\n\n"
        f"Q: {question}\nProposed reasoning: {rationale}\nProposed final answer: {answer}\n"
    )


def prompt_ig_verify(question: str, rationale: str, answer: str) -> str:
    """Prompt for IG-RZeroCoT verification (natural language invariants)."""
    return (
        "You are a verifier. Your job is to check if the proposed answer violates any necessary constraints.\n\n"
        "Return STRICT JSON only (no extra text).\n"
        "Requirements:\n"
        "- List 3-6 invariants as natural language statements (e.g., 'answer must be non-negative', 'answer cannot exceed total', etc.).\n"
        "- If you believe the original answer violates any invariant, set status=CORRECTED and provide a corrected answer.\n"
        "- Otherwise set status=CONFIRMED.\n\n"
        "Schema:\n"
        "{\n"
        "  \"status\": \"CONFIRMED\"|\"CORRECTED\",\n"
        "  \"orig_answer\": <string>,\n"
        "  \"new_answer\": <string or empty>,\n"
        "  \"invariants\": [<list of natural language constraint strings>],\n"
        "  \"claimed_violations\": [<list of violated constraint descriptions>]\n"
        "}\n\n"
        f"Q: {question}\nProposed reasoning: {rationale}\nProposed final answer: {answer}\n"
    )


def in_range(x: float, lo: float, hi: float) -> bool:
    """Check if x is within range [lo, hi]."""
    if lo > hi:
        lo, hi = hi, lo
    return lo <= x <= hi


def solve_zeroshot_cot(model: LLMModel, question: str, max_tokens_solve: int, max_tokens_verify: int) -> Tuple[Optional[str], str]:
    """
    Solve using Zero-shot CoT (baseline).
    
    Returns:
        Tuple of (answer, rationale)
    """
    rationale = model.generate(prompt_cot_rationale(question), max_new_tokens=max_tokens_solve)
    answer_text = model.generate(prompt_answer_from_rationale(question, rationale), max_new_tokens=128)
    answer = extract_last_number(answer_text)
    return answer, rationale


def solve_xic_rzerocot(
    model: LLMModel, 
    question: str, 
    max_tokens_solve: int, 
    max_tokens_verify: int
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Solve using XIC-RZeroCoT (proposed method).
    
    Returns:
        Tuple of (final_answer, debug_info)
    """
    # Step 1: Base solve
    base_answer, rationale = solve_zeroshot_cot(model, question, max_tokens_solve, max_tokens_verify)
    
    debug_info = {
        "base_answer": base_answer,
        "rationale": rationale,
        "changed": False,
        "certificate_valid": False,
        "gating_reason": "no_verify"
    }
    
    if base_answer is None:
        return None, debug_info
    
    # Step 2: Verifier
    verify_text = model.generate(
        prompt_xic_verify(question, rationale, base_answer),
        max_new_tokens=max_tokens_verify
    )
    
    debug_info["verify_text"] = verify_text
    
    # Parse JSON
    v = find_json(verify_text)
    if not v:
        debug_info["gating_reason"] = "json_parse_failed"
        return base_answer, debug_info
    
    debug_info["verify_json"] = v
    
    # Parse answers
    orig = extract_last_number(str(v.get("orig_answer", base_answer))) or base_answer
    new = extract_last_number(str(v.get("new_answer", "")))
    
    try:
        A_orig = float(orig)
    except Exception:
        debug_info["gating_reason"] = "orig_answer_not_numeric"
        return base_answer, debug_info
    
    invs = v.get("invariants", []) or []
    c = v.get("constants", {}) or {}
    
    # Evaluate invariants on original
    orig_results = []
    for inv in invs:
        expr = str(inv.get("expr", ""))
        if not expr:
            continue
        try:
            orig_results.append(eval_invariant(expr, A_orig, c))
        except Exception as e:
            debug_info["gating_reason"] = f"invariant_eval_failed: {e}"
            return base_answer, debug_info
    
    if len(orig_results) == 0:
        debug_info["gating_reason"] = "no_valid_invariants"
        return base_answer, debug_info
    
    debug_info["certificate_valid"] = True
    debug_info["invariant_results_orig"] = orig_results
    
    any_fail_orig = not all(orig_results)
    
    status = str(v.get("status", "")).upper()
    if status != "CORRECTED" or new is None:
        debug_info["gating_reason"] = "status_not_corrected"
        return orig, debug_info
    
    # Gate correction with external checks
    try:
        A_new = float(new)
    except Exception:
        debug_info["gating_reason"] = "new_answer_not_numeric"
        return orig, debug_info
    
    # Corrected must pass all invariants
    new_results = []
    for inv in invs:
        expr = str(inv.get("expr", ""))
        if not expr:
            continue
        try:
            result = eval_invariant(expr, A_new, c)
            new_results.append(result)
            if not result:
                debug_info["gating_reason"] = "new_answer_fails_invariant"
                return orig, debug_info
        except Exception as e:
            debug_info["gating_reason"] = f"new_answer_eval_failed: {e}"
            return orig, debug_info
    
    debug_info["invariant_results_new"] = new_results
    
    # Must have found a concrete contradiction in the original
    if not any_fail_orig:
        debug_info["gating_reason"] = "no_violation_in_orig"
        return orig, debug_info
    
    # Estimate-range check (soft gate)
    est = v.get("estimate_range", {}) or {}
    try:
        lo, hi = float(est.get("low")), float(est.get("high"))
        if not in_range(A_new, lo, hi):
            debug_info["gating_reason"] = "out_of_estimate_range"
            return orig, debug_info
    except Exception:
        pass
    
    debug_info["changed"] = True
    debug_info["gating_reason"] = "accepted"
    return str(new), debug_info


def solve_ig_rzerocot(
    model: LLMModel, 
    question: str, 
    max_tokens_solve: int, 
    max_tokens_verify: int
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Solve using IG-RZeroCoT (comparative method - natural language invariants).
    
    Returns:
        Tuple of (final_answer, debug_info)
    """
    # Step 1: Base solve
    base_answer, rationale = solve_zeroshot_cot(model, question, max_tokens_solve, max_tokens_verify)
    
    debug_info = {
        "base_answer": base_answer,
        "rationale": rationale,
        "changed": False
    }
    
    if base_answer is None:
        return None, debug_info
    
    # Step 2: Verifier
    verify_text = model.generate(
        prompt_ig_verify(question, rationale, base_answer),
        max_new_tokens=max_tokens_verify
    )
    
    debug_info["verify_text"] = verify_text
    
    # Parse JSON
    v = find_json(verify_text)
    if not v:
        return base_answer, debug_info
    
    debug_info["verify_json"] = v
    
    status = str(v.get("status", "")).upper()
    if status != "CORRECTED":
        return base_answer, debug_info
    
    # Accept the model's correction directly (no external gating)
    new = extract_last_number(str(v.get("new_answer", "")))
    if new is not None:
        debug_info["changed"] = True
        return new, debug_info
    
    return base_answer, debug_info


def exact_match(pred: Optional[str], gold: Optional[str]) -> int:
    """Check if prediction matches gold answer."""
    if pred is None or gold is None:
        return 0
    return int(normalize_answer(pred) == normalize_answer(gold))


def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """Run the experiment."""
    logger.info(f"Starting experiment: {cfg.run.run_id}")
    logger.info(f"Method: {cfg.run.method.name}")
    logger.info(f"Model: {cfg.run.model.name}")
    logger.info(f"Dataset: {cfg.run.dataset.name}")
    
    # Initialize WandB if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        logger.info(f"WandB run URL: {wandb.run.get_url()}")
    else:
        logger.info("WandB disabled")
    
    # Load model
    logger.info("Loading model...")
    model = LLMModel(
        model_id=cfg.run.model.hf_model_id,
        dtype=cfg.run.model.dtype,
        device_map=cfg.run.model.device_map,
        temperature=cfg.run.model.temperature,
        top_p=cfg.run.model.top_p,
        do_sample=cfg.run.model.do_sample,
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    questions, gold_answers = load_dataset_by_name(
        dataset_name=cfg.run.dataset.name,
        subset=cfg.run.dataset.subset,
        split=cfg.run.dataset.split,
        cache_dir=cfg.run.dataset.cache_dir
    )
    
    num_examples = len(questions)
    logger.info(f"Loaded {num_examples} examples")
    
    # Run inference
    logger.info("Running inference...")
    results = []
    
    method_type = cfg.run.method.type
    
    for idx, (question, gold) in enumerate(zip(questions, gold_answers)):
        if idx % cfg.run.training.log_every == 0:
            logger.info(f"Processing example {idx}/{num_examples}")
        
        # Select method
        if method_type == "proposed":
            final_answer, debug_info = solve_xic_rzerocot(
                model, question,
                cfg.run.model.max_new_tokens_solve,
                cfg.run.model.max_new_tokens_verify
            )
        elif method_type == "comparative":
            final_answer, debug_info = solve_ig_rzerocot(
                model, question,
                cfg.run.model.max_new_tokens_solve,
                cfg.run.model.max_new_tokens_verify
            )
        else:
            raise ValueError(f"Unknown method type: {method_type}")
        
        # Compute metrics
        is_correct = exact_match(final_answer, gold)
        base_answer = debug_info.get("base_answer")
        base_correct = exact_match(base_answer, gold)
        
        result = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "base_answer": base_answer,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "base_correct": base_correct,
            "debug_info": debug_info
        }
        results.append(result)
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "step": idx,
                "is_correct": is_correct,
                "base_correct": base_correct,
                "changed": debug_info.get("changed", False)
            })
    
    # Compute aggregated metrics
    logger.info("Computing metrics...")
    
    accuracy = sum(r["is_correct"] for r in results) / num_examples
    base_accuracy = sum(r["base_correct"] for r in results) / num_examples
    
    # Regression rate: P(final wrong | base correct)
    base_correct_mask = [r["base_correct"] for r in results]
    if sum(base_correct_mask) > 0:
        regressions = sum(
            1 for r in results 
            if r["base_correct"] and not r["is_correct"]
        )
        regression_rate = regressions / sum(base_correct_mask)
    else:
        regression_rate = 0.0
    
    # Change rate
    change_rate = sum(r["debug_info"].get("changed", False) for r in results) / num_examples
    
    # Correction precision: P(final correct | changed)
    changed_mask = [r["debug_info"].get("changed", False) for r in results]
    if sum(changed_mask) > 0:
        correction_precision = sum(
            1 for r in results 
            if r["debug_info"].get("changed", False) and r["is_correct"]
        ) / sum(changed_mask)
    else:
        correction_precision = 0.0
    
    # Certificate valid rate (for XIC only)
    if method_type == "proposed":
        certificate_valid_rate = sum(
            r["debug_info"].get("certificate_valid", False) for r in results
        ) / num_examples
        invariant_falsification_rate = sum(
            1 for r in results
            if r["debug_info"].get("certificate_valid", False) and
            not all(r["debug_info"].get("invariant_results_orig", [True]))
        ) / num_examples
    else:
        certificate_valid_rate = None
        invariant_falsification_rate = None
    
    metrics = {
        "accuracy": accuracy,
        "base_accuracy": base_accuracy,
        "regression_rate": regression_rate,
        "change_rate": change_rate,
        "correction_precision": correction_precision,
        "certificate_valid_rate": certificate_valid_rate,
        "invariant_falsification_rate": invariant_falsification_rate,
        "num_examples": num_examples
    }
    
    logger.info(f"Results: {metrics}")
    
    # Save results
    results_dir = cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Log to WandB
    if cfg.wandb.mode != "disabled":
        for key, value in metrics.items():
            if value is not None:
                wandb.summary[key] = value
        wandb.finish()
    
    # Sanity validation
    perform_sanity_validation(results, metrics, cfg)
    
    return metrics


def perform_sanity_validation(results: List[Dict], metrics: Dict, cfg: DictConfig):
    """Perform sanity validation checks."""
    if not cfg.mode.sanity_check:
        return
    
    logger.info("Performing sanity validation...")
    
    num_steps = len(results)
    
    # Check we have at least 5 steps
    if num_steps < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_steps (got {num_steps}, need >=5)")
        summary = {"steps": num_steps}
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
        return
    
    # Check all metrics are finite
    for key, value in metrics.items():
        if value is not None and (math.isnan(value) or math.isinf(value)):
            print(f"SANITY_VALIDATION: FAIL reason=non_finite_metric ({key}={value})")
            print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")
            return
    
    # Check accuracy is not always 0
    if metrics["accuracy"] == 0.0:
        print("SANITY_VALIDATION: FAIL reason=zero_accuracy")
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")
        return
    
    # All checks passed
    print("SANITY_VALIDATION: PASS")
    summary = {
        "steps": num_steps,
        "accuracy": metrics["accuracy"],
        "base_accuracy": metrics["base_accuracy"],
        "regression_rate": metrics["regression_rate"],
        "change_rate": metrics["change_rate"]
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
