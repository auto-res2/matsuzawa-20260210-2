"""Dataset loading and preprocessing utilities."""

import re
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last numeric value from text, removing commas."""
    num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
    cleaned_text = text.replace(",", "")
    nums = num_re.findall(cleaned_text)
    return nums[-1] if nums else None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize an answer for comparison."""
    if answer is None:
        return None
    # Remove whitespace and convert to lowercase
    answer = str(answer).strip().lower()
    # Try to parse as number and normalize
    try:
        # Check if it's an integer
        if '.' not in answer:
            return str(int(float(answer)))
        else:
            # Keep as float but normalize
            return str(float(answer))
    except (ValueError, TypeError):
        return answer


def load_gsm8k_dataset(subset: int = 200, split: str = "test", cache_dir: str = ".cache") -> Tuple[List[str], List[str]]:
    """
    Load GSM8K dataset.
    
    Args:
        subset: Number of examples to load
        split: Dataset split (train/test)
        cache_dir: Directory for caching
        
    Returns:
        Tuple of (questions, gold_answers)
    """
    ds = load_dataset("gsm8k", "main", split=f"{split}[:{subset}]", cache_dir=cache_dir)
    
    questions = [x["question"] for x in ds]
    # Extract numeric answers from the answer field (which contains explanation + #### answer)
    gold_answers = []
    for x in ds:
        answer_text = x["answer"]
        # GSM8K answers are in format "explanation #### numeric_answer"
        if "####" in answer_text:
            gold_answer = answer_text.split("####")[-1].strip()
        else:
            gold_answer = answer_text.strip()
        # Extract just the number
        gold_answer = extract_last_number(gold_answer)
        gold_answers.append(gold_answer)
    
    return questions, gold_answers


def load_dataset_by_name(
    dataset_name: str, 
    subset: int = 200, 
    split: str = "test", 
    cache_dir: str = ".cache"
) -> Tuple[List[str], List[str]]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of the dataset
        subset: Number of examples to load
        split: Dataset split
        cache_dir: Cache directory
        
    Returns:
        Tuple of (questions, gold_answers)
    """
    if dataset_name.lower() == "gsm8k":
        return load_gsm8k_dataset(subset=subset, split=split, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
