"""Model interface for LLM inference."""

import ast
import json
import re
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnsafeExpr(Exception):
    """Exception for unsafe expressions in invariant evaluation."""
    pass


# Safe AST evaluation for invariants
_ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Index,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call,
}
_ALLOWED_FUNCS = {"abs": abs, "round": round}


def _check_ast(node: ast.AST) -> None:
    """Check if AST is safe for evaluation."""
    for n in ast.walk(node):
        if type(n) not in _ALLOWED_NODES:
            raise UnsafeExpr(f"Disallowed node: {type(n).__name__}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED_FUNCS:
                raise UnsafeExpr("Only abs/round calls allowed")
        if isinstance(n, ast.Name):
            if n.id not in {"A", "c"} and n.id not in _ALLOWED_FUNCS:
                raise UnsafeExpr(f"Disallowed name: {n.id}")


def eval_invariant(expr: str, A: float, c: Dict[str, Any]) -> bool:
    """
    Safely evaluate an invariant expression.
    
    Args:
        expr: Boolean expression string
        A: Answer value
        c: Constants dictionary
        
    Returns:
        Boolean result of evaluation
    """
    tree = ast.parse(expr, mode="eval")
    _check_ast(tree)
    code = compile(tree, "<inv>", "eval")
    return bool(eval(code, {"__builtins__": {}}, {"A": A, "c": c, **_ALLOWED_FUNCS}))


def find_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class LLMModel:
    """Wrapper for HuggingFace LLM."""
    
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ):
        """
        Initialize LLM model.
        
        Args:
            model_id: HuggingFace model ID
            dtype: Data type for model weights
            device_map: Device mapping strategy
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
        """
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=".cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=".cache"
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # Format as chat message for instruct models
        messages = [{"role": "user", "content": prompt}]
        
        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
