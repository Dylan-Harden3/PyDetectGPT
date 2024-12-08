"""Implementations of detection algorithms."""

from typing import Literal
from .utils import load_model, log_likelihood
import torch

DETECTION_FUNCS = {"loglikelihood": log_likelihood}
THRESHOLDS = {"loglikelihood": -1.8}


def detect_ai_text(
    text: str,
    method: Literal["loglikelihood"] = "loglikelihood",
    threshold: float = None,
    detection_model: str = "Qwen/Qwen2.5-1.5B",
) -> int:
    """Detect if `text` is written by human or ai.

    Args:
        text (str): The text to check.
        method (str, optional), default='loglikelihood': Detection method to use, must be one of ['loglikelihood'].
        threshold (float, optional), default=None: Decision threshold for `method` to use. If not provided, a default value will be used based on `method`.
        detection_model (str, optional), default=Qwen/Qwen2.5-1.5B: Huggingface Repo name for the model that `method` will use to generate logits.

    Returns:
        int: 0 if human generated 1 if machine generated.

    Raises:
        ValueError: If method is not one of ['loglikelihood'].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(detection_model)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    ).to(device)

    if method not in DETECTION_FUNCS or method not in THRESHOLDS:
        raise ValueError(
            f"In detect_ai_text `method` must be one of ['loglikelihood'], but got {method}"
        )

    method_func = DETECTION_FUNCS[method]
    if threshold is None:
        threshold = THRESHOLDS[method]

    labels = tokens.input_ids[:, 1:]  # remove bos token
    with torch.no_grad():
        logits = model(**tokens).logits[:, :-1]  # remove next token logits
    pred = method_func(labels, logits)
    print(pred)
    return 0 if pred < threshold else 1
