"""Utils used throughout source code."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


def load_model(hf_repo: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from the Hugging Face repository.

    Args:
        hf_repo (str): The Hugging Face model repository identifier (e.g., 'Qwen/Qwen2.5-1.5B').

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the model and tokenizer.

    Raises:
        ValueError: If there is an issue loading the model or tokenizer from HuggingFace.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(hf_repo).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def validate_tensor_shapes(labels: torch.Tensor, logits: torch.Tensor) -> None:
    """Validate the compatibility of labels and logits.

    Args:
        labels (torch.Tensor): Ground truth labels of shape: (1, sequence_length).
        logits (torch.Tensor): Logits of shape: (1, sequence_length, vocab_size).

    Raises:
        ValueError: If the shapes of `labels` and `logits` are incompatible or batch size is > 1.
    """
    if logits.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError(
            f"Batch size must be 1, but got logits batch size {logits.shape[0]} "
            f"and labels batch size {labels.shape[0]}"
        )

    if logits.dim() < 2:
        raise ValueError(
            f"Logits must have at least 2 dimensions, but got shape {logits.shape}"
        )

    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"Labels and logits must have compatible shapes. "
            f"Got labels shape {labels.shape} and logits shape {logits.shape[:-1]}"
        )

    if labels.max().item() >= logits.shape[-1]:
        raise ValueError(
            f"Labels must be in vocab size ({logits.shape[-1]}), "
            f"but got label {labels.max().item()}"
        )
