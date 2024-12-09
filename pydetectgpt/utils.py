"""Utils used throughout source code."""

import torch
import torch.nn.functional as F
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


def log_likelihood(labels: torch.Tensor, logits: torch.Tensor) -> float:
    """Compute the loglikelihood of labels in logits.

    Args:
        labels (torch.Tensor): Ground truth labels of shape: (1, sequence_length).
        logits (torch.Tensor): Logits of shape: (1, sequence_length, vocab_size).

    Returns:
        float: The mean loglikelihood.

    Raises:
        ValueError: If the shapes of `labels` and `logits` are incompatible or batch size is > 1.
    """
    if logits.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError(
            f"In log_likelihood, batch size must be 1, but got logits batch size {logits.shape[0]} "
            f"and labels batch size {labels.shape[0]}"
        )

    if logits.dim() < 2:
        raise ValueError(
            f"In log_likelihood, logits must have at least 2 dimensions, but got shape {logits.shape}"
        )

    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"In log_likelihood, labels and logits must have compatible shapes. "
            f"Got labels shape {labels.shape} and logits shape {logits.shape[:-1]}"
        )

    if labels.max().item() >= logits.shape[-1]:
        raise ValueError(
            f"In log_likelihood, labels must be in vocab size ({logits.shape[-1]}), "
            f"but got label {labels.max().item()}"
        )

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    log_probs = F.log_softmax(logits, dim=-1)
    actual_token_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )
    return actual_token_probs.mean().item()


def log_rank(labels: torch.Tensor, logits: torch.Tensor) -> float:
    """Compute the negative average log rank of labels in logits.

    Args:
        labels (torch.Tensor): Ground truth labels of shape: (1, sequence_length).
        logits (torch.Tensor): Logits of shape: (1, sequence_length, vocab_size).

    Returns:
        float: The negative mean logrank.

    Raises:
        ValueError: If the shapes of `labels` and `logits` are incompatible or batch size is > 1.
    """
    if logits.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError(
            f"In log_likelihood, batch size must be 1, but got logits batch size {logits.shape[0]} "
            f"and labels batch size {labels.shape[0]}"
        )

    if logits.dim() < 2:
        raise ValueError(
            f"In log_likelihood, logits must have at least 2 dimensions, but got shape {logits.shape}"
        )

    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"In log_likelihood, labels and logits must have compatible shapes. "
            f"Got labels shape {labels.shape} and logits shape {logits.shape[:-1]}"
        )

    if labels.max().item() >= logits.shape[-1]:
        raise ValueError(
            f"In log_likelihood, labels must be in vocab size ({logits.shape[-1]}), "
            f"but got label {labels.max().item()}"
        )

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]

    log_ranks = torch.log(ranks.float() + 1)

    return -log_ranks.mean().item()
