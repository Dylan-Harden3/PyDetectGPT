"""Detection functions."""

import torch
import torch.nn.functional as F
from .utils import validate_tensor_shapes


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
    validate_tensor_shapes(labels, logits)

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
    validate_tensor_shapes(labels, logits)

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1]

    log_ranks = torch.log(ranks.float() + 1)

    return -log_ranks.mean().item()


def likelihood_logrank_ratio(labels: torch.Tensor, logits: torch.Tensor) -> float:
    """Compute the Likelihood Logrank Ratio (LRR) from DetectLLM paper.

    Args:
        labels (torch.Tensor): Ground truth labels of shape: (1, sequence_length).
        logits (torch.Tensor): Logits of shape: (1, sequence_length, vocab_size).

    Returns:
        float: The LRR Ratio.

    Raises:
        ValueError: If the shapes of `labels` and `logits` are incompatible or batch size is > 1.
    """
    validate_tensor_shapes(labels, logits)

    _log_likelihood = log_likelihood(labels, logits)
    _log_rank = log_rank(labels, logits)

    return _log_likelihood / _log_rank


def fast_detect_gpt(labels: torch.Tensor, logits: torch.Tensor) -> float:
    """Estimate the conditional probability gap using FastDetectGPT.

    Args:
        labels (torch.Tensor): Ground truth labels of shape: (1, sequence_length).
        logits (torch.Tensor): Logits of shape: (1, sequence_length, vocab_size).

    Returns:
        float: The estimated mean gap which serves as the detection metric.

    Raises:
        ValueError: If the shapes of `labels` and `logits` are incompatible or batch size is > 1.
    """
    validate_tensor_shapes(labels, logits)

    # conditional sampling
    log_probs = F.log_softmax(logits, dim=-1)
    distribution = torch.distributions.categorical.Categorical(logits=log_probs)
    x_tilde = distribution.sample([10000]).permute([1, 2, 0])

    log_likelihood_x = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).mean(dim=1)
    log_likelihood_x_tilde = log_probs.gather(dim=-1, index=x_tilde).mean(dim=1)

    # estimate the mean/variance
    mu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    # estimate conditional probability curvature
    dhat = (log_likelihood_x - mu_tilde) / sigma_tilde

    return dhat.item()
