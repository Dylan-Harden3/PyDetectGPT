import torch
import pytest
from torch import Tensor

from pydetectgpt import (
    log_likelihood,
    log_rank,
    likelihood_logrank_ratio,
    fast_detect_gpt,
)


def test_log_likelihood():
    # shape mismatch
    logits: Tensor = torch.randn(1, 5, 10)
    labels: Tensor = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="Labels and logits must have compatible shapes"
    ):
        log_likelihood(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="Batch size must be 1"):
        log_likelihood(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="Labels must be in vocab size"):
        log_likelihood(labels, logits)

    # some simple tests I calculated manually
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert abs(log_likelihood(labels, logits) - -1.1019428) < 1e-5

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert abs(log_likelihood(labels, logits) - -1.05657327) < 1e-5


def test_log_rank():
    # shape mismatch
    logits: Tensor = torch.randn(1, 5, 10)
    labels: Tensor = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="Labels and logits must have compatible shapes"
    ):
        log_rank(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="Batch size must be 1"):
        log_rank(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="Labels must be in vocab size"):
        log_rank(labels, logits)

    # some simple tests I calculated manually
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert abs(log_rank(labels, logits) - -0.693147) < 1e-5

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert abs(log_rank(labels, logits) - -0.3662) < 1e-5


def test_likelihood_logrank_ratio():
    # shape mismatch
    logits: Tensor = torch.randn(1, 5, 10)
    labels: Tensor = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="Labels and logits must have compatible shapes"
    ):
        likelihood_logrank_ratio(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="Batch size must be 1"):
        likelihood_logrank_ratio(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="Labels must be in vocab size"):
        likelihood_logrank_ratio(labels, logits)

    # some simple tests I calculated manually
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert abs(likelihood_logrank_ratio(labels, logits) - 1.5897675) < 1e-5

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert abs(likelihood_logrank_ratio(labels, logits) - 2.8852) < 1e-5


def test_fast_detect_gpt():
    # shape mismatch
    logits: Tensor = torch.randn(1, 5, 10)
    labels: Tensor = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="Labels and logits must have compatible shapes"
    ):
        fast_detect_gpt(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="Batch size must be 1"):
        fast_detect_gpt(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="Labels must be in vocab size"):
        fast_detect_gpt(labels, logits)

    # some simple tests I calculated manually
    # since sampling is used in fastdetectgpt I lowered the difference threshold
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert abs(fast_detect_gpt(labels, logits) - -0.085) < 5e-2

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert abs(fast_detect_gpt(labels, logits) - -0.55) < 5e-2
