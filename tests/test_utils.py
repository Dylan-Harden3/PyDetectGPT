import torch
import pytest
from pydetectgpt import log_likelihood, log_rank


def test_log_likelihood():
    # shape mismatch
    logits = torch.randn(1, 5, 10)
    labels = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="labels and logits must have compatible shapes"
    ):
        log_likelihood(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="batch size must be 1"):
        log_likelihood(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="labels must be in vocab size"):
        log_likelihood(labels, logits)

    # some simple tests I calculated manually
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert (log_likelihood(labels, logits) - -1.1019) < 1e-5

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert (log_likelihood(labels, logits) - 0.0202326) < 1e-5


def test_log_rank():
    # shape mismatch
    logits = torch.randn(1, 5, 10)
    labels = torch.randint(0, 9, (1, 6))

    with pytest.raises(
        ValueError, match="labels and logits must have compatible shapes"
    ):
        log_rank(labels, logits)

    # batch size > 1
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 9, (2, 5))

    with pytest.raises(ValueError, match="batch size must be 1"):
        log_rank(labels, logits)

    # label > vocab size
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[2, 5, 10]])

    with pytest.raises(ValueError, match="labels must be in vocab size"):
        log_rank(labels, logits)

    # some simple tests I calculated manually
    logits = torch.tensor([[[0.2, 0.3, 0.4]]])
    labels = torch.tensor([[1]])

    assert (log_rank(labels, logits) - -0.693147) < 1e-5

    logits = torch.tensor([[[2.3, 1.1, 0.5], [0.8, 2.5, 1.1], [1.5, 2.1, 0.2]]])
    labels = torch.tensor([[0, 1, 2]])

    assert (log_rank(labels, logits) - -0.3662) < 1e-5
