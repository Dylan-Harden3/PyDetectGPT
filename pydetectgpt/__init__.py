"""On device LLM-Generated text detection in Pytorch."""

from .detect import detect_ai_text
from .methods import log_likelihood, log_rank, likelihood_logrank_ratio, fast_detect_gpt

__version__ = "0.1.1"
__all__ = [
    "detect_ai_text",
    "log_likelihood",
    "log_rank",
    "likelihood_logrank_ratio",
    "fast_detect_gpt",
]
