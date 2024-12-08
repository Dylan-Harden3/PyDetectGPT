"""On device LLM-Generated text detection in Pytorch."""

from .detect import detect_ai_text
from .utils import log_likelihood

__version__ = "0.1.0"
__all__ = ["detect_ai_text", "log_likelihood"]
