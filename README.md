# PyDetectGPT
![PyPI](https://img.shields.io/pypi/v/pydetectgpt?color=blue)
![Downloads](https://img.shields.io/pypi/dm/pydetectgpt?color=blue)
![License](https://img.shields.io/github/license/Dylan-Harden3/pydetectgpt?style=flat-square)
![CI](https://github.com/Dylan-Harden3/pydetectgpt/actions/workflows/ci.yml/badge.svg)


Python package for AI-generated text detection. Provides a high level api for easy adoption and more granular customization for advanced use cases.

## Quick Start
Implement an AI Plagarism detector in 4 lines of Python:
```python
from pydetectgpt import detect_ai_text

text = "text you want to check here"
result = detect_ai_text(text)
print("AI Generated" if result else "Human Written")
```

On the first run it may some time to load the model from [Hugging Face](https://huggingface.co/). After that it will be *relatively* fast.

## Usage
You can also chose different [Detection Methods](#methods), decision thresholds and use any [transformers](https://huggingface.co/docs/transformers/en/index) model for the logits:
```python
from pydetectgpt import detect_ai_text

text = "text you want to check here"
result = detect_ai_text(text, method="fastdetectgpt", threshold=1.9, model="Qwen/Qwen2.5-1.5B")
print("AI Generated" if result else "Human Written")
```
The default thresholds are:
```
"loglikelihood": -1.8,
"logrank": -0.8,
"detectllm": 2.14,
"fastdetectgpt": 1.9,
```
These were selected to minimize false positives (minimize saying its AI text when its not).

## CLI

There is also a CLI wrapper:
```bash
pydetectgpt "Your text here"
```
> "Detection Result: AI Generated" or "Detection Result: Human Written"

If you want just the 0/1 result (ex for scripting) use the `-q` flag:

```bash
pydetectgpt "Your text here" -q
```
> 0 or 1

For a full list of args see [cli.py](pydetectgpt/cli.py)

## Methods

PyDetectGPT supports four detection methods, in order of effectiveness:

1. **FastDetectGPT** (default): Implementation of [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text][1]
2. **DetectLLM**: Implementation of [DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text][2]
3. **LogRank**: Average log token rank
4. **LogLikelihood**: Basic log likelihood of the text

[1]: https://arxiv.org/abs/2310.05130 "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text"
[2]: https://arxiv.org/abs/2306.05540 "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text"

## Acknowledgements

- [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text][1] (Bao et al., ICLR 2024)
- [DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text][2] (Su et al., 2023)

## License

MIT
