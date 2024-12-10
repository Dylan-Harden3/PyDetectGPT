# PyDetectGPT

LLM-Generated text detection in Pytorch.

## Quick Start
```python
from pydetectgpt import detect_ai_text

text = "text you want to check here"
result = detect_ai_text(text)
print("AI Generated" if result else "Human Written")
```

## CLI

There is also a CLI wrapper.
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

## Detection Methods

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
