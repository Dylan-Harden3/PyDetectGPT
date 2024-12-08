"""PyDetectGPT Metadata."""

from setuptools import setup, find_packages

setup(
    name="pydetectgpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "numpy", "transformers"],
    extras_require={
        "dev": ["black", "ruff", "pydocstyle", "pytest", "pytest-cov"],
    },
)
