"""CLI for using PyDetectGPT."""

import argparse
from .detect import detect_ai_text, DETECTION_FUNCS, THRESHOLDS


def main() -> None:
    """Run detection via CLI with argparse."""
    parser = argparse.ArgumentParser(
        description="Detect if text is AI-generated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  pydetectgpt "Your text here"
""",
    )

    parser.add_argument("text", help="text to analyze")

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=list(DETECTION_FUNCS.keys()),
        default="fastdetectgpt",
        help="detection method to use (default: %(default)s)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model to use (default: %(default)s)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help=f"decision threshold (defaults: {THRESHOLDS})",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="output only the result (0 for human, 1 for AI)",
    )

    args = parser.parse_args()

    result = detect_ai_text(
        text=args.text,
        method=args.method,
        detection_model=args.model,
        threshold=args.threshold,
    )

    if args.quiet:
        print(result)
    else:
        print(f"Detection Result: {'AI Generated' if result else 'Human Written'}")


if __name__ == "__main__":
    main()
