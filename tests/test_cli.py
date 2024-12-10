import pytest
from unittest.mock import patch
from pydetectgpt.cli import main

AI_TEXT: str = (
    "Texas A&M University is located in College Station, Texas, in the southeastern part of the state. It's about 90 miles northwest of Houston and around 150 miles south of Dallas. The university's full name is Texas Agricultural and Mechanical University, and it is one of the largest public universities in the United States."
)
HUMAN_TEXT: str = (
    "The main problem the authors are trying to address is that Large Language Models require large computational resources to use. This means that as a common setup we see companies deploying GPU clusters which act as a cloud server to generate responses when a user presents a query. Aside from the vast resources needed to set up a GPU cluster this approach has 2 main downsides: sending queries over the internet via an API exposes users' private data and results in additional latency when generating responses"
)


def test_cli_loglikelihood(capsys) -> None:
    method: str = "loglikelihood"

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method, "-t", "99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method, "-t", "-99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out


def test_cli_logrank(capsys) -> None:
    method: str = "logrank"

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method, "-t", "99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method, "-t", "-99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out


def test_cli_detectllm(capsys) -> None:
    """Test CLI with detectllm method."""
    method: str = "detectllm"

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method, "-t", "99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method, "-t", "-99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out


def test_cli_fastdetectgpt(capsys) -> None:
    method: str = "fastdetectgpt"

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", method, "-t", "99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "Human Written" in captured.out

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-m", method, "-t", "-99999.9"]):
        main()
        captured = capsys.readouterr()
        assert "AI Generated" in captured.out


def test_cli_invalid_method() -> None:
    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-m", "invalid_method"]):
        with pytest.raises(SystemExit):
            main()


def test_cli_quiet_mode(capsys) -> None:
    with patch("sys.argv", ["pydetectgpt", AI_TEXT, "-q"]):
        main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "1"

    with patch("sys.argv", ["pydetectgpt", HUMAN_TEXT, "-q"]):
        main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "0"
