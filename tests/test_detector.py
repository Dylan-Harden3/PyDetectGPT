from pydetectgpt import detect_ai_text
import pytest


def test_detect_ai_text():
    # I asked chatgpt "Where is Texas A&M?"
    ai_text = "Texas A&M University is located in College Station, Texas, in the southeastern part of the state. It's about 90 miles northwest of Houston and around 150 miles south of Dallas. The university's full name is Texas Agricultural and Mechanical University, and it is one of the largest public universities in the United States."

    assert detect_ai_text(ai_text, "loglikelihood") == 1

    # random paragraph from one of my assignments (written by human)
    human_text = "The main problem the authors are trying to address is that Large Language Models require large computational resources to use. This means that as a common setup we see companies deploying GPU clusters which act as a cloud server to generate responses when a user presents a query. Aside from the vast resources needed to set up a GPU cluster this approach has 2 main downsides: sending queries over the internet via an API exposes users’ private data and results in additional latency when generating responses"
    assert detect_ai_text(human_text, "loglikelihood") == 0

    # invalid method name
    with pytest.raises(ValueError, match="must be one of"):
        detect_ai_text(ai_text, method="notvalidmethodname")

    # high threshold should always be 0 (human)
    assert detect_ai_text(ai_text, "loglikelihood", 99999.9) == 0

    # low threshold should always be 1 (ai)
    assert detect_ai_text(human_text, "loglikelihood", -99999.9) == 1
