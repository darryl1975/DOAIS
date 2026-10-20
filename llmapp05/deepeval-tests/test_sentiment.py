"""
DeepEval LLM evaluation tests for POST /api/ai/sentiment.

Evaluates whether the sentiment analysis endpoint returns correct sentiment
labels, scores, emotions, and properly structured responses.
"""

import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

from api_client import analyze_sentiment
from conftest import JUDGE_MODEL, json_schema_metric, output_correctness_metric, answer_relevancy_metric


# ---------------------------------------------------------------------------
# Test inputs and expected behaviors
# ---------------------------------------------------------------------------

SENTIMENT_TEST_DATA = [
    {
        "input": (
            "I absolutely love this product! It exceeded all my expectations "
            "and the customer service was fantastic. Best purchase I have made "
            "this year!"
        ),
        "expected_sentiment": "positive",
        "expected_score_range": (0.5, 1.0),
        "expected_emotions": ["joy", "satisfaction", "excitement"],
    },
    {
        "input": (
            "This is the worst experience I have ever had. The product broke "
            "after one day, customer support was rude, and I still have not "
            "received my refund after three weeks."
        ),
        "expected_sentiment": "negative",
        "expected_score_range": (-1.0, -0.3),
        "expected_emotions": ["anger", "frustration", "disappointment"],
    },
    {
        "input": (
            "The meeting is scheduled for 3pm tomorrow in conference room B. "
            "Please bring your laptop and the quarterly report."
        ),
        "expected_sentiment": "neutral",
        "expected_score_range": (-0.3, 0.3),
        "expected_emotions": [],
    },
    {
        "input": (
            "The movie had incredible special effects and a talented cast, "
            "but the plot was confusing and the ending felt rushed and "
            "unsatisfying."
        ),
        "expected_sentiment": "mixed",
        "expected_score_range": (-0.3, 0.5),
        "expected_emotions": ["admiration", "disappointment"],
    },
    {
        "input": (
            "After years of hard work, I finally graduated from medical school "
            "today. My family was there to support me and I could not be more "
            "grateful for this moment."
        ),
        "expected_sentiment": "positive",
        "expected_score_range": (0.5, 1.0),
        "expected_emotions": ["joy", "gratitude", "pride"],
    },
]


# ---------------------------------------------------------------------------
# Build test cases by calling the live API
# ---------------------------------------------------------------------------

def build_sentiment_test_cases():
    test_cases = []
    for data in SENTIMENT_TEST_DATA:
        response = analyze_sentiment(data["input"])
        actual_output = json.dumps(response)
        expected_output = json.dumps({
            "overallSentiment": data["expected_sentiment"],
            "sentimentScore": sum(data["expected_score_range"]) / 2,
            "emotions": data["expected_emotions"],
            "confidence": 0.9,
        })
        test_cases.append(
            LLMTestCase(
                input=data["input"],
                actual_output=actual_output,
                expected_output=expected_output,
            )
        )
    return test_cases


sentiment_test_cases = build_sentiment_test_cases()
sentiment_dataset = EvaluationDataset()
for tc in sentiment_test_cases:
    sentiment_dataset.add_test_case(tc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

SCHEMA_DESC = (
    "The JSON must contain: 'overallSentiment' (string: positive, negative, "
    "neutral, or mixed), 'sentimentScore' (number between -1 and 1), 'emotions' "
    "(array of strings), and 'confidence' (number between 0 and 1)."
)

sentiment_schema_metric = json_schema_metric(SCHEMA_DESC)

sentiment_correctness_metric = GEval(
    name="Sentiment Correctness",
    criteria=(
        "Evaluate whether the sentiment analysis is accurate for the input text. "
        "Check that: (1) overallSentiment correctly identifies the dominant "
        "sentiment as positive, negative, neutral, or mixed, (2) sentimentScore is "
        "numerically consistent with the overallSentiment (positive text should "
        "have positive scores, negative text should have negative scores), "
        "(3) the detected emotions are plausible for the given text."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=JUDGE_MODEL,
)

sentiment_emotion_metric = GEval(
    name="Emotion Detection Accuracy",
    criteria=(
        "Evaluate whether the emotions detected in the actual output are "
        "reasonable and plausible for the given input text. The emotions "
        "should reflect the emotional tone conveyed in the text. Synonyms "
        "and closely related emotions should be considered acceptable."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
    model=JUDGE_MODEL,
)

sentiment_relevancy_metric = answer_relevancy_metric()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_case", sentiment_dataset.test_cases)
def test_sentiment_schema_compliance(test_case: LLMTestCase):
    """Verify the sentiment response has the correct JSON structure."""
    assert_test(test_case, [sentiment_schema_metric])


@pytest.mark.parametrize("test_case", sentiment_dataset.test_cases)
def test_sentiment_correctness(test_case: LLMTestCase):
    """Verify the sentiment label and score are accurate."""
    assert_test(test_case, [sentiment_correctness_metric])


@pytest.mark.parametrize("test_case", sentiment_dataset.test_cases)
def test_sentiment_emotion_detection(test_case: LLMTestCase):
    """Verify the detected emotions are plausible for the input."""
    assert_test(test_case, [sentiment_emotion_metric])


@pytest.mark.parametrize("test_case", sentiment_dataset.test_cases)
def test_sentiment_relevancy(test_case: LLMTestCase):
    """Verify the sentiment response is relevant to the input."""
    assert_test(test_case, [sentiment_relevancy_metric])
