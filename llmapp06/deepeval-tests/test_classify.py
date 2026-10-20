"""
DeepEval LLM evaluation tests for POST /api/ai/classify.

Evaluates whether the classification endpoint returns correct, relevant,
and properly structured responses for various input texts.
"""

import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

from api_client import classify_text
from conftest import JUDGE_MODEL, json_schema_metric, output_correctness_metric, answer_relevancy_metric


# ---------------------------------------------------------------------------
# Test inputs and expected behaviors
# ---------------------------------------------------------------------------

CLASSIFY_TEST_DATA = [
    {
        "input": (
            "Python 3.12 introduces several new features including improved error "
            "messages, a new type parameter syntax, and performance improvements "
            "in the interpreter."
        ),
        "expected_labels": ["technology", "programming", "software"],
        "expected_category": "technology",
    },
    {
        "input": (
            "The Lakers defeated the Celtics 112-108 in overtime last night. "
            "LeBron James scored 35 points and had 10 assists in the victory."
        ),
        "expected_labels": ["sports", "basketball", "NBA"],
        "expected_category": "sports",
    },
    {
        "input": (
            "To make a classic Italian carbonara, you need guanciale, eggs, "
            "pecorino romano cheese, black pepper, and spaghetti. Cook the "
            "pasta al dente and toss with the egg and cheese mixture."
        ),
        "expected_labels": ["food", "cooking", "recipe"],
        "expected_category": "food",
    },
    {
        "input": (
            "The Federal Reserve announced it will maintain interest rates at "
            "their current level, citing concerns about inflation and the labor "
            "market outlook for the coming quarter."
        ),
        "expected_labels": ["finance", "economics", "policy"],
        "expected_category": "finance",
    },
    {
        "input": (
            "A new study published in Nature shows that regular exercise can "
            "reduce the risk of heart disease by up to 30 percent and improve "
            "overall mental health outcomes."
        ),
        "expected_labels": ["health", "science", "medical"],
        "expected_category": "health",
    },
]


# ---------------------------------------------------------------------------
# Build test cases by calling the live API
# ---------------------------------------------------------------------------

def build_classify_test_cases():
    test_cases = []
    for data in CLASSIFY_TEST_DATA:
        response = classify_text(data["input"])
        actual_output = json.dumps(response)
        expected_output = json.dumps({
            "labels": data["expected_labels"],
            "primaryCategory": data["expected_category"],
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


classify_test_cases = build_classify_test_cases()
classify_dataset = EvaluationDataset()
for tc in classify_test_cases:
    classify_dataset.add_test_case(tc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

SCHEMA_DESC = (
    "The JSON must contain: 'labels' (array of strings), "
    "'primaryCategory' (string), and 'confidence' (number between 0 and 1)."
)

classify_schema_metric = json_schema_metric(SCHEMA_DESC)

classify_correctness_metric = GEval(
    name="Classification Correctness",
    criteria=(
        "Evaluate whether the classification labels and primary category in the "
        "actual output are reasonable and accurate for the given input text. "
        "The labels should be topically relevant to the text content. "
        "The primaryCategory should represent the dominant topic. "
        "Closely related categories (e.g. 'business' vs 'finance') should be "
        "considered acceptable. "
        "The confidence score should be between 0 and 1."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
    model=JUDGE_MODEL,
)

classify_relevancy_metric = answer_relevancy_metric()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_case", classify_dataset.test_cases)
def test_classify_schema_compliance(test_case: LLMTestCase):
    """Verify the classification response has the correct JSON structure."""
    assert_test(test_case, [classify_schema_metric])


@pytest.mark.parametrize("test_case", classify_dataset.test_cases)
def test_classify_correctness(test_case: LLMTestCase):
    """Verify the classification labels and category are accurate."""
    assert_test(test_case, [classify_correctness_metric])


@pytest.mark.parametrize("test_case", classify_dataset.test_cases)
def test_classify_relevancy(test_case: LLMTestCase):
    """Verify the classification response is relevant to the input."""
    assert_test(test_case, [classify_relevancy_metric])
