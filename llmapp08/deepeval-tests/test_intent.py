"""
DeepEval LLM evaluation tests for POST /api/ai/intent.

Evaluates whether the intent detection endpoint correctly identifies
the purpose and category behind different types of text input.
"""

import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

from api_client import detect_intent
from conftest import JUDGE_MODEL, json_schema_metric, output_correctness_metric, answer_relevancy_metric


# ---------------------------------------------------------------------------
# Test inputs and expected behaviors
# ---------------------------------------------------------------------------

INTENT_TEST_DATA = [
    {
        "input": "What time does the store close on weekends?",
        "expected_category": "question",
        "expected_primary_intent": "inquiry about store hours",
    },
    {
        "input": "Delete all the files in the temp directory and restart the server.",
        "expected_category": "command",
        "expected_primary_intent": "system administration task",
    },
    {
        "input": (
            "Could you please send me the quarterly financial report by end "
            "of day? I need it for the board meeting tomorrow."
        ),
        "expected_category": "request",
        "expected_primary_intent": "requesting a document",
    },
    {
        "input": (
            "The annual revenue for the company increased by 15 percent compared "
            "to last year, driven primarily by growth in the cloud services "
            "division."
        ),
        "expected_category": "statement",
        "expected_primary_intent": "reporting financial results",
    },
    {
        "input": "How do I reset my password if I no longer have access to my email?",
        "expected_category": "question",
        "expected_primary_intent": "account recovery help",
    },
    {
        "input": "Turn off the lights, lock the doors, and set the alarm to 6am.",
        "expected_category": "command",
        "expected_primary_intent": "home automation control",
    },
    {
        "input": (
            "I would appreciate it if you could review my pull request when you "
            "have a chance. It includes the bug fix we discussed yesterday."
        ),
        "expected_category": "request",
        "expected_primary_intent": "code review request",
    },
    {
        "input": (
            "The new office building will be located downtown and is expected "
            "to accommodate 500 employees across 10 floors."
        ),
        "expected_category": "statement",
        "expected_primary_intent": "providing information about office plans",
    },
]


# ---------------------------------------------------------------------------
# Build test cases by calling the live API
# ---------------------------------------------------------------------------

def build_intent_test_cases():
    test_cases = []
    for data in INTENT_TEST_DATA:
        response = detect_intent(data["input"])
        actual_output = json.dumps(response)
        expected_output = json.dumps({
            "primaryIntent": data["expected_primary_intent"],
            "secondaryIntents": [],
            "intentCategory": data["expected_category"],
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


intent_test_cases = build_intent_test_cases()
intent_dataset = EvaluationDataset()
for tc in intent_test_cases:
    intent_dataset.add_test_case(tc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

SCHEMA_DESC = (
    "The JSON must contain: 'primaryIntent' (string), 'secondaryIntents' "
    "(array of strings), 'intentCategory' (string: one of question, request, "
    "command, or statement), and 'confidence' (number between 0 and 1)."
)

intent_schema_metric = json_schema_metric(SCHEMA_DESC)

intent_category_metric = GEval(
    name="Intent Category Accuracy",
    criteria=(
        "Evaluate whether the intentCategory in the actual output correctly "
        "classifies the input text. Questions should be classified as "
        "'question', direct orders as 'command', polite asks as 'request', "
        "and factual declarations as 'statement'. Compare with the expected "
        "output to verify the category is correct."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
    model=JUDGE_MODEL,
)

intent_primary_metric = GEval(
    name="Primary Intent Accuracy",
    criteria=(
        "Evaluate whether the primaryIntent in the actual output accurately "
        "describes the main purpose behind the input text. The detected intent "
        "should capture what the user is trying to accomplish. Synonyms and "
        "semantically equivalent descriptions should be considered correct."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
    model=JUDGE_MODEL,
)

intent_relevancy_metric = answer_relevancy_metric()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_case", intent_dataset.test_cases)
def test_intent_schema_compliance(test_case: LLMTestCase):
    """Verify the intent response has the correct JSON structure."""
    assert_test(test_case, [intent_schema_metric])


@pytest.mark.parametrize("test_case", intent_dataset.test_cases)
def test_intent_category_accuracy(test_case: LLMTestCase):
    """Verify the intent category (question/command/request/statement) is correct."""
    assert_test(test_case, [intent_category_metric])


@pytest.mark.parametrize("test_case", intent_dataset.test_cases)
def test_intent_primary_accuracy(test_case: LLMTestCase):
    """Verify the primary intent description is accurate."""
    assert_test(test_case, [intent_primary_metric])


@pytest.mark.parametrize("test_case", intent_dataset.test_cases)
def test_intent_relevancy(test_case: LLMTestCase):
    """Verify the intent response is relevant to the input."""
    assert_test(test_case, [intent_relevancy_metric])
