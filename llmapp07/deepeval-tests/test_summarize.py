"""
DeepEval LLM evaluation tests for POST /api/ai/summarize.

Evaluates whether the summarization endpoint returns concise, accurate
summaries with relevant key points.
"""

import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

from api_client import summarize_text
from conftest import JUDGE_MODEL, json_schema_metric, output_correctness_metric, answer_relevancy_metric


# ---------------------------------------------------------------------------
# Test inputs and expected behaviors
# ---------------------------------------------------------------------------

SUMMARIZE_TEST_DATA = [
    {
        "input": (
            "Artificial intelligence has transformed the healthcare industry in "
            "numerous ways. Machine learning algorithms can now detect diseases "
            "from medical images with accuracy rivaling human doctors. Natural "
            "language processing helps extract insights from clinical notes and "
            "research papers. Predictive models identify patients at risk of "
            "readmission, enabling proactive interventions. Drug discovery has "
            "been accelerated through AI-powered molecular simulations. Despite "
            "these advances, challenges remain in data privacy, algorithmic bias, "
            "and regulatory approval for AI-based medical devices."
        ),
        "expected_key_topics": [
            "AI in healthcare",
            "disease detection",
            "drug discovery",
            "challenges",
        ],
    },
    {
        "input": (
            "Climate change is causing rising sea levels worldwide. Scientists "
            "have observed that Arctic ice sheets are melting at an unprecedented "
            "rate, contributing to coastal flooding in many regions. Small island "
            "nations are particularly vulnerable, with some facing the prospect "
            "of becoming uninhabitable within decades. International agreements "
            "like the Paris Accord aim to limit global warming to 1.5 degrees "
            "Celsius, but many experts argue current commitments are insufficient."
        ),
        "expected_key_topics": [
            "rising sea levels",
            "Arctic ice melting",
            "island nations at risk",
            "Paris Accord",
        ],
    },
    {
        "input": (
            "Remote work has fundamentally changed how companies operate. "
            "Employees report higher satisfaction due to flexible schedules and "
            "elimination of commuting. However, managers face challenges in "
            "maintaining team cohesion and monitoring productivity. Companies "
            "are investing in collaboration tools like Slack, Zoom, and project "
            "management software. The hybrid model, combining remote and "
            "in-office work, has emerged as a popular compromise. Studies show "
            "that remote workers are equally or more productive than their "
            "in-office counterparts."
        ),
        "expected_key_topics": [
            "remote work impact",
            "employee satisfaction",
            "management challenges",
            "hybrid model",
        ],
    },
    {
        "input": (
            "The James Webb Space Telescope has captured unprecedented images "
            "of distant galaxies, nebulae, and exoplanets. Launched in December "
            "2021, JWST orbits the Sun at the second Lagrange point, about 1.5 "
            "million kilometers from Earth. Its infrared capabilities allow "
            "scientists to observe objects that are too faint or distant for "
            "previous telescopes. Recent discoveries include the detection of "
            "carbon dioxide in an exoplanet atmosphere and the observation of "
            "some of the earliest galaxies formed after the Big Bang."
        ),
        "expected_key_topics": [
            "James Webb Space Telescope",
            "distant galaxies",
            "infrared capabilities",
            "exoplanet discoveries",
        ],
    },
    {
        "input": (
            "Electric vehicles are gaining market share rapidly. Tesla remains "
            "the market leader, but traditional automakers like Ford, GM, and "
            "Volkswagen are launching competitive EV models. Battery technology "
            "continues to improve, with solid-state batteries promising faster "
            "charging and longer range. Government incentives and stricter "
            "emission standards are accelerating adoption. Charging "
            "infrastructure is expanding, though rural areas still lack "
            "adequate coverage."
        ),
        "expected_key_topics": [
            "EV market growth",
            "battery technology",
            "government incentives",
            "charging infrastructure",
        ],
    },
]


# ---------------------------------------------------------------------------
# Build test cases by calling the live API
# ---------------------------------------------------------------------------

def build_summarize_test_cases():
    test_cases = []
    for data in SUMMARIZE_TEST_DATA:
        response = summarize_text(data["input"])
        actual_output = json.dumps(response)
        expected_output = json.dumps({
            "summary": "A concise summary covering: " + ", ".join(data["expected_key_topics"]),
            "keyPoints": data["expected_key_topics"],
            "wordCount": 30,
        })
        test_cases.append(
            LLMTestCase(
                input=data["input"],
                actual_output=actual_output,
                expected_output=expected_output,
            )
        )
    return test_cases


summarize_test_cases = build_summarize_test_cases()
summarize_dataset = EvaluationDataset()
for tc in summarize_test_cases:
    summarize_dataset.add_test_case(tc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

SCHEMA_DESC = (
    "The JSON must contain: 'summary' (string), 'keyPoints' (array of "
    "strings), and 'wordCount' (integer)."
)

summarize_schema_metric = json_schema_metric(SCHEMA_DESC)

summarize_correctness_metric = GEval(
    name="Summary Correctness",
    criteria=(
        "Evaluate whether the summary accurately captures the main ideas of "
        "the input text without introducing information that is not present "
        "in the original. The summary should be concise and not miss major "
        "topics. The key points should reflect the most important aspects of "
        "the input text."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=JUDGE_MODEL,
)

summarize_conciseness_metric = GEval(
    name="Summary Conciseness",
    criteria=(
        "Evaluate whether the summary is concise and avoids unnecessary "
        "verbosity. A good summary should be significantly shorter than the "
        "input text while retaining all essential information. The wordCount "
        "field should be reasonable relative to the input length."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model=JUDGE_MODEL,
)

summarize_faithfulness_metric = GEval(
    name="Summary Faithfulness",
    criteria=(
        "Evaluate whether every claim in the summary is supported by the "
        "original input text. The summary should not contain any hallucinated "
        "facts, statistics, or claims that are not present in the input. "
        "Penalize any fabricated information heavily."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.8,
    model=JUDGE_MODEL,
)

summarize_relevancy_metric = answer_relevancy_metric()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_case", summarize_dataset.test_cases)
def test_summarize_schema_compliance(test_case: LLMTestCase):
    """Verify the summary response has the correct JSON structure."""
    assert_test(test_case, [summarize_schema_metric])


@pytest.mark.parametrize("test_case", summarize_dataset.test_cases)
def test_summarize_correctness(test_case: LLMTestCase):
    """Verify the summary captures the main ideas accurately."""
    assert_test(test_case, [summarize_correctness_metric])


@pytest.mark.parametrize("test_case", summarize_dataset.test_cases)
def test_summarize_conciseness(test_case: LLMTestCase):
    """Verify the summary is concise relative to the input."""
    assert_test(test_case, [summarize_conciseness_metric])


@pytest.mark.parametrize("test_case", summarize_dataset.test_cases)
def test_summarize_faithfulness(test_case: LLMTestCase):
    """Verify the summary does not hallucinate facts not in the input."""
    assert_test(test_case, [summarize_faithfulness_metric])


@pytest.mark.parametrize("test_case", summarize_dataset.test_cases)
def test_summarize_relevancy(test_case: LLMTestCase):
    """Verify the summary response is relevant to the input."""
    assert_test(test_case, [summarize_relevancy_metric])
