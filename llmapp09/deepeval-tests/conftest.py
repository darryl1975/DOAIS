"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import os

# Local judge models are slow; raise deepeval's per-call timeout (~90s default)
# before deepeval loads its settings. Override by exporting the variable.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

import pytest
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Judge model (LLM-as-a-judge) configuration
# ---------------------------------------------------------------------------
# By default the GEval metrics are scored by a local Ollama model. Override via:
#   DEEPEVAL_JUDGE_PROVIDER  "ollama" (default) or "openai" (uses OPENAI_API_KEY)
#   DEEPEVAL_JUDGE_MODEL     Ollama model name (default: qwen2.5:7b)
#   DEEPEVAL_JUDGE_BASE_URL  Ollama server URL (default: http://localhost:11434)
# Distinct names are used so they don't clash with OLLAMA_BASE_URL, which points
# the backend at Ollama Cloud.
# Use a model of ~7B+ parameters: 1-3B models (llama3.2:1b, gemma2:2b,
# deepseek-r1:1.5b) can't follow GEval's scoring prompt and return arbitrary
# scores, e.g. failing valid JSON as "labels is not an array of strings".
def get_judge_model():
    """Returns the judge model passed to every metric (None = deepeval default)."""
    provider = os.getenv("DEEPEVAL_JUDGE_PROVIDER", "ollama").lower()
    if provider == "openai":
        return None
    return OllamaModel(
        model=os.getenv("DEEPEVAL_JUDGE_MODEL", "glm-5.2:cloud"),
        base_url=os.getenv("DEEPEVAL_JUDGE_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
    )


JUDGE_MODEL = get_judge_model()


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=JUDGE_MODEL,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=JUDGE_MODEL,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=JUDGE_MODEL,
    )
