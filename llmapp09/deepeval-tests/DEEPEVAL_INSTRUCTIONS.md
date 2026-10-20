# Project Structure

  deepeval-tests/
  ├── requirements.txt        # Python dependencies
  ├── api_client.py           # HTTP client for all 4 API endpoints
  ├── conftest.py             # Shared metric factories and fixtures
  ├── test_classify.py        # 5 inputs × 3 metrics = 15 tests
  ├── test_sentiment.py       # 5 inputs × 4 metrics = 20 tests
  ├── test_summarize.py       # 5 inputs × 5 metrics = 25 tests
  └── test_intent.py          # 8 inputs × 4 metrics = 32 tests
  
  Total: 92 evaluation tests across all 4 endpoints.
  
  Endpoints Covered
  Endpoint: POST /api/ai/classify
  Test File: test_classify.py
  Inputs: 5 (tech, sports, food, finance, health)
  Metrics: Schema, Correctness, Relevancy
  ────────────────────────────────────────
  Endpoint: POST /api/ai/sentiment
  Test File: test_sentiment.py
  Inputs: 5 (positive, negative, neutral, mixed, positive)
  Metrics: Schema, Correctness, Emotion Detection, Relevancy
  ────────────────────────────────────────
  Endpoint: POST /api/ai/summarize
  Test File: test_summarize.py
  Inputs: 5 (AI/healthcare, climate, remote work, JWST, EVs)
  Metrics: Schema, Correctness, Conciseness, Faithfulness, Relevancy
  ────────────────────────────────────────
  Endpoint: POST /api/ai/intent
  Test File: test_intent.py
  Inputs: 8 (2× question, command, request, statement each)
  Metrics: Schema, Category Accuracy, Primary Intent, Relevancy
  Metrics Used

  - JSON Schema Compliance (GEval) -- validates response structure matches the DTO
  - Output Correctness (GEval) -- validates the analysis is accurate for the input
  - Answer Relevancy (AnswerRelevancyMetric) -- validates response relevance to input
  - Endpoint-specific metrics (GEval):
    - Classification: label/category accuracy
    - Sentiment: emotion detection accuracy
    - Summarize: conciseness + faithfulness (no hallucinated facts)
    - Intent: category accuracy + primary intent accuracy

  # How to Run

  cd /Users/Darryl/Downloads/projects/llmapp04/deepeval-tests

  # 1. Install dependencies
  pip install -r requirements.txt

  # 2. Configure the judge LLM (LLM-as-a-judge). Defaults to a local Ollama model.
  ollama pull llama3.2                                  # or any local model
  export DEEPEVAL_JUDGE_MODEL="llama3.2"                # default: llama3.2
  export DEEPEVAL_JUDGE_BASE_URL="http://localhost:11434"  # default
  # Slow local models: per-call timeout defaults to 600s in conftest.py
  # export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=900

  # To use OpenAI as the judge instead:
  # export DEEPEVAL_JUDGE_PROVIDER=openai
  # export OPENAI_API_KEY="your-openai-api-key"

  # 3. Make sure the Spring Boot app is running on localhost:8080

  # 4. Run all tests
  deepeval test run test_classify.py test_sentiment.py test_summarize.py
  test_intent.py

  # Or run a single endpoint's tests
  deepeval test run test_classify.py

  # Run in parallel for speed
  deepeval test run test_classify.py test_sentiment.py test_summarize.py
  test_intent.py -n 4

  # Verbose output
  deepeval test run test_classify.py -v

  The tests call each endpoint live on localhost:8080, then use the judge LLM (local Ollama by default, configured in conftest.py) to evaluate whether the responses are correct, relevant, properly structured, and free of hallucinations.