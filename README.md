# QuantumCodeHub Backend

FastAPI backend for quantum developer workflows:
- code generation
- transpilation
- completion
- explanation
- error fixing
- session-aware chat

## 1. What Changed

The backend LLM layer is now LangChain-oriented in `services/llm_service.py`:
- LangChain message objects are used for provider input shaping.
- LangChain `RunnableLambda` is used for provider execution wrappers.
- `ChatOllama` (LangChain) is used when available, with HTTP fallback preserved.
- Existing API contracts and response metadata are preserved (provider/model/tokens/attempt/fallback).

This keeps behavior stable while moving orchestration toward LangChain.

Validation and quality control now use a hybrid, framework-wide pipeline:
- static semantic validation for `qiskit`, `pennylane`, `cirq`, `torchquantum`
- framework-agnostic LLM evaluator for non-rule-based correctness checks
- one-pass automatic repair when generated code fails validation
- strict modernization pass that rewrites deprecated APIs to stable ones across frameworks
- RL-inspired adaptive provider routing (epsilon-greedy bandit) to improve success/latency over time
- explicit `runtime_preferences` in request body to target legacy or specific versions

## 2. API Surface

Main routes:
- `/api/auth`
- `/api/code`
- `/api/transpile`
- `/api/complete`
- `/api/explain`
- `/api/fix`
- `/api/chat`
- `/health`
- `/metrics`

## 3. Setup

### 3.1 Prerequisites

- Python `3.10+`
- PostgreSQL
- Redis
- Optional: Ollama (if you want local LLM provider)

### 3.2 Install

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3.3 Environment

```bash
copy .env.example .env
```

Set at minimum:
- `DATABASE_URL`
- `REDIS_URL`
- `SECRET_KEY`
- `ALLOWED_ORIGINS`
- `EMBEDDING_MODEL`
- one LLM option:
  - `HF_API_KEY` (HF Router), or
  - `RESP_API_URL`, or
  - `OLLAMA_BASE_URL` + `OLLAMA_MODEL`

Recommended quality controls:
- `VALIDATION_MODE=hybrid`
- `VALIDATION_ENABLE_LLM_EVAL=true`
- `VALIDATION_LLM_MAX_TOKENS=220`
- `VALIDATION_LLM_ON_STATIC_PASS_ONLY=true`
- `VALIDATION_USE_RAG=true`
- `VALIDATION_RAG_TOP_K=3`
- `VALIDATION_MAX_CODE_CHARS=2400`
- `VALIDATION_MAX_RAG_CHARS=1200`
- `VALIDATION_MAX_COMPATIBILITY_CHARS=700`
- `VALIDATION_FAIL_ON_LLM_CRITICAL=true`
- `VALIDATION_REQUIRE_LLM_PASS=false`
- `VALIDATION_HALLUCINATION_GUARD_ENABLED=true`
- `VALIDATION_HALLUCINATION_MIN_CODE_OVERLAP=0.22`
- `VALIDATION_HALLUCINATION_MIN_RAG_OVERLAP=0.18`
- `VALIDATION_HALLUCINATION_MAX_DROPPED_REPORT=8`
- `ENABLE_MODERNIZATION_REPAIR=true`
- `MODERNIZATION_ON_DEPRECATION=true`
- `MODERNIZATION_STRICT=true`
- `MODERNIZATION_MAX_TOKENS=700`
- `MODERNIZATION_APPLY_ON_GENERATE=true`
- `MODERNIZATION_APPLY_ON_TRANSPILE=true`
- `MODERNIZATION_APPLY_ON_FIX=true`
- `ENABLE_ADAPTIVE_ROUTING=true`
- `ADAPTIVE_ROUTING_EPSILON=0.08`
- `ADAPTIVE_ROUTING_ALPHA=0.25`
- `ADAPTIVE_ROUTING_TARGET_LATENCY_MS=3500`

### 3.4 Initialize Data Stores

```bash
python scripts/setup_db.py
python scripts/setup_chroma.py
```

### 3.5 Run API

```bash
uvicorn api.main:app --reload
```

Open:
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## 4. Auth Behavior

- `ENABLE_AUTH=true`: protected routes need `Authorization: Bearer <token>`.
- `ENABLE_AUTH=false`: protected routes can be called without JWT (test mode).

## 5. Postman Testing Guide

## 5.1 Base setup

1. Create Postman environment variable:
   - `base_url = http://127.0.0.1:8000`
2. If auth enabled, create `token` variable.

## 5.2 Get token

Request:
- `POST {{base_url}}/api/auth/login`
- Body type: `x-www-form-urlencoded`
  - `username`: your email
  - `password`: your password
  - `grant_type`: `password`

Copy `access_token` from response into `token`.

## 5.3 Protected request header

For protected endpoints add header:
- `Authorization: Bearer {{token}}`

## 5.4 Core endpoint examples

### `POST /api/code/generate`

```json
{
  "prompt": "Create a 3-qubit GHZ state",
  "framework": "qiskit",
  "include_explanation": true,
  "runtime_preferences": {
    "mode": "legacy",
    "python_version": "3.9",
    "package_versions": {
      "qiskit": "0.45.*",
      "qiskit-aer": "0.12.*"
    },
    "allow_deprecated_apis": true
  },
  "client_context": { "client_type": "website" }
}
```

`/api/code/generate` response metadata now includes:
- `requested_runtime`
- `runtime_requirements` (effective Python + package requirements used for generation)
- `runtime_recommendations` (same effective target, validated/merged with RAG evidence)
- `validation_evaluation` (static + LLM evaluator details)
- `validation_warnings`
- `auto_repair_attempted`
- `auto_repair_used`
- `modernization_attempted`
- `modernization_applied`
- `adaptive_preferred_chain`

### `POST /api/transpile/convert`

```json
{
  "source_code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)",
  "source_framework": "qiskit",
  "target_framework": "pennylane",
  "preserve_comments": true,
  "optimize": false,
  "runtime_preferences": {
    "mode": "legacy",
    "python_version": "3.10",
    "package_versions": { "pennylane": "0.33.*" }
  },
  "client_context": { "client_type": "api" }
}
```

### `POST /api/complete/suggest`

```json
{
  "code_prefix": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.",
  "framework": "qiskit",
  "cursor_line": 3,
  "cursor_column": 3,
  "max_suggestions": 5,
  "client_context": { "client_type": "vscode_extension" }
}
```

### `POST /api/explain/code`

```json
{
  "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)",
  "framework": "qiskit",
  "detail_level": "intermediate",
  "include_math": true,
  "client_context": { "client_type": "website" }
}
```

### `POST /api/fix/code`

```json
{
  "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0)",
  "framework": "qiskit",
  "error_message": "TypeError: cx() missing 1 required positional argument",
  "include_explanation": true,
  "client_context": { "client_type": "api" }
}
```

### `POST /api/chat/message`

```json
{
  "message": "Explain Bell state in simple terms",
  "framework": "qiskit",
  "detail_level": "beginner",
  "new_session": false,
  "include_other_session_summaries": false,
  "client_context": { "client_type": "website" }
}
```

## 6. Endpoint + Response-Time Test Script (`test.py`)

A root-level script `test.py` is included to run endpoint smoke tests and measure response times.

### 6.1 Run without auth

Use this if `ENABLE_AUTH=false`:

```bash
python test.py
```

If auto-detection is blocked by networking/proxy behavior, force it:

```bash
python test.py --auth-disabled
```

### 6.2 Run with JWT token

```bash
python test.py --token <JWT_TOKEN>
```

### 6.3 Run with login credentials (auto-login)

```bash
python test.py --email your@email.com --password YourPassword123!
```

### 6.4 Useful flags

```bash
python test.py --base-url http://127.0.0.1:8000 --timeout 90 --fail-on-error
```

- `--base-url`: API host
- `--timeout`: per-request timeout seconds
- `--fail-on-error`: exits with code `1` if any executed endpoint fails

The script prints:
- status per endpoint
- per-endpoint latency (ms)
- summary (passed/failed/skipped)
- average and p95 latency

## 7. Troubleshooting

- `401 Unauthorized`: missing/expired JWT or wrong auth mode.
- `500` on generation routes: verify at least one LLM provider is configured.
- RAG empty context / Chroma unhealthy: run `python scripts/setup_chroma.py` and verify `CHROMA_PERSIST_DIR`.
- DB startup failures: verify `DATABASE_URL` and run `python scripts/setup_db.py`.
- Validation too strict: set `VALIDATION_FAIL_ON_LLM_CRITICAL=false`.
- Validation too slow: lower `VALIDATION_LLM_MAX_TOKENS` or set `VALIDATION_ENABLE_LLM_EVAL=false`.
- Adaptive routing too unstable: reduce `ADAPTIVE_ROUTING_EPSILON` (less exploration).
- If deprecated APIs are still present: ensure `ENABLE_MODERNIZATION_REPAIR=true` and `MODERNIZATION_ON_DEPRECATION=true`.

## 8. Service Notes

More internals and runtime knobs are documented in `SERVICES.md`.

## 9. Validation Architecture

Hybrid validation flow (`services/validator_service.py`):
1. Framework semantic validator
2. Validation-time RAG retrieval (framework docs + API usage patterns)
3. LLM evaluator (`services/code_evaluation_service.py`) using code signals + RAG context
4. Aggregated pass/fail with critical issue gating

Runtime-targeted generation:
1. Provide `runtime_preferences` in request body.
2. Runtime bundle merges requested versions with RAG+LLM validated recommendations.
3. Validation and generation are constrained by this runtime target.
4. Modernization rewrite is skipped when legacy runtime is explicitly requested.

Framework validators:
- `ml/validators/qiskit_validator.py`
- `ml/validators/pennylane_validator.py`
- `ml/validators/cirq_validator.py`
- `ml/validators/torchquantum_validator.py`

Validation is applied to generated code and transpiled output (target framework validation).

Auto-repair:
- Implemented in `api/routers/code_generation.py`
- Triggered when validation fails
- Revalidates repaired output before returning

Modernization pass:
- Implemented in `services/modernization_service.py`
- Triggered by deprecation/legacy warnings
- Applied on generate/transpile/fix routes (configurable)
- Revalidates rewritten code and applies only when quality improves

Adaptive routing (RL-inspired):
- Implemented in `services/adaptive_routing_service.py`
- Uses epsilon-greedy exploration + EMA reward updates
- Reward blends validation pass, confidence, and latency

## 10. End-to-End Quantum Code Flow

`POST /api/code/generate` pipeline:
1. Runtime compatibility bundle from `RAG + LLM + doc-grounded validation`
   - includes user `runtime_preferences`
   - computes `runtime_requirements` (effective target versions)
2. Main RAG retrieval for generation context
3. LLM code generation (adaptive routing can reorder provider chain)
4. Hybrid validation:
   - static framework validator
   - optional validation-specific RAG retrieval
   - LLM evaluator with signal-aware contradiction filtering
   - hallucination guard that drops low-evidence warnings/issues
5. Optional one-pass auto-repair if validation fails
6. Optional strict modernization rewrite to remove deprecated APIs
7. Confidence scoring + metadata emission

Bottleneck fix for false warnings:
- The evaluator now suppresses contradictory warnings (example: "missing measurement" is ignored when code signals confirm measurement APIs are present).
- The evaluator drops warnings/critical issues with insufficient code+RAG grounding evidence.
- LLM evaluation only fails on definitive runtime/correctness issues.
- Evaluation prompt context is clipped to reduce token usage and latency.
