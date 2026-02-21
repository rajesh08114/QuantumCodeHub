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
  "client_context": { "client_type": "website" }
}
```

### `POST /api/transpile/convert`

```json
{
  "source_code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)",
  "source_framework": "qiskit",
  "target_framework": "pennylane",
  "preserve_comments": true,
  "optimize": false,
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

## 8. Service Notes

More internals and runtime knobs are documented in `SERVICES.md`.
