# Services and Features Guide

This file explains how the backend works internally and how to control behavior for production.

## 1. Service Overview

- `services/llm_service.py`
  - Ordered fallback provider execution.
  - LangChain message + runnable orchestration.
  - LangChain `ChatOllama` path with HTTP fallback.
  - Default chain: `hf_api -> resp_api -> ollama -> hf_qwen_api`.
- `services/rag_service.py`
  - Chroma retrieval with multi-query + hybrid reranking + diversity constraints.
- `services/runtime_compatibility.py`
  - New compatibility engine for VS Code and website client contexts.
  - Generates:
    - prompt compatibility block
    - retrieval compatibility suffix
    - runtime recommendations
    - conflict warnings
    - cache fingerprint
- `services/transpiler_service.py`
  - Framework conversion generation.
- `services/chat_memory_service.py`
  - Session persistence + summary memory.
- `services/cache_service.py`
  - Redis caching.
- `services/quota_service.py`
  - Quota/rate limits.

## 2. Client-Aware Compatibility Flow

### 2.1 Input Contract
Routers accept optional `client_context`:
- `client_type` (`vscode_extension`, `website`, `api`)
- `python_version`
- `framework_version`
- `installed_packages`
- extension metadata (for VS Code)

### 2.2 Processing
`build_runtime_bundle()`:
1. Normalizes client type.
2. Loads framework support matrix from `core/config.py`.
3. Compares installed package majors to supported ranges.
4. Produces:
   - `compatibility_context` for prompts
   - `rag_query_suffix` for retrieval
   - `runtime_recommendations` for response metadata
   - `version_conflicts`
   - `cache_fingerprint` for compatibility-safe cache keys

### 2.3 Router Coverage
Integrated into:
- `/api/code/generate`
- `/api/transpile/convert`
- `/api/complete/suggest`
- `/api/explain/code`
- `/api/fix/code`
- `/api/chat/message`

## 3. VS Code Extension Behavior

Expected behavior:
1. Send local Python + package versions in `client_context`.
2. Read `metadata.version_conflicts`.
3. If conflicts exist, show warning before running generated code.
4. Use `metadata.runtime_recommendations` to offer quick install/update commands.

## 4. Website Behavior

Expected behavior:
1. Send `client_type=website`.
2. Always render `runtime_recommendations` beside generated code.
3. Show warnings when conflicts are returned.
4. Store returned recommendations with shared snippets or export payloads.

## 5. LLM Controls

Config:
- `LLM_PROVIDER_CHAIN`
- `HF_*`, `RESP_API_*`, `OLLAMA_*`, `LOCAL_*`
- `LLM_REQUEST_TIMEOUT_SECONDS`
- `MAX_TOKENS`, `TEMPERATURE`

Operational notes:
- Provider attempts/failures are logged.
- Response metadata exposes provider/model/attempt/fallback.
- `get_routing_info()` also reports LangChain capability flags.

## 6. RAG Controls

Config:
- `RAG_TRAFFIC_PROFILE`
- `RAG_FETCH_MULTIPLIER`
- `RAG_MAX_FETCH_RESULTS`
- `RAG_QUERY_VARIANTS`
- `RAG_MAX_DOCS_PER_SOURCE`
- `RAG_MAX_DOC_CHARS`
- `RAG_MAX_CONTEXT_CHARS`
- `RAG_SEMANTIC_WEIGHT`
- `RAG_LEXICAL_WEIGHT`
- `RAG_RRF_WEIGHT`
- `RAG_FRAMEWORK_BOOST`
- `RAG_RRF_K`

Compatibility-aware retrieval:
- `rag_query_suffix` is appended so retrieval includes version constraints.

## 7. Runtime Support Matrix Controls

Config keys:
- `SUPPORTED_PYTHON_VERSION`
- `SUPPORTED_QISKIT_VERSION`
- `SUPPORTED_QISKIT_AER_VERSION`
- `SUPPORTED_PENNYLANE_VERSION`
- `SUPPORTED_CIRQ_VERSION`
- `SUPPORTED_TORCHQUANTUM_VERSION`
- `SUPPORTED_TORCH_VERSION`

Update these when your supported runtime baseline changes.

## 8. Token Controls

Per-feature token ceilings:
- `GENERATION_MAX_TOKENS`
- `TRANSPILATION_MAX_TOKENS`
- `EXPLANATION_MAX_TOKENS`
- `ERROR_FIXING_MAX_TOKENS`
- `COMPLETION_MAX_TOKENS`
- `CHAT_GENERAL_MAX_TOKENS`

Guideline:
- least: `128-256`
- balanced: `512-900`
- best quality: `1200-2200`

## 9. Chat Memory Controls

- `CHAT_ENABLE_SESSION_MEMORY`
- `CHAT_CONTEXT_MESSAGES`
- `CHAT_MAX_MESSAGES_PER_SESSION`
- `CHAT_MAX_SESSIONS_PER_USER`
- `CHAT_CROSS_SESSION_SUMMARY_COUNT`
- `CHAT_SESSION_SUMMARY_MAX_CHARS`
- `CHAT_MEMORY_QUERY_MAX_CHARS`

When auth is disabled (`ENABLE_AUTH=false`), persistent memory is effectively unavailable because user id is absent.

## 10. Logging Controls

- `LOG_LEVEL`
- `LOG_HTTP_REQUEST_BODY`
- `LOG_HTTP_RESPONSE_BODY`
- `LOG_HTTP_BODY_MAX_CHARS`
- `LOG_HTTP_EXCLUDE_PATHS`

Logs include:
- request/response correlation id
- sanitized headers/bodies
- LLM provider attempt chain
- RAG retrieval plan/results
- quota/rate-limit decisions

## 11. Cleanup Performed

Unused code cleanup included:
- removed unused `UserLogin` model in `api/routers/auth.py`
- removed unused `asyncpg` import in `api/routers/auth.py`
- removed unused local variable in `services/transpiler_service.py`
- replaced unused dependency variable names with `_current_user` in multiple routers
