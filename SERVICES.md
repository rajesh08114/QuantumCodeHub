# Services and Features Guide

This file explains how the backend works internally and how to control behavior for production.

## 1. Service Overview

- `services/llm_service.py`
  - Ordered fallback provider execution.
  - LangChain message + runnable orchestration.
  - LangChain `ChatOllama` path with HTTP fallback.
  - Provider chain is environment-configurable (commonly `hf_qwen_api`).
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
- `services/code_evaluation_service.py`
  - LLM + RAG-assisted evaluator for non-rule-based validation.
  - Signal-aware contradiction filtering to suppress false warnings.
  - Hallucination guard that keeps warnings/issues only when grounded in code or RAG evidence.
- `services/modernization_service.py`
  - Cross-framework deprecated API rewrite pass.
  - Applies only when validation quality improves.
- `services/adaptive_routing_service.py`
  - RL-inspired epsilon-greedy provider routing with EMA reward updates.
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

Routers that generate/transform code also accept optional `runtime_preferences`:
- `mode` (`auto`, `modern`, `legacy`)
- `python_version`
- `framework_version`
- `package_versions`
- `allow_deprecated_apis`

### 2.2 Processing
`build_runtime_bundle_with_rag()`:
1. Normalizes client type.
2. Normalizes user-requested runtime preferences.
3. Retrieves version-related docs from RAG.
4. Uses LLM to synthesize structured runtime recommendations from retrieved docs.
5. Validates suggested versions against retrieved docs (token-overlap evidence check).
6. Merges requested runtime with validated recommendations into effective runtime target.
7. Produces:
   - `compatibility_context` for prompts
   - `rag_query_suffix` for retrieval
   - `requested_runtime`
   - `effective_runtime_target`
   - `runtime_recommendations` for response metadata (effective target used by generation)
   - `version_conflicts`
   - `runtime_validation`
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

## 7. Runtime Recommendation Controls

Runtime recommendations are now derived from retrieved documentation plus LLM synthesis, then validated against those docs.

Operational notes:
- If RAG has weak or missing version docs, `runtime_recommendations` may be empty.
- `runtime_validation.status` explains whether recommendations were validated.
- Conflict checks run only on validated version suggestions.

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

## 9. Validation + Modernization Controls

Validation controls:
- `VALIDATION_MODE`
- `VALIDATION_ENABLE_LLM_EVAL`
- `VALIDATION_LLM_MAX_TOKENS`
- `VALIDATION_LLM_ON_STATIC_PASS_ONLY`
- `VALIDATION_MAX_CODE_CHARS`
- `VALIDATION_MAX_RAG_CHARS`
- `VALIDATION_MAX_COMPATIBILITY_CHARS`
- `VALIDATION_USE_RAG`
- `VALIDATION_RAG_TOP_K`
- `VALIDATION_RAG_AUGMENT_WHEN_CONTEXT_PRESENT`
- `VALIDATION_FAIL_ON_LLM_CRITICAL`
- `VALIDATION_REQUIRE_LLM_PASS`
- `VALIDATION_HALLUCINATION_GUARD_ENABLED`
- `VALIDATION_HALLUCINATION_MIN_CODE_OVERLAP`
- `VALIDATION_HALLUCINATION_MIN_RAG_OVERLAP`
- `VALIDATION_HALLUCINATION_MAX_DROPPED_REPORT`

Modernization controls:
- `ENABLE_MODERNIZATION_REPAIR`
- `MODERNIZATION_ON_DEPRECATION`
- `MODERNIZATION_STRICT`
- `MODERNIZATION_MAX_TOKENS`
- `MODERNIZATION_APPLY_ON_GENERATE`
- `MODERNIZATION_APPLY_ON_TRANSPILE`
- `MODERNIZATION_APPLY_ON_FIX`

Endpoint behavior:
- Generate/transpile/fix routes validate output.
- If deprecated APIs are detected, modernization rewrite is attempted.
- If `runtime_preferences.mode=legacy` or `allow_deprecated_apis=true`, modernization rewrite is skipped.
- Rewritten code is revalidated before acceptance.

## 10. Adaptive Routing Controls

- `ENABLE_ADAPTIVE_ROUTING`
- `ADAPTIVE_ROUTING_EPSILON`
- `ADAPTIVE_ROUTING_ALPHA`
- `ADAPTIVE_ROUTING_TARGET_LATENCY_MS`

Reward blend uses validation pass + confidence + latency.

## 11. Chat Memory Controls

- `CHAT_ENABLE_SESSION_MEMORY`
- `CHAT_CONTEXT_MESSAGES`
- `CHAT_MAX_MESSAGES_PER_SESSION`
- `CHAT_MAX_SESSIONS_PER_USER`
- `CHAT_CROSS_SESSION_SUMMARY_COUNT`
- `CHAT_SESSION_SUMMARY_MAX_CHARS`
- `CHAT_MEMORY_QUERY_MAX_CHARS`

When auth is disabled (`ENABLE_AUTH=false`), persistent memory is effectively unavailable because user id is absent.

## 12. Logging Controls

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

## 13. Cleanup Performed

Unused code cleanup included:
- removed unused `UserLogin` model in `api/routers/auth.py`
- removed unused `asyncpg` import in `api/routers/auth.py`
- removed unused local variable in `services/transpiler_service.py`
- replaced unused dependency variable names with `_current_user` in multiple routers
