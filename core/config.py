"""
Configuration management.
"""
"""
Application configuration using Pydantic Settings.
Loads environment variables from .env file automatically.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # -----------------------
    # App Settings
    # -----------------------
    APP_NAME: str
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # -----------------------
    # Database
    # -----------------------
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # -----------------------
    # Redis
    # -----------------------
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = 50

    # -----------------------
    # Authentication
    # -----------------------
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ENABLE_AUTH: bool = False
    INTERNAL_API_KEY: str = ""

    # -----------------------
    # LLM Service
    # -----------------------
    HF_API_KEY: str = ""
    HF_INFERENCE_URL: str = "https://router.huggingface.co/v1"
    HF_MODEL_ID: str = "ibm-granite/granite-4.0-h-small"
    HF_QWEN_MODEL_ID: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    LLM_BACKEND: str = "api"  # "api", "local", or "ollama"
    LLM_PROVIDER_CHAIN: str = ""  # e.g. "hf_api,resp_api,ollama,hf_qwen_api"
    RESP_API_URL: str = ""
    RESP_API_KEY: str = ""
    RESP_API_MODEL: str = ""
    LLM_REQUEST_TIMEOUT_SECONDS: int = 45
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    LOCAL_MODEL_ID: str = "ibm-granite/granite-4.0-h-small"
    LOCAL_MODEL_PATH: str = ""  # Optional filesystem path to local model
    LOCAL_MODEL_FILES_ONLY: bool = False
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    # Token guidance:
    # least: 128-256, balanced: 512-900, best quality: 1200-2200
    GENERATION_MAX_TOKENS: int = 1500
    TRANSPILATION_MAX_TOKENS: int = 2000
    EXPLANATION_MAX_TOKENS: int = 1500
    ERROR_FIXING_MAX_TOKENS: int = 1800
    COMPLETION_MAX_TOKENS: int = 300
    CHAT_GENERAL_MAX_TOKENS: int = 900
    SUPPORTED_PYTHON_VERSION: str = "3.10-3.12"
    SUPPORTED_QISKIT_VERSION: str = ">=1.2,<2.0"
    SUPPORTED_QISKIT_AER_VERSION: str = ">=0.14,<0.16"
    SUPPORTED_PENNYLANE_VERSION: str = ">=0.36,<0.39"
    SUPPORTED_CIRQ_VERSION: str = ">=1.3,<1.5"
    SUPPORTED_TORCHQUANTUM_VERSION: str = ">=0.1,<0.2"
    SUPPORTED_TORCH_VERSION: str = ">=2.1,<2.4"

    # -----------------------
    # RAG
    # -----------------------
    EMBEDDING_MODEL: str
    RAG_TOP_K: int = 5
    CHROMA_PERSIST_DIR: str = "Rag_pipeline/chroma_db"
    CHROMA_COLLECTION_NAME: str = "quantum_knowledge"
    RAG_DEFAULT_TO_LATEST_VERSION: bool = True
    RAG_LATEST_VERSION_CACHE_TTL_SECONDS: int = 600
    RAG_LEGACY_MODE_ALLOW_ALL_VERSIONS: bool = True
    RAG_STRICT_VERSION_SELECTION: bool = True
    RAG_TRAFFIC_PROFILE: str = "auto"  # auto | code_heavy | balanced | conceptual
    RAG_FETCH_MULTIPLIER: int = 12
    RAG_MAX_FETCH_RESULTS: int = 200
    RAG_QUERY_VARIANTS: int = 3
    RAG_MAX_DOCS_PER_SOURCE: int = 2
    RAG_MAX_DOC_CHARS: int = 1200
    RAG_MAX_CONTEXT_CHARS: int = 9000
    RAG_SEMANTIC_WEIGHT: float = 0.65
    RAG_LEXICAL_WEIGHT: float = 0.25
    RAG_RRF_WEIGHT: float = 0.10
    RAG_FRAMEWORK_BOOST: float = 0.08
    RAG_RRF_K: int = 60

    # -----------------------
    # CORS
    # -----------------------
    ALLOWED_ORIGINS: str  # comma-separated in .env

    # -----------------------
    # Monitoring
    # -----------------------
    SENTRY_DSN: str
    SENTRY_ENVIRONMENT: str = "production"

    # -----------------------
    # Rate Limiting
    # -----------------------
    RATE_LIMIT_FREE: int = 50
    RATE_LIMIT_PRO: int = 500
    RATE_LIMIT_TEAM: int = 2000

    # -----------------------
    # Feature Flags
    # -----------------------
    ENABLE_CACHE: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_RATE_LIMITING: bool = True
    LOG_HTTP_REQUEST_BODY: bool = True
    LOG_HTTP_RESPONSE_BODY: bool = True
    LOG_HTTP_BODY_MAX_CHARS: int = 2000
    LOG_HTTP_EXCLUDE_PATHS: str = "/metrics"
    CHAT_ENABLE_SESSION_MEMORY: bool = True
    CHAT_CONTEXT_MESSAGES: int = 8
    CHAT_MAX_MESSAGES_PER_SESSION: int = 120
    CHAT_MAX_SESSIONS_PER_USER: int = 100
    CHAT_CROSS_SESSION_SUMMARY_COUNT: int = 3
    CHAT_SESSION_SUMMARY_MAX_CHARS: int = 1200
    CHAT_MEMORY_QUERY_MAX_CHARS: int = 1200
    VALIDATION_MODE: str = "hybrid"  # static | llm | hybrid
    VALIDATION_ENABLE_LLM_EVAL: bool = True
    VALIDATION_LLM_MAX_TOKENS: int = 220
    VALIDATION_LLM_ON_STATIC_PASS_ONLY: bool = True
    VALIDATION_MAX_CODE_CHARS: int = 2400
    VALIDATION_MAX_RAG_CHARS: int = 1200
    VALIDATION_MAX_COMPATIBILITY_CHARS: int = 700
    VALIDATION_USE_RAG: bool = True
    VALIDATION_RAG_TOP_K: int = 3
    VALIDATION_RAG_AUGMENT_WHEN_CONTEXT_PRESENT: bool = False
    VALIDATION_FAIL_ON_LLM_CRITICAL: bool = True
    VALIDATION_REQUIRE_LLM_PASS: bool = False
    VALIDATION_HALLUCINATION_GUARD_ENABLED: bool = True
    VALIDATION_HALLUCINATION_MIN_CODE_OVERLAP: float = 0.22
    VALIDATION_HALLUCINATION_MIN_RAG_OVERLAP: float = 0.18
    VALIDATION_HALLUCINATION_MAX_DROPPED_REPORT: int = 8
    ENABLE_MODERNIZATION_REPAIR: bool = True
    MODERNIZATION_ON_DEPRECATION: bool = True
    MODERNIZATION_STRICT: bool = True
    MODERNIZATION_MAX_TOKENS: int = 700
    MODERNIZATION_APPLY_ON_GENERATE: bool = True
    MODERNIZATION_APPLY_ON_TRANSPILE: bool = True
    MODERNIZATION_APPLY_ON_FIX: bool = True
    ENABLE_ADAPTIVE_ROUTING: bool = True
    ADAPTIVE_ROUTING_EPSILON: float = 0.08
    ADAPTIVE_ROUTING_ALPHA: float = 0.25
    ADAPTIVE_ROUTING_TARGET_LATENCY_MS: int = 3500

    # -----------------------
    # Pydantic Config
    # -----------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # -----------------------
    # Helper Properties
    # -----------------------
    @property
    def cors_origins(self) -> List[str]:
        origins: List[str] = []
        for origin in (self.ALLOWED_ORIGINS or "").split(","):
            value = origin.strip()
            if value and value not in origins:
                origins.append(value)

        if (self.ENVIRONMENT or "").strip().lower() in {"development", "dev", "local"}:
            for local_origin in (
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ):
                if local_origin not in origins:
                    origins.append(local_origin)

        return origins


# Singleton settings instance
settings = Settings()
