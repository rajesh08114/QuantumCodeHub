"""
FastAPI app initialization.
"""
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.middleware import rate_limit_middleware, request_response_logging_middleware
from api.routers import auth, chatbot, code_generation, completion, error_fixing, explanation, transpilation
from core.config import settings
from core.database import close_db, connect_to_db, get_db_connection, release_db_connection
from services.cache_service import cache_service
from services.chat_memory_service import chat_memory_service
from services.llm_service import llm_service
from services.quota_service import quota_service
from services.rag_service import rag_service

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Initialize app logging so service-level INFO logs are visible."""
    level_name = (settings.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("services.rag_service").setLevel(level)
    logging.getLogger("services.llm_service").setLevel(level)


_configure_logging()

app = FastAPI(
    title="QuantumCodeHub API",
    version="1.0.0",
    description="AI-powered quantum code generation and assistance",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP middleware
app.middleware("http")(request_response_logging_middleware)
app.middleware("http")(rate_limit_middleware)

# Routers
app.include_router(auth.router)
app.include_router(code_generation.router)
app.include_router(transpilation.router)
app.include_router(completion.router)
app.include_router(explanation.router)
app.include_router(error_fixing.router)
app.include_router(chatbot.router)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await connect_to_db()
    await cache_service.connect()
    await quota_service.connect()
    await chat_memory_service.init_schema()
    logger.info("QuantumCodeHub API started")
    logger.info("LLM routing info: %s", llm_service.get_routing_info())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await close_db()
    logger.info("QuantumCodeHub API shutting down")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns system health status and component availability.
    """
    health_status = {"status": "healthy", "components": {}}

    # Check database (fail-fast, uses existing pool)
    db_conn = None
    try:
        db_conn = await asyncio.wait_for(get_db_connection(), timeout=2.0)
        await asyncio.wait_for(db_conn.fetchval("SELECT 1"), timeout=2.0)
        health_status["components"]["database"] = "healthy"
    except asyncio.TimeoutError:
        health_status["components"]["database"] = "unhealthy: timeout"
        health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    finally:
        await release_db_connection(db_conn)

    # Check Redis (fail-fast)
    try:
        if cache_service.redis_client:
            await asyncio.wait_for(cache_service.redis_client.ping(), timeout=1.5)
            health_status["components"]["redis"] = "healthy"
        else:
            health_status["components"]["redis"] = "not connected"
    except asyncio.TimeoutError:
        health_status["components"]["redis"] = "unhealthy: timeout"
    except Exception as e:
        health_status["components"]["redis"] = f"unhealthy: {str(e)}"

    # Check LLM service
    health_status["components"]["llm"] = llm_service.get_routing_info()

    # Check Chroma (run sync check off event loop + timeout)
    try:
        chroma_health = await asyncio.wait_for(
            asyncio.to_thread(rag_service.health_check),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        chroma_health = {"status": "unhealthy", "reason": "timeout"}

    health_status["components"]["chroma"] = chroma_health
    if chroma_health.get("status") != "healthy":
        health_status["status"] = "degraded"

    # Check chat memory
    health_status["components"]["chat_memory"] = chat_memory_service.health_check()

    return health_status


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "QuantumCodeHub API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }
