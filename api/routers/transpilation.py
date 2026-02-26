"""
F2: Code transpilation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import ClientContext, RuntimePreferences
from services.transpiler_service import transpiler_service
from services.cache_service import cache_service
from services.validator_service import validator_service
from services.modernization_service import modernization_service
from services.runtime_compatibility import build_runtime_bundle_with_rag
from services.rag_service import rag_service
from core.security import get_current_active_user
from core.config import settings
import time
import hashlib
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/transpile", tags=["Transpilation"])

class TranspilationRequest(BaseModel):
    source_code: str = Field(..., description="Source code to transpile")
    source_framework: str = Field(..., description="Source framework name")
    target_framework: str = Field(..., description="Target framework name")
    preserve_comments: bool = Field(True, description="Preserve code comments")
    optimize: bool = Field(False, description="Apply circuit optimization")
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware transpilation.",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target (legacy/modern/version-specific).",
    )

class TranspilationResponse(BaseModel):
    transpiled_code: str
    source_framework: str
    target_framework: str
    success: bool
    validation_passed: bool
    differences: List[str] = []
    warnings: List[str] = []
    metadata: dict


def _normalize_framework_name(name: str) -> str:
    return (name or "").strip().lower()


def _clean_source_code(code: str) -> str:
    """
    Accept raw python or markdown-fenced code and return python source only.
    """
    text = (code or "").strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return text

@router.post("/convert", response_model=TranspilationResponse)
async def transpile_code(
    request: TranspilationRequest,
    _current_user: dict = Depends(get_current_active_user)
):
    """
    Convert quantum code from one framework to another.
    """
    start_time = time.time()

    try:
        source_framework = _normalize_framework_name(request.source_framework)
        target_framework = _normalize_framework_name(request.target_framework)
        source_code = _clean_source_code(request.source_code)
        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=target_framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/transpile/convert",
        )

        # Only required-field checks (no pre-syntax validation)
        if not source_code:
            raise HTTPException(status_code=400, detail="source_code is required")
        if not source_framework:
            raise HTTPException(status_code=400, detail="source_framework is required")
        if not target_framework:
            raise HTTPException(status_code=400, detail="target_framework is required")
        if source_framework == target_framework:
            raise HTTPException(400, "source_framework and target_framework must be different")

        # Check cache
        cache_key = hashlib.md5(
            (
                f"{source_code}:{source_framework}:{target_framework}:"
                f"{runtime_bundle['cache_fingerprint']}:"
                f"{rag_service.get_framework_version_marker(target_framework)}"
            ).encode()
        ).hexdigest()

        cached = await cache_service.get(f"transpile:{cache_key}")
        if cached:
            logger.info(f"Cache hit for transpilation: {cache_key}")
            return cached

        # Perform transpilation
        transpilation_result = await transpiler_service.transpile(
            source_code=source_code,
            source_framework=source_framework,
            target_framework=target_framework,
            preserve_comments=request.preserve_comments,
            optimize=request.optimize,
            compatibility_context=runtime_bundle["compatibility_context"],
            rag_query_suffix=runtime_bundle["rag_query_suffix"],
            runtime_preferences=runtime_bundle.get("requested_runtime", {}),
        )

        if not transpilation_result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "invalid_input_or_unsupported_conversion",
                    "warnings": transpilation_result.get("warnings", []),
                },
            )

        validation_result = await validator_service.validate(
            code=transpilation_result["code"],
            framework=target_framework,
            user_query=(
                f"Transpile {source_framework} to {target_framework} while preserving behavior."
            ),
            compatibility_context=runtime_bundle["compatibility_context"],
            runtime_preferences=runtime_bundle.get("requested_runtime", {}),
        )

        modernization_result = {
            "attempted": False,
            "applied": False,
            "reason": "disabled",
            "before_deprecation_count": 0,
            "after_deprecation_count": 0,
            "llm_provider": None,
            "llm_model": None,
            "tokens_used": 0,
        }
        if settings.MODERNIZATION_APPLY_ON_TRANSPILE:
            modernization_result = await modernization_service.maybe_modernize(
                framework=target_framework,
                code=transpilation_result["code"],
                validation_result=validation_result,
                user_query=(
                    f"Transpile {source_framework} to {target_framework} while preserving behavior."
                ),
                compatibility_context=runtime_bundle["compatibility_context"],
                runtime_preferences=runtime_bundle.get("requested_runtime"),
            )
            if modernization_result.get("applied"):
                transpilation_result["code"] = modernization_result["code"]
                validation_result = modernization_result["validation_result"]

        response = {
            "transpiled_code": transpilation_result["code"],
            "source_framework": source_framework,
            "target_framework": target_framework,
            "success": transpilation_result["success"],
            "validation_passed": bool(validation_result.get("passed", False)),
            "differences": transpilation_result.get("differences", []),
            "warnings": transpilation_result.get("warnings", []) + validation_result.get("warnings", []),
            "metadata": {
                "latency_ms": int((time.time() - start_time) * 1000),
                "method": transpilation_result.get("method", "llm"),
                "tokens_used": transpilation_result.get("tokens_used", 0),
                "llm_provider": transpilation_result.get("llm_provider"),
                "llm_model": transpilation_result.get("llm_model"),
                "llm_attempt": transpilation_result.get("llm_attempt"),
                "llm_fallback_used": transpilation_result.get("llm_fallback_used", False),
                "cached": False,
                "validation_errors": validation_result.get("errors", []),
                "validation_evaluation": validation_result.get("evaluation", {}),
                "modernization_attempted": modernization_result.get("attempted", False),
                "modernization_applied": modernization_result.get("applied", False),
                "modernization_reason": modernization_result.get("reason"),
                "modernization_before_deprecations": modernization_result.get("before_deprecation_count", 0),
                "modernization_after_deprecations": modernization_result.get("after_deprecation_count", 0),
                "modernization_llm_provider": modernization_result.get("llm_provider"),
                "modernization_llm_model": modernization_result.get("llm_model"),
                "modernization_tokens_used": modernization_result.get("tokens_used", 0),
                "client_type": runtime_bundle["client_type"],
                "client_context": runtime_bundle["client_context"],
                "requested_runtime": runtime_bundle.get("requested_runtime", {}),
                "runtime_requirements": runtime_bundle.get("effective_runtime_target", {}),
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
                "runtime_validation": runtime_bundle["runtime_validation"],
            }
        }

        # Cache successful transpilations
        if response["success"]:
            await cache_service.set(f"transpile:{cache_key}", response, ttl=7200)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Transpilation error: {e}")
        raise HTTPException(500, f"Transpilation failed: {str(e)}")

@router.get("/supported-conversions")
async def get_supported_conversions():
    """Get list of all supported framework conversions"""
    return {
        "conversions": [
            {"from": "qiskit", "to": "pennylane", "status": "stable"},
            {"from": "qiskit", "to": "cirq", "status": "stable"},
            {"from": "pennylane", "to": "qiskit", "status": "stable"},
            {"from": "pennylane", "to": "cirq", "status": "beta"},
            {"from": "cirq", "to": "qiskit", "status": "stable"},
            {"from": "cirq", "to": "pennylane", "status": "beta"},
            {"from": "qiskit", "to": "torchquantum", "status": "beta"},
            {"from": "pennylane", "to": "torchquantum", "status": "beta"},
            {"from": "cirq", "to": "torchquantum", "status": "beta"},
        ]
    }
