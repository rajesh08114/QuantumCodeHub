"""
F1: Code generation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List

from schemas.common import ClientContext, RuntimePreferences
from services.runtime_compatibility import build_runtime_bundle_with_rag
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.validator_service import validator_service
from services.adaptive_routing_service import adaptive_routing_service
from services.modernization_service import modernization_service
from services.cache_service import cache_service
from core.security import get_current_active_user
from core.config import settings
from ml.prompts import CodeGenerationPrompts, ErrorFixingPrompts
import time
import hashlib
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/code", tags=["Code Generation"])

# Request/Response models
class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of quantum circuit")
    framework: str = Field("qiskit", description="Target framework: qiskit, pennylane, cirq, torchquantum")
    num_qubits: Optional[int] = Field(None, description="Optional: specify number of qubits")
    include_explanation: bool = Field(False, description="Include step-by-step explanation")
    include_visualization: bool = Field(False, description="Include circuit diagram")
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware generation (website/vscode_extension).",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target (version-aware legacy/modern generation).",
    )

class CodeGenerationResponse(BaseModel):
    code: str
    framework: str
    explanation: Optional[str] = None
    visualization: Optional[str] = None
    confidence_score: float
    validation_passed: bool
    validation_errors: List[str] = []
    metadata: dict

@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    **F1: Multi-Framework Code Generation**
    
    Generate quantum circuit code from natural language description.
    
    **Supported Frameworks:**
    - qiskit (IBM)
    - pennylane (Xanadu)
    - cirq (Google)
    - torchquantum (MIT)
    
    **Example Request:**
    ```json
    {
      "prompt": "Create a 3-qubit GHZ state",
      "framework": "qiskit",
      "include_explanation": true
    }
    ```
    """
    start_time = time.time()
    user_id = current_user["user_id"]
    framework = (request.framework or "qiskit").strip().lower()
    
    try:
        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/code/generate",
        )

        # Validate framework
        valid_frameworks = ["qiskit", "pennylane", "cirq", "torchquantum"]
        if framework not in valid_frameworks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid framework. Must be one of: {valid_frameworks}"
            )
        
        # Check cache
        cache_key = hashlib.md5(
            (
                f"{request.prompt}:{framework}:{request.num_qubits}:"
                f"{runtime_bundle['cache_fingerprint']}"
            ).encode()
        ).hexdigest()
        
        cached_result = await cache_service.get(f"code:gen:{cache_key}")
        if cached_result:
            logger.info(f"Cache hit for code generation: {cache_key}")
            return cached_result
        
        # Step 1: RAG - Retrieve relevant documentation
        rag_query = f"{request.prompt}\n\n{runtime_bundle['rag_query_suffix']}"
        rag_results = await rag_service.retrieve_context(
            query=rag_query,
            framework=framework,
            top_k=5,
            request_source="/api/code/generate",
        )
        
        # Step 2: Build prompt
        system_message = CodeGenerationPrompts.get_system_message(framework)
        user_prompt = CodeGenerationPrompts.build_generation_prompt(
            user_query=request.prompt,
            framework=framework,
            rag_context=rag_results["context"],
            num_qubits=request.num_qubits,
            include_explanation=request.include_explanation,
            compatibility_context=runtime_bundle["compatibility_context"],
        )

        generation_max_tokens = _resolve_generation_max_tokens(
            include_explanation=request.include_explanation
        )
        preferred_chain = adaptive_routing_service.get_preferred_chain(
            framework=framework,
            default_chain=llm_service.provider_chain,
        )
        
        # Step 3: Generate code with LLM
        llm_response = await llm_service.generate_code(
            prompt=user_prompt,
            system_message=system_message,
            max_tokens=generation_max_tokens,
            temperature=0.12,
            preferred_chain=preferred_chain,
        )
        logger.info(
            "Code generation LLM response provider=%s model=%s tokens=%s fallback_used=%s",
            llm_response.get("provider"),
            llm_response.get("model"),
            llm_response.get("tokens_used"),
            llm_response.get("fallback_used", False),
        )
        
        generated_text = llm_response["generated_text"]
        
        # Step 4: Parse response
        code = extract_code_from_response(generated_text)
        explanation = extract_explanation(generated_text) if request.include_explanation else None
        
        # Step 5: Validate code
        validation_result = await validator_service.validate(
            code=code,
            framework=framework,
            user_query=request.prompt,
            rag_context=rag_results.get("context", ""),
            compatibility_context=runtime_bundle["compatibility_context"],
        )

        repair_result = None
        initial_validation_result = dict(validation_result)
        repair_applied = False
        if _should_attempt_auto_repair(validation_result):
            repair_result = await _attempt_auto_repair(
                framework=framework,
                code=code,
                user_query=request.prompt,
                validation_errors=validation_result.get("errors", []),
                rag_context=rag_results.get("context", ""),
                compatibility_context=runtime_bundle["compatibility_context"],
                runtime_preferences=runtime_bundle.get("requested_runtime"),
                preferred_chain=preferred_chain,
            )
            if repair_result and _is_repair_better(
                before=validation_result,
                after=repair_result["validation_result"],
            ):
                repair_applied = True
                code = repair_result["code"]
                validation_result = repair_result["validation_result"]
                generated_text = repair_result["generated_text"]
                llm_response = repair_result["llm_response"]
                explanation = extract_explanation(generated_text) if request.include_explanation else None

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
        if settings.MODERNIZATION_APPLY_ON_GENERATE:
            modernization_result = await modernization_service.maybe_modernize(
                framework=framework,
                code=code,
                validation_result=validation_result,
                user_query=request.prompt,
                rag_context=rag_results.get("context", ""),
                compatibility_context=runtime_bundle["compatibility_context"],
                preferred_chain=preferred_chain,
            )
            if modernization_result.get("applied"):
                code = modernization_result["code"]
                validation_result = modernization_result["validation_result"]
                if request.include_explanation:
                    explanation = (
                        "Code was modernized to stable non-deprecated APIs after validation."
                    )
        
        # Step 6: Calculate confidence score
        confidence_score = calculate_confidence_score(
            validation_result=validation_result,
            rag_score=rag_results["average_score"],
            llm_tokens=llm_response["tokens_used"],
            auto_repair_used=repair_applied,
            llm_eval_score=_extract_llm_eval_score(validation_result),
        )
        adaptive_routing_service.record_outcome(
            framework=framework,
            provider=llm_response.get("provider", ""),
            validation_passed=bool(validation_result.get("passed")),
            confidence_score=confidence_score,
            latency_ms=int((time.time() - start_time) * 1000),
        )
        
        # Build response
        response = {
            "code": code,
            "framework": framework,
            "explanation": explanation,
            "visualization": None,  # TODO: Implement circuit visualization
            "confidence_score": confidence_score,
            "validation_passed": validation_result["passed"],
            "validation_errors": validation_result["errors"],
            "metadata": {
                "tokens_used": llm_response["tokens_used"],
                "llm_provider": llm_response.get("provider"),
                "llm_model": llm_response.get("model"),
                "llm_attempt": llm_response.get("attempt"),
                "llm_fallback_used": llm_response.get("fallback_used", False),
                "rag_documents": len(rag_results["documents"]),
                "latency_ms": int((time.time() - start_time) * 1000),
                "cached": False,
                "auto_repair_used": repair_applied,
                "auto_repair_attempted": bool(repair_result),
                "initial_validation_errors": initial_validation_result.get("errors", []),
                "validation_warnings": validation_result.get("warnings", []),
                "validation_evaluation": validation_result.get("evaluation", {}),
                "modernization_attempted": modernization_result.get("attempted", False),
                "modernization_applied": modernization_result.get("applied", False),
                "modernization_reason": modernization_result.get("reason"),
                "modernization_before_deprecations": modernization_result.get("before_deprecation_count", 0),
                "modernization_after_deprecations": modernization_result.get("after_deprecation_count", 0),
                "modernization_llm_provider": modernization_result.get("llm_provider"),
                "modernization_llm_model": modernization_result.get("llm_model"),
                "modernization_tokens_used": modernization_result.get("tokens_used", 0),
                "generation_max_tokens": generation_max_tokens,
                "adaptive_routing_enabled": settings.ENABLE_ADAPTIVE_ROUTING,
                "adaptive_preferred_chain": preferred_chain,
                "client_type": runtime_bundle["client_type"],
                "client_context": runtime_bundle["client_context"],
                "requested_runtime": runtime_bundle.get("requested_runtime", {}),
                "runtime_requirements": runtime_bundle.get("effective_runtime_target", {}),
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
                "runtime_validation": runtime_bundle["runtime_validation"],
            }
        }
        
        # Cache successful results
        if validation_result["passed"]:
            await cache_service.set(
                key=f"code:gen:{cache_key}",
                value=response,
                ttl=3600  # 1 hour
            )
        
        # Log request only when the request has a real authenticated user.
        if user_id:
            await log_api_request(
                user_id=user_id,
                endpoint="/api/code/generate",
                feature_type="code_generation",
                request_data=request.model_dump(),
                response_data=response,
                latency_ms=response["metadata"]["latency_ms"]
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Code generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}"
        )

def extract_code_from_response(text: str) -> str:
    """Extract code block from LLM response"""
    # Look for code between fenced code blocks.
    import re
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: return entire text if no code block found
    return text.strip()

def extract_explanation(text: str) -> str:
    """Extract explanation text from LLM response"""
    # Extract text before or after code block
    import re
    pattern = r"```(?:python)?\s*.*?```"
    explanation = re.sub(pattern, "", text, flags=re.DOTALL)
    return explanation.strip()

def calculate_confidence_score(
    validation_result: dict,
    rag_score: float,
    llm_tokens: int,
    auto_repair_used: bool = False,
    llm_eval_score: Optional[float] = None,
) -> float:
    """
    Calculate confidence score for generated code
    
    Factors:
    - Validation success (70%)
    - RAG relevance (20%)
    - Token efficiency (10%)
    """
    passed = bool(validation_result.get("passed"))
    error_count = len(validation_result.get("errors", []))
    validation_score = 1.0 if passed else max(0.1, 0.5 - (min(error_count, 6) * 0.08))
    rag_relevance = max(0.0, min(float(rag_score or 0.0), 1.0))

    if 120 <= llm_tokens <= 650:
        token_score = 1.0
    elif llm_tokens <= 900:
        token_score = 0.85
    else:
        token_score = 0.7

    confidence = (
        validation_score * 0.7 +
        rag_relevance * 0.2 +
        token_score * 0.1
    )

    if auto_repair_used and passed:
        confidence += 0.05
    if isinstance(llm_eval_score, (int, float)):
        confidence = (confidence * 0.8) + (max(0.0, min(float(llm_eval_score), 1.0)) * 0.2)

    return round(min(confidence, 0.99), 2)


def _extract_llm_eval_score(validation_result: dict) -> Optional[float]:
    evaluation = validation_result.get("evaluation", {})
    llm_eval = evaluation.get("llm", {}) if isinstance(evaluation, dict) else {}
    score = llm_eval.get("score") if isinstance(llm_eval, dict) else None
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _resolve_generation_max_tokens(include_explanation: bool) -> int:
    base = int(settings.GENERATION_MAX_TOKENS or 1500)
    if include_explanation:
        return max(700, min(base, 1500))
    return max(450, min(base, 900))


def _should_attempt_auto_repair(validation_result: dict) -> bool:
    if validation_result.get("passed"):
        return False
    return bool(validation_result.get("errors"))


def _is_repair_better(before: dict, after: dict) -> bool:
    before_errors = len(before.get("errors", []))
    after_errors = len(after.get("errors", []))
    if after.get("passed") and not before.get("passed"):
        return True
    return after_errors < before_errors


async def _attempt_auto_repair(
    framework: str,
    code: str,
    user_query: str,
    validation_errors: List[str],
    rag_context: str,
    compatibility_context: str,
    preferred_chain: Optional[List[str]] = None,
) -> Optional[dict]:
    error_message = "\n".join(validation_errors[:8])
    repair_prompt = ErrorFixingPrompts.build_error_fixing_prompt(
        code=code,
        framework=framework,
        error_message=error_message,
        rag_context=rag_context,
        compatibility_context=compatibility_context,
    )
    repair_llm_response = await llm_service.generate_code(
        prompt=repair_prompt,
        system_message=CodeGenerationPrompts.get_system_message(framework),
        max_tokens=max(500, min(int(settings.ERROR_FIXING_MAX_TOKENS or 1800), 900)),
        temperature=0.05,
        preferred_chain=preferred_chain,
    )
    repaired_text = repair_llm_response.get("generated_text", "")
    repaired_code = extract_code_from_response(repaired_text)
    if not repaired_code:
        return None

    repaired_validation = await validator_service.validate(
        code=repaired_code,
        framework=framework,
        user_query=user_query,
        rag_context=rag_context,
        compatibility_context=compatibility_context,
    )
    return {
        "code": repaired_code,
        "validation_result": repaired_validation,
        "generated_text": repaired_text,
        "llm_response": repair_llm_response,
    }

async def log_api_request(
    user_id: str,
    endpoint: str,
    feature_type: str,
    request_data: dict,
    response_data: dict,
    latency_ms: int
):
    """Log API request to database for analytics"""
    from core.database import get_db_connection, release_db_connection
    import uuid
    
    conn = await get_db_connection()
    try:
        await conn.execute(
            """
            INSERT INTO api_requests (
                user_id, endpoint, feature_type, request_data, response_data,
                status_code, latency_ms, validation_passed, confidence_score,
                llm_tokens_used, rag_documents_retrieved
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            uuid.UUID(user_id),
            endpoint,
            feature_type,
            request_data,
            response_data,
            200,
            latency_ms,
            response_data.get("validation_passed"),
            response_data.get("confidence_score"),
            response_data["metadata"].get("tokens_used"),
            response_data["metadata"].get("rag_documents")
        )
    finally:
        await release_db_connection(conn)
