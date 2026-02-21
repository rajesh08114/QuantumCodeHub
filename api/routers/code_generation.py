"""
F1: Code generation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List

from schemas.common import ClientContext
from services.runtime_compatibility import build_runtime_bundle
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.validator_service import validator_service
from services.cache_service import cache_service
from core.security import get_current_active_user
from core.config import settings
from ml.prompts import CodeGenerationPrompts
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
        runtime_bundle = build_runtime_bundle(
            framework=framework,
            client_context=request.client_context,
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
            compatibility_context=runtime_bundle["compatibility_context"],
        )
        
        # Step 3: Generate code with LLM
        llm_response = await llm_service.generate_code(
            prompt=user_prompt,
            system_message=system_message,
            max_tokens=settings.GENERATION_MAX_TOKENS,
            temperature=0.2
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
            framework=framework
        )
        
        # Step 6: Calculate confidence score
        confidence_score = calculate_confidence_score(
            validation_result=validation_result,
            rag_score=rag_results["average_score"],
            llm_tokens=llm_response["tokens_used"]
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
                "client_type": runtime_bundle["client_type"],
                "client_context": runtime_bundle["client_context"],
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
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
    llm_tokens: int
) -> float:
    """
    Calculate confidence score for generated code
    
    Factors:
    - Validation success (60%)
    - RAG relevance (30%)
    - Code length appropriateness (10%)
    """
    validation_score = 1.0 if validation_result["passed"] else 0.3
    rag_relevance = min(rag_score, 1.0)
    
    # Normalize token count (prefer 100-500 tokens)
    token_score = 1.0 if 100 <= llm_tokens <= 500 else 0.7
    
    confidence = (
        validation_score * 0.6 +
        rag_relevance * 0.3 +
        token_score * 0.1
    )
    
    return round(confidence, 2)

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
