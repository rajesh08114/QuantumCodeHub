"""
F4: Code explanation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from schemas.common import ClientContext, RuntimePreferences
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.rag_guardrails import ensure_rag_consistency, build_version_enforcement_context
from services.runtime_compatibility import build_runtime_bundle_with_rag
from ml.prompts import ExplanationPrompts
from core.config import settings
from core.security import get_current_active_user
from utils.domain_classifier import is_quantum_domain_text
from utils.explanation_parser import parse_explanation, extract_mathematics
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/explain", tags=["Code Explanation"])
VALID_FRAMEWORKS = {"qiskit", "pennylane", "cirq", "torchquantum"}

class ExplanationRequest(BaseModel):
    code: str = Field(..., description="Code to explain")
    framework: str = Field(..., description="Framework used")
    detail_level: str = Field("intermediate", description="beginner, intermediate, or advanced")
    include_math: bool = Field(False, description="Include mathematical formulations")
    include_visualization: bool = Field(False, description="Include circuit diagrams")
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware explanations.",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target for explanation context.",
    )

class ExplanationResponse(BaseModel):
    overview: str
    gate_breakdown: str
    quantum_concepts: str
    mathematics: Optional[str] = None
    applications: str
    visualization: Optional[str] = None
    requested_runtime: Optional[dict] = None
    runtime_requirements: Optional[dict] = None
    runtime_recommendations: Optional[dict] = None
    runtime_validation: Optional[dict] = None

@router.post("/code", response_model=ExplanationResponse)
async def explain_code(
    request: ExplanationRequest,
    _current_user: dict = Depends(get_current_active_user)
):
    """
    **F4: Multi-Level Code Explanation**
    
    Explain quantum code with gate-level, conceptual, and mathematical details.
    
    **Detail Levels:**
    - beginner: Simple explanations for newcomers
    - intermediate: Detailed with quantum concepts
    - advanced: Includes mathematical formulations
    
    **Example:**
    ```json
    {
      "code": "qc = QuantumCircuit(2)\\nqc.h(0)\\nqc.cx(0,1)",
      "framework": "qiskit",
      "detail_level": "intermediate",
      "include_math": true
    }
    ```
    """
    try:
        framework = (request.framework or "").strip().lower()
        if framework not in VALID_FRAMEWORKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid framework. Must be one of: {sorted(VALID_FRAMEWORKS)}",
            )
        if not is_quantum_domain_text(request.code):
            logger.warning(
                "Non-quantum domain request blocked endpoint=/api/explain/code field=code preview=%s",
                " ".join((request.code or "").split())[:220],
            )
            raise HTTPException(status_code=400, detail="not quantum domain")

        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/explain/code",
        )

        # Get relevant documentation for context
        rag_results = await rag_service.retrieve_context(
            query=(
                f"Explain this {request.framework} code:\n{request.code}\n\n"
                f"{runtime_bundle['rag_query_suffix']}"
            ),
            framework=framework,
            top_k=3,
            request_source="/api/explain/code",
            runtime_preferences=runtime_bundle.get("requested_runtime", {}),
            prefer_latest_version=True,
        )
        if rag_results.get("error"):
            raise HTTPException(
                status_code=400,
                detail=f"RAG retrieval failed for framework '{framework}': {rag_results.get('error')}",
            )
        try:
            ensure_rag_consistency(
                rag_results=rag_results,
                framework=framework,
                runtime_preferences=runtime_bundle.get("requested_runtime", {}),
                prefer_latest_version=True,
                require_documents=1,
                allow_unfiltered_fallback=False,
            )
        except Exception as rag_consistency_error:
            raise HTTPException(
                status_code=400,
                detail=f"RAG consistency validation failed for framework '{framework}': {rag_consistency_error}",
            )
        strict_version_context = build_version_enforcement_context(
            framework=framework,
            rag_results=rag_results,
        )
        compatibility_context = runtime_bundle["compatibility_context"]
        if strict_version_context:
            compatibility_context = f"{compatibility_context}\n{strict_version_context}".strip()
        
        # Build explanation prompt
        prompt = ExplanationPrompts.build_explanation_prompt(
            code=request.code,
            framework=framework,
            detail_level=request.detail_level,
            rag_context=rag_results.get("context", ""),
            compatibility_context=compatibility_context,
        )
        
        # Generate explanation
        llm_response = await llm_service.generate_code(
            prompt=prompt,
            max_tokens=settings.EXPLANATION_MAX_TOKENS,
            temperature=0.4
        )
        logger.info(
            "Explanation LLM response provider=%s model=%s tokens=%s fallback_used=%s",
            llm_response.get("provider"),
            llm_response.get("model"),
            llm_response.get("tokens_used"),
            llm_response.get("fallback_used", False),
        )
        
        # Parse structured explanation
        explanation = parse_explanation(llm_response["generated_text"])
        
        # Add mathematics if requested
        if request.include_math:
            explanation["mathematics"] = extract_mathematics(llm_response["generated_text"])

        explanation["requested_runtime"] = runtime_bundle.get("requested_runtime", {})
        explanation["runtime_requirements"] = runtime_bundle.get("effective_runtime_target", {})
        explanation["runtime_recommendations"] = runtime_bundle["runtime_recommendations"]
        explanation["runtime_validation"] = runtime_bundle["runtime_validation"]
        
        return explanation
        
    except Exception as e:
        logger.exception(f"Explanation error: {e}")
        raise HTTPException(500, f"Explanation generation failed: {str(e)}")

