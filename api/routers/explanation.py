"""
F4: Code explanation endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from schemas.common import ClientContext
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.runtime_compatibility import build_runtime_bundle
from ml.prompts import ExplanationPrompts
from core.config import settings
from core.security import get_current_active_user
from utils.explanation_parser import parse_explanation, extract_mathematics
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/explain", tags=["Code Explanation"])

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

class ExplanationResponse(BaseModel):
    overview: str
    gate_breakdown: str
    quantum_concepts: str
    mathematics: Optional[str] = None
    applications: str
    visualization: Optional[str] = None
    runtime_recommendations: Optional[dict] = None

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
        runtime_bundle = build_runtime_bundle(
            framework=request.framework,
            client_context=request.client_context,
        )

        # Get relevant documentation for context
        rag_results = await rag_service.retrieve_context(
            query=(
                f"Explain this {request.framework} code:\n{request.code}\n\n"
                f"{runtime_bundle['rag_query_suffix']}"
            ),
            framework=request.framework,
            top_k=3,
            request_source="/api/explain/code",
        )
        
        # Build explanation prompt
        prompt = ExplanationPrompts.build_explanation_prompt(
            code=request.code,
            framework=request.framework,
            detail_level=request.detail_level,
            rag_context=rag_results.get("context", ""),
            compatibility_context=runtime_bundle["compatibility_context"],
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

        explanation["runtime_recommendations"] = runtime_bundle["runtime_recommendations"]
        
        return explanation
        
    except Exception as e:
        logger.exception(f"Explanation error: {e}")
        raise HTTPException(500, f"Explanation generation failed: {str(e)}")

