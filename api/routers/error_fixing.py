"""
F5: Error detection and fixing endpoints.
"""
import logging
import re
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.config import settings
from core.security import get_current_active_user
from ml.prompts import CodeGenerationPrompts, ErrorFixingPrompts
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.runtime_compatibility import build_runtime_bundle_with_rag
from services.validator_service import validator_service
from services.modernization_service import modernization_service
from schemas.common import ClientContext, RuntimePreferences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/fix", tags=["Error Fixing"])

VALID_FRAMEWORKS = {"qiskit", "pennylane", "cirq", "torchquantum"}


class ErrorFixRequest(BaseModel):
    code: str = Field(..., description="Buggy code to fix")
    framework: str = Field(..., description="Framework used")
    error_message: Optional[str] = Field(
        None,
        description="Optional runtime/validation error text or traceback",
    )
    include_explanation: bool = Field(
        True,
        description="Include issue list and explanation of fixes",
    )
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware error fixing.",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target (legacy/modern/version-specific).",
    )


class ErrorFixResponse(BaseModel):
    fixed_code: str
    issues_identified: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    metadata: dict


def _extract_code_response(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text or "", re.DOTALL)
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _remove_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text or "", flags=re.DOTALL).strip()


def _extract_markdown_section(text: str, aliases: List[str]) -> Optional[str]:
    if not text:
        return None

    escaped_aliases = "|".join(re.escape(alias) for alias in aliases)
    pattern = (
        rf"(?ims)(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?(?:{escaped_aliases})(?:\*\*)?\s*:?\s*\n"
        r"(.*?)(?=\n\s*(?:#+\s*|\*\*[A-Za-z][^\n]{0,80}\*\*\s*$|[A-Za-z][A-Za-z0-9 _/\-]{2,80}:\s*$)|\Z)"
    )
    match = re.search(pattern, text)
    if not match:
        return None
    section = (match.group(1) or "").strip()
    return section or None


def _parse_issue_lines(text: Optional[str], max_items: int = 12) -> List[str]:
    if not text:
        return []

    issues: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line or "").strip()
        if not cleaned:
            continue
        issues.append(cleaned)
        if len(issues) >= max_items:
            break
    return issues


def _build_issue_summary(raw_text: str) -> List[str]:
    issue_section = _extract_markdown_section(
        raw_text,
        aliases=[
            "identified issues",
            "issues",
            "errors found",
            "problems found",
        ],
    )
    issues = _parse_issue_lines(issue_section)
    if issues:
        return issues

    # Fallback: collect bullets from non-code text.
    non_code = _remove_code_blocks(raw_text)
    return _parse_issue_lines(non_code, max_items=8)


def _build_fix_explanation(raw_text: str) -> Optional[str]:
    explanation = _extract_markdown_section(
        raw_text,
        aliases=[
            "explanation of fixes",
            "fix explanation",
            "explanation",
            "summary of fixes",
        ],
    )
    if explanation:
        return explanation

    non_code = _remove_code_blocks(raw_text)
    return non_code or None


@router.post("/code", response_model=ErrorFixResponse)
async def fix_code(
    request: ErrorFixRequest,
    _current_user: dict = Depends(get_current_active_user),
):
    """
    Fix quantum code errors and return corrected runnable code.
    """
    start = time.time()

    try:
        framework = (request.framework or "").strip().lower()
        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/fix/code",
        )
        source_code = (request.code or "").strip()
        error_message = (request.error_message or "").strip() or None

        if not source_code:
            raise HTTPException(status_code=400, detail="code is required")
        if framework not in VALID_FRAMEWORKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid framework. Must be one of: {sorted(VALID_FRAMEWORKS)}",
            )

        logger.info(
            "Fix request received framework=%s error_message_present=%s code_preview=%s",
            framework,
            bool(error_message),
            " ".join(source_code.split())[:200],
        )

        rag_results = await rag_service.retrieve_context(
            query=(
                f"Fix this {framework} code:\n{source_code}\nError: {error_message or ''}\n\n"
                f"{runtime_bundle['rag_query_suffix']}"
            ),
            framework=framework,
            top_k=5,
            request_source="/api/fix/code",
        )

        prompt = ErrorFixingPrompts.build_error_fixing_prompt(
            code=source_code,
            framework=framework,
            error_message=error_message,
            rag_context=rag_results.get("context", ""),
            compatibility_context=runtime_bundle["compatibility_context"],
        )

        llm_response = await llm_service.generate_code(
            prompt=prompt,
            system_message=CodeGenerationPrompts.get_system_message(framework),
            max_tokens=settings.ERROR_FIXING_MAX_TOKENS,
            temperature=0.2,
        )
        logger.info(
            "Fix LLM response provider=%s model=%s tokens=%s fallback_used=%s",
            llm_response.get("provider"),
            llm_response.get("model"),
            llm_response.get("tokens_used"),
            llm_response.get("fallback_used", False),
        )

        raw_text = llm_response.get("generated_text", "")
        fixed_code = _extract_code_response(raw_text)
        if not fixed_code:
            fixed_code = source_code

        validation_result = await validator_service.validate(
            code=fixed_code,
            framework=framework,
            user_query=(error_message or f"Fix {framework} code errors and return runnable code."),
            rag_context=rag_results.get("context", ""),
            compatibility_context=runtime_bundle["compatibility_context"],
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
        if settings.MODERNIZATION_APPLY_ON_FIX:
            modernization_result = await modernization_service.maybe_modernize(
                framework=framework,
                code=fixed_code,
                validation_result=validation_result,
                user_query=(error_message or "Modernize fixed code to stable APIs."),
                rag_context=rag_results.get("context", ""),
                compatibility_context=runtime_bundle["compatibility_context"],
                runtime_preferences=runtime_bundle.get("requested_runtime"),
            )
            if modernization_result.get("applied"):
                fixed_code = modernization_result["code"]
                validation_result = modernization_result["validation_result"]

        issues_identified = _build_issue_summary(raw_text) if request.include_explanation else []
        explanation = _build_fix_explanation(raw_text) if request.include_explanation else None

        response = {
            "fixed_code": fixed_code,
            "issues_identified": issues_identified,
            "explanation": explanation,
            "metadata": {
                "framework": framework,
                "rag_documents": len(rag_results.get("documents", [])),
                "rag_average_score": rag_results.get("average_score", 0),
                "tokens_used": llm_response.get("tokens_used", 0),
                "llm_provider": llm_response.get("provider"),
                "llm_model": llm_response.get("model"),
                "llm_attempt": llm_response.get("attempt"),
                "llm_fallback_used": llm_response.get("fallback_used", False),
                "validation_passed": bool(validation_result.get("passed", False)),
                "validation_errors": validation_result.get("errors", []),
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
                "latency_ms": int((time.time() - start) * 1000),
                "client_type": runtime_bundle["client_type"],
                "client_context": runtime_bundle["client_context"],
                "requested_runtime": runtime_bundle.get("requested_runtime", {}),
                "runtime_requirements": runtime_bundle.get("effective_runtime_target", {}),
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
                "runtime_validation": runtime_bundle["runtime_validation"],
            },
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fixing failed: %s", e)
        raise HTTPException(500, f"Error fixing failed: {str(e)}")
