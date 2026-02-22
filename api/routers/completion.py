"""
F3: Auto-completion endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.common import ClientContext, RuntimePreferences
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.runtime_compatibility import build_runtime_bundle_with_rag
from ml.prompts import CompletionPrompts
from core.config import settings
from core.security import get_current_active_user
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/complete", tags=["Auto-Completion"])


class CompletionRequest(BaseModel):
    code_prefix: str = Field(..., description="Code before cursor")
    framework: str = Field("qiskit", description="Framework being used")
    cursor_line: int = Field(..., description="Line number of cursor")
    cursor_column: int = Field(..., description="Column number of cursor")
    max_suggestions: int = Field(5, description="Maximum number of suggestions")
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware completion suggestions.",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target for version-aware completion.",
    )


class CompletionSuggestion(BaseModel):
    code: str
    description: str
    priority: int
    confidence: float


class CompletionResponse(BaseModel):
    suggestions: List[CompletionSuggestion]
    context_detected: dict
    latency_ms: int
    metadata: dict = Field(default_factory=dict)


def _normalize_framework_name(name: str) -> str:
    return (name or "qiskit").strip().lower()


@router.post("/suggest", response_model=CompletionResponse)
async def get_completions(
    request: CompletionRequest,
    _current_user: dict = Depends(get_current_active_user)
):
    """
    Generate code completion suggestions based on current context.
    """
    start_time = time.time()

    try:
        framework = _normalize_framework_name(request.framework)
        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/complete/suggest",
        )
        code_prefix = (request.code_prefix or "").rstrip()
        max_suggestions = max(1, min(request.max_suggestions, 10))

        if not code_prefix:
            raise HTTPException(status_code=400, detail="code_prefix is required")

        # Analyze code context
        context = analyze_code_context(
            code=code_prefix,
            line=request.cursor_line,
            column=request.cursor_column,
            framework=framework
        )

        # Get relevant documentation
        rag_query = (
            f"{context.get('last_statement', '')} {context.get('scope', 'global')}".strip()
            + f"\n\n{runtime_bundle['rag_query_suffix']}"
        )
        rag_results = await rag_service.retrieve_context(
            query=rag_query,
            framework=framework,
            top_k=3,
            request_source="/api/complete/suggest",
        )

        # Generate completion suggestions
        prompt = build_completion_prompt(
            code_prefix=code_prefix,
            context=context,
            rag_context=rag_results.get("context", ""),
            framework=framework,
            max_suggestions=max_suggestions,
            compatibility_context=runtime_bundle["compatibility_context"],
        )

        llm_response = await llm_service.generate_code(
            prompt=prompt,
            max_tokens=settings.COMPLETION_MAX_TOKENS,
            temperature=0.5
        )
        logger.info(
            "Completion LLM response provider=%s model=%s tokens=%s fallback_used=%s",
            llm_response.get("provider"),
            llm_response.get("model"),
            llm_response.get("tokens_used"),
            llm_response.get("fallback_used", False),
        )

        # Parse suggestions
        suggestions = parse_completion_suggestions(
            llm_response.get("generated_text", ""),
            max_suggestions=max_suggestions
        )

        if not suggestions:
            suggestions = [
                CompletionSuggestion(
                    code="",
                    description="No structured suggestions parsed from model output.",
                    priority=1,
                    confidence=0.5,
                )
            ]

        return {
            "suggestions": suggestions,
            "context_detected": context,
            "latency_ms": int((time.time() - start_time) * 1000),
            "metadata": {
                "tokens_used": llm_response.get("tokens_used", 0),
                "llm_provider": llm_response.get("provider"),
                "llm_model": llm_response.get("model"),
                "llm_attempt": llm_response.get("attempt"),
                "llm_fallback_used": llm_response.get("fallback_used", False),
                "client_type": runtime_bundle["client_type"],
                "client_context": runtime_bundle["client_context"],
                "requested_runtime": runtime_bundle.get("requested_runtime", {}),
                "runtime_requirements": runtime_bundle.get("effective_runtime_target", {}),
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
                "runtime_validation": runtime_bundle["runtime_validation"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Completion error: {e}")
        raise HTTPException(500, f"Completion generation failed: {str(e)}")


def analyze_code_context(code: str, line: int, column: int, framework: str) -> dict:
    """Analyze code to understand completion context"""
    import ast

    lines = code.split("\n")
    safe_line = max(line, 1)
    current_line = lines[safe_line - 1] if safe_line <= len(lines) else ""
    safe_col = max(column, 0)

    context = {
        "current_line": current_line,
        "last_statement": current_line[:safe_col].strip(),
        "scope": "global",
        "variables": [],
        "imports": [],
        "framework": framework,
    }

    try:
        # Parse AST to extract variables and imports
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                context["imports"].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    context["imports"].append(node.module)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        context["variables"].append(target.id)
    except Exception:
        # If parsing fails, use basic context
        pass

    return context


def build_completion_prompt(
    code_prefix: str,
    context: dict,
    rag_context: str,
    framework: str,
    max_suggestions: int,
    compatibility_context: str = "",
) -> str:
    """Build prompt for completion suggestions"""
    return CompletionPrompts.build_completion_prompt(
        code_prefix=code_prefix,
        framework=framework,
        cursor_context=context,
        rag_context=rag_context,
        max_suggestions=max_suggestions,
        compatibility_context=compatibility_context,
    )


def parse_completion_suggestions(text: str, max_suggestions: int) -> List[CompletionSuggestion]:
    """Parse LLM response into structured suggestions"""
    suggestions: List[CompletionSuggestion] = []
    lines = text.strip().split("\n")

    priority = 1
    for line in lines:
        if len(suggestions) >= max_suggestions:
            break

        raw = line.strip()
        if not raw:
            continue

        try:
            if "." in raw and raw[0].isdigit():
                raw = raw.split(".", 1)[1].strip()

            parts = raw.split("-", 1)
            if len(parts) != 2:
                continue

            code = parts[0].strip()
            description = parts[1].strip()
            if not code and not description:
                continue

            suggestions.append(
                CompletionSuggestion(
                    code=code,
                    description=description,
                    priority=priority,
                    confidence=max(0.9 - (priority * 0.1), 0.5),
                )
            )
            priority += 1
        except Exception:
            continue

    return suggestions
