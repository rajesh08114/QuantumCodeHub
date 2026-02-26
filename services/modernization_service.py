"""
Framework-agnostic modernization pass to rewrite deprecated/legacy APIs.
"""
import logging
import re
from typing import Dict, List, Optional

from core.config import settings
from services.llm_service import llm_service
from services.validator_service import validator_service

logger = logging.getLogger(__name__)


def _extract_code_from_text(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _deprecation_warnings(validation_result: Dict) -> List[str]:
    warnings = validation_result.get("warnings", []) if isinstance(validation_result, dict) else []
    output: List[str] = []
    for warning in warnings:
        text = (warning or "").strip()
        lowered = text.lower()
        if "deprecat" in lowered or "legacy" in lowered or "obsolete" in lowered:
            output.append(text)
    return output


def _error_count(validation_result: Dict) -> int:
    if not isinstance(validation_result, dict):
        return 0
    return len(validation_result.get("errors", []) or [])


def _build_modernization_prompt(
    framework: str,
    code: str,
    deprecation_warnings: List[str],
    user_query: str = "",
    rag_context: str = "",
    compatibility_context: str = "",
) -> str:
    warnings_block = "\n".join(f"- {item}" for item in deprecation_warnings) or "- Deprecated API usage detected."
    query_text = (user_query or "").strip() or "Modernize generated quantum code."
    docs = (rag_context or "").strip() or "No additional documentation context provided."
    runtime = (compatibility_context or "").strip() or "No runtime compatibility context provided."

    return f"""Task: Rewrite this {framework} code to modern stable APIs and remove deprecated usage.

User intent:
{query_text}

Deprecated/legacy findings:
{warnings_block}

Runtime compatibility context:
{runtime}

Documentation context:
{docs}

Current code:
```python
{code}
```

Requirements:
1. Preserve algorithm behavior and output semantics.
2. Replace deprecated/legacy APIs with current stable alternatives.
3. Keep code fully runnable with required imports.
4. Do not change framework.
5. Do not introduce placeholders.

Output format (strict):
```python
# modernized {framework} code
```
"""


class ModernizationService:
    """Modernization controller used across generation/transpilation/fix flows."""

    def should_attempt(self, validation_result: Dict, runtime_preferences: Optional[Dict] = None) -> bool:
        if not settings.ENABLE_MODERNIZATION_REPAIR:
            return False
        if not settings.MODERNIZATION_ON_DEPRECATION:
            return False
        prefs = runtime_preferences or {}
        mode = (prefs.get("mode") or "auto").strip().lower()
        allow_deprecated = bool(prefs.get("allow_deprecated_apis", False))
        if mode == "legacy" or allow_deprecated:
            return False
        return bool(_deprecation_warnings(validation_result))

    async def maybe_modernize(
        self,
        *,
        framework: str,
        code: str,
        validation_result: Dict,
        user_query: str = "",
        rag_context: str = "",
        compatibility_context: str = "",
        runtime_preferences: Optional[Dict] = None,
        preferred_chain: Optional[List[str]] = None,
    ) -> Dict:
        prefs = runtime_preferences or {}
        attempted = self.should_attempt(validation_result, runtime_preferences=prefs)
        before_deprecations = _deprecation_warnings(validation_result)
        payload = {
            "attempted": attempted,
            "applied": False,
            "reason": "skipped",
            "before_deprecation_count": len(before_deprecations),
            "after_deprecation_count": len(before_deprecations),
            "before_deprecations": before_deprecations,
            "after_deprecations": before_deprecations,
            "code": code,
            "validation_result": validation_result,
            "llm_provider": None,
            "llm_model": None,
            "tokens_used": 0,
        }
        if not attempted:
            mode = (prefs.get("mode") or "auto").strip().lower()
            if mode == "legacy" or bool(prefs.get("allow_deprecated_apis", False)):
                payload["reason"] = "runtime_preferences_require_legacy_apis"
            else:
                payload["reason"] = "no_deprecation_warning_or_disabled"
            return payload

        try:
            prompt = _build_modernization_prompt(
                framework=framework,
                code=code,
                deprecation_warnings=before_deprecations,
                user_query=user_query,
                rag_context=rag_context,
                compatibility_context=compatibility_context,
            )
            llm = await llm_service.generate_code(
                prompt=prompt,
                max_tokens=max(280, min(int(settings.MODERNIZATION_MAX_TOKENS or 700), 1400)),
                temperature=0.05,
                preferred_chain=preferred_chain,
            )
            rewritten_code = _extract_code_from_text(llm.get("generated_text", ""))
            if not rewritten_code:
                payload["reason"] = "empty_rewrite"
                payload["llm_provider"] = llm.get("provider")
                payload["llm_model"] = llm.get("model")
                payload["tokens_used"] = llm.get("tokens_used", 0)
                return payload

            rewritten_validation = await validator_service.validate(
                code=rewritten_code,
                framework=framework,
                user_query=user_query,
                rag_context=rag_context,
                compatibility_context=compatibility_context,
                runtime_preferences=prefs,
            )

            before_errors = _error_count(validation_result)
            after_errors = _error_count(rewritten_validation)
            after_deprecations = _deprecation_warnings(rewritten_validation)

            improved = (
                (after_errors < before_errors)
                or (
                    after_errors <= before_errors
                    and len(after_deprecations) < len(before_deprecations)
                )
            )
            if settings.MODERNIZATION_STRICT:
                improved = improved and (len(after_deprecations) == 0 or len(after_deprecations) < len(before_deprecations))

            payload.update(
                {
                    "applied": bool(improved),
                    "reason": "applied" if improved else "no_quality_improvement",
                    "after_deprecation_count": len(after_deprecations),
                    "after_deprecations": after_deprecations,
                    "llm_provider": llm.get("provider"),
                    "llm_model": llm.get("model"),
                    "tokens_used": llm.get("tokens_used", 0),
                }
            )
            if improved:
                payload["code"] = rewritten_code
                payload["validation_result"] = rewritten_validation

            return payload
        except Exception as e:
            logger.warning("Modernization failed framework=%s error=%s", framework, e)
            payload["reason"] = f"error:{e}"
            return payload


modernization_service = ModernizationService()
