"""
Code validation service.
"""
from ml.validators.qiskit_validator import QiskitValidator
from ml.validators.pennylane_validator import PennyLaneValidator
from ml.validators.cirq_validator import CirqValidator
from ml.validators.torchquantum_validator import TorchQuantumValidator
from ml.validators.base_validator import BaseValidator
from typing import Dict, Optional
import logging

from core.config import settings
from services.code_evaluation_service import code_evaluation_service

logger = logging.getLogger(__name__)


def _build_validation_rag_query(framework: str, user_query: str, code: str) -> str:
    intent = (user_query or "").strip() or "Validate generated quantum program correctness."
    code_preview = " ".join((code or "").split())
    if len(code_preview) > 900:
        code_preview = f"{code_preview[:900]}..."
    return (
        f"{intent}\n\n"
        f"Validate this {framework} code for API correctness, measurement/output correctness, "
        f"deprecation-safe usage, and runtime compatibility:\n{code_preview}"
    )

class ValidatorService:
    """Service for code validation across frameworks"""
    
    def __init__(self):
        self.validators = {
            "qiskit": QiskitValidator(),
            "pennylane": PennyLaneValidator(),
            "cirq": CirqValidator(),
            "torchquantum": TorchQuantumValidator(),
        }
    
    async def validate(
        self,
        code: str,
        framework: str,
        user_query: str = "",
        rag_context: str = "",
        compatibility_context: str = "",
        runtime_preferences: Optional[Dict] = None,
    ) -> Dict:
        """
        Validate code for specific framework
        
        Args:
            code: Code to validate
            framework: Framework name
            
        Returns:
            Validation results
        """
        try:
            mode = (settings.VALIDATION_MODE or "hybrid").strip().lower()
            use_static = mode in {"hybrid", "static", ""}
            use_llm = mode in {"hybrid", "llm"} and bool(settings.VALIDATION_ENABLE_LLM_EVAL)

            validator = self.validators.get(framework)
            
            if not validator:
                logger.warning(f"No validator for framework: {framework}")
                # Use base validator as fallback
                validator = BaseValidator(framework)
            
            static_result = {
                "passed": True,
                "errors": [],
                "warnings": [],
                "framework": framework,
            }
            if use_static:
                static_result = await validator.validate(code)

            errors = list(static_result.get("errors", []))
            warnings = list(static_result.get("warnings", []))

            llm_evaluation = {
                "enabled": False,
                "status": "disabled_by_mode",
                "passed": True,
                "score": None,
                "critical_issues": [],
                "warnings": [],
                "improvements": [],
            }
            should_run_llm = use_llm and (
                bool(static_result.get("passed", False))
                or not bool(settings.VALIDATION_LLM_ON_STATIC_PASS_ONLY)
            )
            if should_run_llm:
                validation_rag_context = (rag_context or "").strip()
                should_fetch_validation_rag = bool(settings.VALIDATION_USE_RAG) and (
                    not validation_rag_context
                    or bool(settings.VALIDATION_RAG_AUGMENT_WHEN_CONTEXT_PRESENT)
                )
                if should_fetch_validation_rag:
                    try:
                        from services.rag_service import rag_service
                        from services.rag_guardrails import ensure_rag_consistency

                        rag_payload = await rag_service.retrieve_context(
                            query=_build_validation_rag_query(framework, user_query, code),
                            framework=framework,
                            top_k=max(1, int(settings.VALIDATION_RAG_TOP_K or 3)),
                            request_source="/validation/llm",
                            runtime_preferences=runtime_preferences,
                            prefer_latest_version=True,
                        )
                        retrieved_context = (rag_payload.get("context") or "").strip()
                        rag_error = str(rag_payload.get("error") or "").strip()
                        if rag_error:
                            logger.warning(
                                "Validation RAG strict retrieval error framework=%s error=%s",
                                framework,
                                rag_error,
                            )
                        else:
                            try:
                                ensure_rag_consistency(
                                    rag_results=rag_payload,
                                    framework=framework,
                                    runtime_preferences=runtime_preferences,
                                    prefer_latest_version=True,
                                    require_documents=1,
                                    allow_unfiltered_fallback=False,
                                )
                            except Exception as rag_consistency_error:
                                logger.warning(
                                    "Validation RAG consistency failed framework=%s error=%s",
                                    framework,
                                    rag_consistency_error,
                                )
                        if retrieved_context:
                            if validation_rag_context and settings.VALIDATION_RAG_AUGMENT_WHEN_CONTEXT_PRESENT:
                                validation_rag_context = (
                                    f"{validation_rag_context}\n\nValidation-specific context:\n{retrieved_context}"
                                )
                            else:
                                validation_rag_context = retrieved_context
                    except Exception as rag_error:
                        logger.warning("Validation RAG retrieval failed framework=%s error=%s", framework, rag_error)

                llm_evaluation = await code_evaluation_service.evaluate(
                    code=code,
                    framework=framework,
                    user_query=user_query,
                    rag_context=validation_rag_context,
                    compatibility_context=compatibility_context,
                )
            elif use_llm:
                llm_evaluation["status"] = "skipped_static_failed"

            llm_critical = llm_evaluation.get("critical_issues", [])
            llm_warnings = llm_evaluation.get("warnings", [])
            llm_improvements = llm_evaluation.get("improvements", [])
            warnings.extend(llm_warnings)

            if settings.VALIDATION_FAIL_ON_LLM_CRITICAL:
                errors.extend([f"LLM evaluation: {issue}" for issue in llm_critical])

            passed = len(errors) == 0 and bool(static_result.get("passed", False))
            if settings.VALIDATION_REQUIRE_LLM_PASS and llm_evaluation.get("enabled"):
                passed = passed and bool(llm_evaluation.get("passed", True))

            return {
                "passed": passed,
                "errors": errors,
                "warnings": warnings,
                "framework": framework,
                "evaluation": {
                    "mode": mode,
                    "static_passed": bool(static_result.get("passed", False)),
                    "llm": llm_evaluation,
                    "critical_issue_count": len(llm_critical),
                    "improvements": llm_improvements,
                },
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "passed": False,
                "errors": [str(e)],
                "warnings": [],
                "framework": framework,
                "evaluation": {
                    "mode": settings.VALIDATION_MODE,
                    "status": "error",
                },
            }

# Singleton instance
validator_service = ValidatorService()
