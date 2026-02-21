"""
Code transpilation service.
"""
from typing import Dict, List
from services.llm_service import llm_service
from services.rag_service import rag_service
from ml.prompts import TranspilationPrompts
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class TranspilerService:
    """Service for framework-to-framework code transpilation"""

    def __init__(self):
        self.supported_conversions = {
            ("qiskit", "pennylane"): True,
            ("qiskit", "cirq"): True,
            ("pennylane", "qiskit"): True,
            ("pennylane", "cirq"): True,
            ("cirq", "qiskit"): True,
            ("cirq", "pennylane"): True,
        }

    async def transpile(
        self,
        source_code: str,
        source_framework: str,
        target_framework: str,
        preserve_comments: bool = True,
        optimize: bool = False,
        compatibility_context: str = "",
        rag_query_suffix: str = "",
    ) -> Dict:
        """
        Transpile code from one framework to another

        Args:
            source_code: Source code to transpile
            source_framework: Source framework name
            target_framework: Target framework name
            preserve_comments: Keep code comments
            optimize: Apply circuit optimization

        Returns:
            Dict with transpiled code and metadata
        """
        try:
            source_code = (source_code or "").strip()
            source_framework = (source_framework or "").strip().lower()
            target_framework = (target_framework or "").strip().lower()

            if not source_code:
                return {
                    "code": "",
                    "success": False,
                    "method": "input",
                    "differences": [],
                    "warnings": ["source_code is required"],
                    "tokens_used": 0
                }

            if not source_framework:
                return {
                    "code": "",
                    "success": False,
                    "method": "input",
                    "differences": [],
                    "warnings": ["source_framework is required"],
                    "tokens_used": 0
                }

            if not target_framework:
                return {
                    "code": "",
                    "success": False,
                    "method": "input",
                    "differences": [],
                    "warnings": ["target_framework is required"],
                    "tokens_used": 0
                }

            # Check if conversion is supported
            if (source_framework, target_framework) not in self.supported_conversions:
                return {
                    "code": "",
                    "success": False,
                    "method": "input",
                    "differences": [],
                    "warnings": [
                        f"Conversion {source_framework} -> {target_framework} not supported"
                    ],
                    "tokens_used": 0
                }

            # Get target framework documentation
            rag_query = f"convert quantum circuit to {target_framework}"
            if rag_query_suffix:
                rag_query = f"{rag_query}\n\n{rag_query_suffix}"

            rag_results = await rag_service.retrieve_context(
                query=rag_query,
                framework=target_framework,
                top_k=5,
                request_source="/api/transpile/convert",
            )

            # Build transpilation prompt
            prompt = TranspilationPrompts.build_transpilation_prompt(
                source_code=source_code,
                source_framework=source_framework,
                target_framework=target_framework,
                rag_context=rag_results["context"],
                compatibility_context=(
                    f"{compatibility_context}\n"
                    f"Preserve comments: {preserve_comments}\n"
                    f"Optimize output circuit: {optimize}"
                ).strip(),
            )

            # Generate transpiled code
            llm_response = await llm_service.generate_code(
                prompt=prompt,
                max_tokens=settings.TRANSPILATION_MAX_TOKENS,
                temperature=0.1  # Low temperature for accuracy
            )
            logger.info(
                "Transpilation LLM response provider=%s model=%s tokens=%s fallback_used=%s",
                llm_response.get("provider"),
                llm_response.get("model"),
                llm_response.get("tokens_used"),
                llm_response.get("fallback_used", False),
            )

            # Extract code
            transpiled_code = self._extract_code(llm_response["generated_text"])

            # Detect differences
            differences = self._detect_differences(
                source_code,
                transpiled_code,
            )

            return {
                "code": transpiled_code,
                "success": True,
                "method": "llm",
                "differences": differences,
                "warnings": [],
                "tokens_used": llm_response["tokens_used"],
                "llm_provider": llm_response.get("provider"),
                "llm_model": llm_response.get("model"),
                "llm_attempt": llm_response.get("attempt"),
                "llm_fallback_used": llm_response.get("fallback_used", False),
            }

        except Exception as e:
            logger.exception(f"Transpilation error: {e}")
            return {
                "code": "",
                "success": False,
                "method": "llm",
                "differences": [],
                "warnings": [str(e)],
                "tokens_used": 0
            }

    def _extract_code(self, text: str) -> str:
        """Extract code from LLM response"""
        import re
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _detect_differences(
        self,
        source: str,
        target: str,
    ) -> List[str]:
        """Detect notable differences between source and target"""
        differences = []

        # Check for structural differences
        if "QuantumCircuit" in source and "@qml.qnode" in target:
            differences.append(
                "Converted imperative circuit to functional QNode style"
            )

        return differences

# Singleton instance
transpiler_service = TranspilerService()
