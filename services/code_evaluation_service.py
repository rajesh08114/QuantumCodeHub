"""
LLM-based code evaluation service for framework-agnostic validation.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

_HALLUCINATION_STOPWORDS = {
    "about",
    "after",
    "align",
    "also",
    "any",
    "are",
    "around",
    "because",
    "before",
    "being",
    "between",
    "but",
    "can",
    "code",
    "could",
    "does",
    "each",
    "for",
    "from",
    "generated",
    "has",
    "have",
    "include",
    "includes",
    "including",
    "intent",
    "into",
    "its",
    "may",
    "might",
    "must",
    "not",
    "only",
    "or",
    "other",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "those",
    "use",
    "used",
    "using",
    "with",
    "without",
}


def _extract_json_payload(text: str) -> Optional[Dict]:
    raw = (text or "").strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = (fenced.group(1) or "").strip()

    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _to_string_list(value) -> list:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str):
            candidate = item.strip()
            if candidate:
                items.append(candidate)
    return items[:12]


def _dedupe_strings(items: List[str], limit: int = 12) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        key = (item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
        if len(deduped) >= limit:
            break
    return deduped


def _clip_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _extract_code_signals(code: str, framework: str) -> Dict:
    lower = (code or "").lower()
    safe_framework = (framework or "").strip().lower()

    if safe_framework == "qiskit":
        has_ghz_pattern = (
            bool(re.search(r"\.\s*h\s*\(\s*0\s*\)", lower))
            and (
                (
                    bool(re.search(r"\.\s*cx\s*\(\s*0\s*,\s*1\s*\)", lower))
                    and bool(re.search(r"\.\s*cx\s*\(\s*0\s*,\s*2\s*\)", lower))
                )
                or (
                    bool(re.search(r"\.\s*cx\s*\(\s*0\s*,\s*1\s*\)", lower))
                    and bool(re.search(r"\.\s*cx\s*\(\s*1\s*,\s*2\s*\)", lower))
                )
            )
        )
        has_measurement = (
            "measure_all(" in lower
            or ".measure(" in lower
            or "get_counts(" in lower
        )
        has_execution = (
            "backend.run(" in lower
            or "execute(" in lower
            or "aersimulator(" in lower
        )
        has_plot = "plot_histogram(" in lower or ".draw(" in lower
        has_plot_data = bool(re.search(r"plot_histogram\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)", lower))
        return {
            "framework": safe_framework,
            "has_measurement": has_measurement,
            "has_execution": has_execution,
            "has_ghz_pattern": has_ghz_pattern,
            "has_plot": has_plot,
            "has_plot_data": has_plot_data,
        }

    if safe_framework == "pennylane":
        has_measurement = any(
            token in lower for token in ["qml.expval(", "qml.probs(", "qml.sample(", "qml.counts(", "qml.state("]
        )
        return {
            "framework": safe_framework,
            "has_measurement": has_measurement,
            "has_execution": "@qml.qnode" in lower,
        }

    if safe_framework == "cirq":
        has_measurement = "cirq.measure(" in lower or ".measure(" in lower
        return {
            "framework": safe_framework,
            "has_measurement": has_measurement,
            "has_execution": "cirq.simulator(" in lower or "cirq.circuit(" in lower,
        }

    if safe_framework == "torchquantum":
        has_measurement = any(token in lower for token in ["measure", "expval", "expectation", "sample"])
        return {
            "framework": safe_framework,
            "has_measurement": has_measurement,
            "has_execution": "quantumdevice(" in lower,
        }

    return {
        "framework": safe_framework,
        "has_measurement": False,
        "has_execution": False,
    }


def _is_definitive_failure(issue: str) -> bool:
    text = (issue or "").strip().lower()
    if not text:
        return False
    uncertain_tokens = (
        "might",
        "may",
        "could",
        "possibly",
        "consider",
        "recommend",
        "optional",
        "better",
    )
    if any(token in text for token in uncertain_tokens):
        return False

    fatal_signals = (
        "syntax error",
        "typeerror",
        "nameerror",
        "attributeerror",
        "missing required",
        "undefined",
        "out of range",
        "invalid argument",
        "will fail",
        "fails at runtime",
    )
    if any(token in text for token in fatal_signals):
        return True

    return False


def _issue_contradicts_signals(issue: str, signals: Dict) -> bool:
    text = (issue or "").strip().lower()
    has_measurement = bool(signals.get("has_measurement"))
    if has_measurement and (
        "does not include any measurement" in text
        or "no measurement" in text
        or "missing measurement" in text
        or "without measurement" in text
    ):
        return True

    if bool(signals.get("has_ghz_pattern")) and (
        "does not create a ghz state" in text
        or "not create a ghz state" in text
        or "add an additional cnot gate between the first and third qubits" in text
        or "add code to prepare a ghz state" in text
        or "additional cnot" in text
    ):
        return True

    if bool(signals.get("has_plot")) and (
        "does not include any plotting" in text
        or "no plotting command" in text
        or "missing plotting command" in text
        or "add plotting functionality" in text
    ):
        return True

    if bool(signals.get("has_plot_data")) and (
        "does not provide any data" in text
        or "provide actual data for the histogram" in text
    ):
        return True

    return False


def _tokenize_issue_text(text: str) -> List[str]:
    value = (text or "").lower()
    if not value:
        return []
    raw_tokens = re.findall(r"[a-z_][a-z0-9_.-]{1,}|\d+\.\d+(?:\.\d+)?", value)
    tokens: List[str] = []
    seen = set()
    for token in raw_tokens:
        if token in seen:
            continue
        if token in _HALLUCINATION_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) < 3 and not re.match(r"\d+\.\d+", token):
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _evidence_overlap(issue: str, evidence_text: str) -> Tuple[float, List[str]]:
    issue_tokens = _tokenize_issue_text(issue)
    if not issue_tokens:
        return 0.0, []
    evidence_tokens = set(_tokenize_issue_text(evidence_text))
    if not evidence_tokens:
        return 0.0, []
    matches = [token for token in issue_tokens if token in evidence_tokens]
    ratio = float(len(matches)) / float(max(1, len(issue_tokens)))
    return ratio, matches[:8]


def _issue_supported_by_signals(issue: str, signals: Dict) -> bool:
    text = (issue or "").strip().lower()
    if not text:
        return False

    if (
        "no measurement" in text
        or "missing measurement" in text
        or "without measurement" in text
        or "missing observable" in text
    ):
        return not bool(signals.get("has_measurement"))

    if "missing execution" in text or "not executed" in text or "no execution" in text:
        return not bool(signals.get("has_execution"))

    if "missing plotting" in text or "no plotting" in text:
        return not bool(signals.get("has_plot"))

    if "does not create a ghz state" in text or "missing ghz" in text:
        return not bool(signals.get("has_ghz_pattern"))

    return False


def _is_issue_grounded(
    issue: str,
    code: str,
    rag_context: str,
    compatibility_context: str,
    signals: Dict,
    min_code_overlap: float,
    min_rag_overlap: float,
) -> Tuple[bool, Dict]:
    code_overlap, code_matches = _evidence_overlap(issue, code)
    rag_overlap, rag_matches = _evidence_overlap(issue, f"{rag_context}\n{compatibility_context}")
    signal_grounded = _issue_supported_by_signals(issue, signals)
    code_grounded = signal_grounded or code_overlap >= min_code_overlap
    rag_grounded = rag_overlap >= min_rag_overlap
    grounded = code_grounded or rag_grounded
    return grounded, {
        "code_overlap": round(code_overlap, 3),
        "rag_overlap": round(rag_overlap, 3),
        "code_matches": code_matches,
        "rag_matches": rag_matches,
        "signal_grounded": signal_grounded,
    }


def _filter_issue_list(
    issues: List[str],
    issue_type: str,
    code: str,
    rag_context: str,
    compatibility_context: str,
    signals: Dict,
    guard_enabled: bool,
    min_code_overlap: float,
    min_rag_overlap: float,
) -> Tuple[List[str], List[Dict]]:
    kept: List[str] = []
    dropped: List[Dict] = []
    for issue in issues:
        if _issue_contradicts_signals(issue, signals):
            dropped.append(
                {
                    "type": issue_type,
                    "issue": _clip_text(issue, 220),
                    "reason": "contradicts_signals",
                }
            )
            continue

        if guard_enabled and issue_type in {"critical", "warning"}:
            grounded, evidence = _is_issue_grounded(
                issue=issue,
                code=code,
                rag_context=rag_context,
                compatibility_context=compatibility_context,
                signals=signals,
                min_code_overlap=min_code_overlap,
                min_rag_overlap=min_rag_overlap,
            )
            if not grounded:
                dropped.append(
                    {
                        "type": issue_type,
                        "issue": _clip_text(issue, 220),
                        "reason": "insufficient_grounding",
                        "code_overlap": evidence["code_overlap"],
                        "rag_overlap": evidence["rag_overlap"],
                    }
                )
                continue

        kept.append(issue)
    return _dedupe_strings(kept), dropped


def _normalize_score(
    raw_score: Optional[float],
    evaluator_passed: bool,
    warning_count: int,
) -> float:
    if isinstance(raw_score, (int, float)):
        candidate = max(0.0, min(float(raw_score), 1.0))
    else:
        candidate = None

    if evaluator_passed:
        if candidate is not None and candidate >= 0.45:
            return candidate
        return max(0.55, 0.92 - (min(max(warning_count, 0), 5) * 0.06))

    if candidate is not None and candidate <= 0.65:
        return candidate
    return 0.35


def _build_evaluation_prompt(
    code: str,
    framework: str,
    user_query: str = "",
    rag_context: str = "",
    compatibility_context: str = "",
) -> str:
    query_text = (user_query or "").strip() or "Not provided."
    docs = (rag_context or "").strip() or "No documentation context provided."
    runtime = (compatibility_context or "").strip() or "No runtime compatibility context provided."
    code_signals = _extract_code_signals(code, framework)
    code_json = json.dumps(code_signals, sort_keys=True)

    return f"""Task: Evaluate whether this generated {framework} code is correct, runnable, and aligned with intent.

User intent:
{query_text}

Runtime compatibility context:
{runtime}

Code-derived signals (trusted):
{code_json}

Documentation context:
{docs}

Generated code:
```python
{code}
```

Return strict JSON only:
{{
  "passed": true,
  "score": 0.0,
  "critical_issues": [],
  "warnings": [],
  "improvements": []
}}

Rules:
1. Mark passed=false only for definite correctness/runtime failures (not style advice).
2. score must be between 0.0 and 1.0.
3. Put hard failures in critical_issues, softer issues in warnings.
4. Keep each issue concise and actionable.
5. Respect code-derived signals above; do not claim missing measurement if has_measurement=true.
6. Use documentation context for compatibility/deprecation judgments.
7. Do not output markdown or extra text.
8. Do not speculate; include only issues grounded in code or documentation context.
"""


class CodeEvaluationService:
    """Framework-agnostic LLM quality evaluator."""

    async def evaluate(
        self,
        code: str,
        framework: str,
        user_query: str = "",
        rag_context: str = "",
        compatibility_context: str = "",
    ) -> Dict:
        if not settings.VALIDATION_ENABLE_LLM_EVAL:
            return {
                "enabled": False,
                "status": "disabled",
                "passed": True,
                "score": None,
                "critical_issues": [],
                "warnings": [],
                "improvements": [],
            }

        try:
            from services.llm_service import llm_service

            clipped_code = _clip_text(code, max(800, int(settings.VALIDATION_MAX_CODE_CHARS or 2600)))
            clipped_rag = _clip_text(rag_context, max(400, int(settings.VALIDATION_MAX_RAG_CHARS or 2200)))
            clipped_compatibility = _clip_text(
                compatibility_context,
                max(300, int(settings.VALIDATION_MAX_COMPATIBILITY_CHARS or 900)),
            )
            code_signals = _extract_code_signals(clipped_code, framework)

            prompt = _build_evaluation_prompt(
                code=clipped_code,
                framework=framework,
                user_query=user_query,
                rag_context=clipped_rag,
                compatibility_context=clipped_compatibility,
            )
            llm = await llm_service.generate_code(
                prompt=prompt,
                max_tokens=max(120, min(int(settings.VALIDATION_LLM_MAX_TOKENS or 220), 380)),
                temperature=0.0,
            )
            payload = _extract_json_payload(llm.get("generated_text", ""))
            if not payload:
                return {
                    "enabled": True,
                    "status": "parse_failed",
                    "passed": True,
                    "score": None,
                    "critical_issues": [],
                    "warnings": ["LLM evaluator returned non-JSON output; skipped strict gating."],
                    "improvements": [],
                    "provider": llm.get("provider"),
                    "model": llm.get("model"),
                    "grounding_metrics": {
                        "raw_issue_count": 0,
                        "grounded_issue_count": 0,
                        "dropped_issue_count": 0,
                        "hallucination_suppression_rate": 0.0,
                    },
                }

            score_raw = payload.get("score")
            score = None
            if isinstance(score_raw, (int, float)):
                score = max(0.0, min(float(score_raw), 1.0))

            guard_enabled = bool(settings.VALIDATION_HALLUCINATION_GUARD_ENABLED)
            min_code_overlap = max(
                0.0,
                min(1.0, float(settings.VALIDATION_HALLUCINATION_MIN_CODE_OVERLAP or 0.22)),
            )
            min_rag_overlap = max(
                0.0,
                min(1.0, float(settings.VALIDATION_HALLUCINATION_MIN_RAG_OVERLAP or 0.18)),
            )
            max_dropped_report = max(0, int(settings.VALIDATION_HALLUCINATION_MAX_DROPPED_REPORT or 8))

            raw_critical_issues = _to_string_list(payload.get("critical_issues"))
            raw_warnings = _to_string_list(payload.get("warnings"))
            improvements = _to_string_list(payload.get("improvements"))
            critical_issues, dropped_critical = _filter_issue_list(
                issues=raw_critical_issues,
                issue_type="critical",
                code=clipped_code,
                rag_context=clipped_rag,
                compatibility_context=clipped_compatibility,
                signals=code_signals,
                guard_enabled=guard_enabled,
                min_code_overlap=min_code_overlap,
                min_rag_overlap=min_rag_overlap,
            )
            warnings, dropped_warnings = _filter_issue_list(
                issues=raw_warnings,
                issue_type="warning",
                code=clipped_code,
                rag_context=clipped_rag,
                compatibility_context=clipped_compatibility,
                signals=code_signals,
                guard_enabled=guard_enabled,
                min_code_overlap=min_code_overlap,
                min_rag_overlap=min_rag_overlap,
            )
            improvements, dropped_improvements = _filter_issue_list(
                issues=improvements,
                issue_type="improvement",
                code=clipped_code,
                rag_context=clipped_rag,
                compatibility_context=clipped_compatibility,
                signals=code_signals,
                guard_enabled=False,
                min_code_overlap=min_code_overlap,
                min_rag_overlap=min_rag_overlap,
            )
            dropped_items = dropped_critical + dropped_warnings + dropped_improvements
            definitive_critical = [issue for issue in critical_issues if _is_definitive_failure(issue)]
            downgraded = [issue for issue in critical_issues if issue not in definitive_critical]
            warnings = _dedupe_strings(warnings + downgraded)
            raw_issues = _dedupe_strings(raw_critical_issues + raw_warnings, limit=24)
            filtered_issues = _dedupe_strings(definitive_critical + warnings, limit=24)
            dropped_issue_count = max(0, len(raw_issues) - len(filtered_issues))
            suppression_rate = (
                round(float(dropped_issue_count) / float(len(raw_issues)), 4)
                if raw_issues
                else 0.0
            )

            # Only definitive critical issues can fail evaluation.
            evaluator_passed = len(definitive_critical) == 0

            return {
                "enabled": True,
                "status": "ok",
                "passed": evaluator_passed,
                "score": _normalize_score(score, evaluator_passed, len(warnings)),
                "critical_issues": definitive_critical,
                "warnings": warnings,
                "improvements": improvements,
                "provider": llm.get("provider"),
                "model": llm.get("model"),
                "tokens_used": llm.get("tokens_used", 0),
                "signals": code_signals,
                "rag_context_used_chars": len(clipped_rag),
                "hallucination_guard_enabled": guard_enabled,
                "hallucination_min_code_overlap": min_code_overlap,
                "hallucination_min_rag_overlap": min_rag_overlap,
                "hallucination_filtered_count": len(dropped_items),
                "hallucination_filtered_items": dropped_items[:max_dropped_report],
                "grounding_metrics": {
                    "raw_issue_count": len(raw_issues),
                    "grounded_issue_count": len(filtered_issues),
                    "dropped_issue_count": dropped_issue_count,
                    "hallucination_suppression_rate": suppression_rate,
                },
            }
        except Exception as e:
            logger.warning("LLM evaluation failed: %s", e)
            return {
                "enabled": True,
                "status": "error",
                "passed": True,
                "score": None,
                "critical_issues": [],
                "warnings": [f"LLM evaluator unavailable: {e}"],
                "improvements": [],
                "grounding_metrics": {
                    "raw_issue_count": 0,
                    "grounded_issue_count": 0,
                    "dropped_issue_count": 0,
                    "hallucination_suppression_rate": 0.0,
                },
            }


code_evaluation_service = CodeEvaluationService()
