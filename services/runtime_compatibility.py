"""
Runtime compatibility utilities for client-aware generation and retrieval.

Version recommendations are derived from:
1) RAG retrieval of version/compatibility docs.
2) LLM synthesis of structured recommendations.
3) Validation of those recommendations against retrieved docs.
"""
import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from schemas.common import ClientContext, RuntimePreferences

logger = logging.getLogger(__name__)


def _normalize_client_type(value: str) -> str:
    candidate = (value or "").strip().lower()
    aliases = {
        "web": "website",
        "website": "website",
        "site": "website",
        "vscode": "vscode_extension",
        "vscode_extension": "vscode_extension",
        "extension": "vscode_extension",
        "api": "api",
    }
    return aliases.get(candidate, "api")


def _normalize_runtime_mode(value: str) -> str:
    candidate = (value or "").strip().lower()
    if candidate in {"auto", "modern", "legacy"}:
        return candidate
    return "auto"


def _parse_version_tuple(value: str) -> Optional[Tuple[int, ...]]:
    parts = re.findall(r"\d+", value or "")
    if not parts:
        return None
    return tuple(int(part) for part in parts[:4])


def _compare_versions(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    max_len = max(len(left), len(right))
    lhs = left + (0,) * (max_len - len(left))
    rhs = right + (0,) * (max_len - len(right))
    if lhs < rhs:
        return -1
    if lhs > rhs:
        return 1
    return 0


def _is_version_in_spec(version_value: str, spec: str) -> Optional[bool]:
    installed = _parse_version_tuple(version_value)
    if installed is None:
        return None

    clauses = [item.strip() for item in (spec or "").split(",") if item.strip()]
    if not clauses:
        return None

    for clause in clauses:
        dash_range = re.match(
            r"^([0-9][0-9A-Za-z.\-+]*)\s*-\s*([0-9][0-9A-Za-z.\-+]*)$",
            clause,
        )
        if dash_range:
            lower = _parse_version_tuple(dash_range.group(1))
            upper = _parse_version_tuple(dash_range.group(2))
            if lower is None or upper is None:
                return None
            if _compare_versions(installed, lower) < 0 or _compare_versions(installed, upper) > 0:
                return False
            continue

        match = re.match(r"^(>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.\-+]*)$", clause)
        operator = "=="
        target_text = clause
        if match:
            operator = match.group(1)
            target_text = match.group(2)
        target = _parse_version_tuple(target_text)
        if target is None:
            return None

        cmp_value = _compare_versions(installed, target)
        if operator == ">" and not (cmp_value > 0):
            return False
        if operator == ">=" and not (cmp_value >= 0):
            return False
        if operator == "<" and not (cmp_value < 0):
            return False
        if operator == "<=" and not (cmp_value <= 0):
            return False
        if operator == "==" and not (cmp_value == 0):
            return False
        if operator == "!=" and not (cmp_value != 0):
            return False
    return True


def _extract_version_tokens(text: str) -> List[str]:
    seen = set()
    tokens = []
    for token in re.findall(r"\d+(?:\.\d+){0,2}", text or ""):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _split_sentences(text: str) -> List[str]:
    lines = re.split(r"(?<=[.;\n])\s+", text or "")
    return [line.strip() for line in lines if line and line.strip()]


def _target_mentions_sentence(target: str, sentence: str) -> bool:
    target_key = (target or "").strip().lower()
    candidate = (sentence or "").strip().lower()
    if not target_key or not candidate:
        return False
    if target_key == "python":
        return "python" in candidate
    variants = {
        target_key,
        target_key.replace("-", "_"),
        target_key.replace("_", "-"),
    }
    return any(variant and variant in candidate for variant in variants)


def _collect_target_evidence(target: str, documents: List[Dict]) -> List[str]:
    evidence_sentences: List[str] = []
    for doc in documents or []:
        for sentence in _split_sentences(doc.get("content", "")):
            if _target_mentions_sentence(target, sentence):
                evidence_sentences.append(sentence)
    return evidence_sentences


def _validate_suggestion_against_docs(
    target: str,
    suggestion: str,
    documents: List[Dict],
) -> Dict:
    normalized = (suggestion or "").strip()
    result = {
        "target": target,
        "suggested": normalized,
        "validated": False,
        "coverage": 0.0,
        "matched_tokens": [],
        "evidence_excerpt": "",
        "reason": "",
    }
    if not normalized:
        result["reason"] = "empty_suggestion"
        return result

    suggestion_tokens = set(_extract_version_tokens(normalized))
    if not suggestion_tokens:
        result["reason"] = "no_version_tokens_in_suggestion"
        return result

    evidence_sentences = _collect_target_evidence(target, documents)
    if not evidence_sentences:
        result["reason"] = "target_not_found_in_docs"
        return result

    evidence_text = " ".join(evidence_sentences)
    doc_tokens = set(_extract_version_tokens(evidence_text))
    overlap = sorted(token for token in suggestion_tokens if token in doc_tokens)
    coverage = len(overlap) / float(max(1, len(suggestion_tokens)))

    validated = bool(overlap) and coverage >= 0.6
    if validated:
        result["evidence_excerpt"] = evidence_sentences[0][:220]
    else:
        result["reason"] = "insufficient_token_overlap"

    result["validated"] = validated
    result["coverage"] = round(coverage, 4)
    result["matched_tokens"] = overlap
    return result


def _validate_requested_runtime_against_docs(requested_runtime: Dict, documents: List[Dict]) -> Dict:
    requested = requested_runtime or {}
    python_request = (requested.get("python") or "").strip()
    package_requests = requested.get("packages") or {}

    python_validation = _validate_suggestion_against_docs(
        target="python",
        suggestion=python_request,
        documents=documents,
    )
    package_validations = {}
    for package, spec in sorted(package_requests.items()):
        package_validations[package] = _validate_suggestion_against_docs(
            target=package,
            suggestion=spec,
            documents=documents,
        )

    validations = [python_validation] + list(package_validations.values())
    validations = [item for item in validations if (item.get("suggested") or "").strip()]
    all_validated = bool(validations) and all(item.get("validated") for item in validations)
    any_validated = any(item.get("validated") for item in validations) if validations else False

    return {
        "python": python_validation,
        "packages": package_validations,
        "all_validated": all_validated,
        "any_validated": any_validated,
    }


def _requested_runtime_conflicts(requested_validation: Dict) -> List[str]:
    conflicts: List[str] = []
    if not isinstance(requested_validation, dict):
        return conflicts
    python_validation = requested_validation.get("python", {}) or {}
    if (python_validation.get("suggested") or "").strip() and not python_validation.get("validated"):
        conflicts.append(
            f"Requested python target '{python_validation.get('suggested')}' not validated by retrieved docs "
            f"({python_validation.get('reason', 'unknown_reason')})"
        )

    for package, item in sorted((requested_validation.get("packages") or {}).items()):
        if (item.get("suggested") or "").strip() and not item.get("validated"):
            conflicts.append(
                f"Requested {package} target '{item.get('suggested')}' not validated by retrieved docs "
                f"({item.get('reason', 'unknown_reason')})"
            )
    return conflicts


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


def _normalize_recommendations(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        return {"python": "", "packages": {}}

    python_spec = payload.get("python")
    safe_python = python_spec.strip() if isinstance(python_spec, str) else ""

    raw_packages = payload.get("packages")
    safe_packages: Dict[str, str] = {}
    if isinstance(raw_packages, dict):
        for package, spec in raw_packages.items():
            if not isinstance(package, str) or not isinstance(spec, str):
                continue
            package_name = package.strip().lower()
            version_spec = spec.strip()
            if package_name and version_spec:
                safe_packages[package_name] = version_spec

    return {
        "python": safe_python,
        "packages": safe_packages,
    }


def _normalize_runtime_preferences(
    framework: str,
    runtime_preferences: Optional[RuntimePreferences],
) -> Dict:
    prefs = runtime_preferences or RuntimePreferences()
    requested_packages = {
        k.strip().lower(): v.strip()
        for k, v in (prefs.package_versions or {}).items()
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
    }
    framework_key = (framework or "").strip().lower()
    if prefs.framework_version and framework_key and framework_key not in requested_packages:
        requested_packages[framework_key] = prefs.framework_version.strip()

    requested_python = (prefs.python_version or "").strip()
    return {
        "mode": _normalize_runtime_mode(prefs.mode),
        "python": requested_python,
        "packages": requested_packages,
        "allow_deprecated_apis": bool(prefs.allow_deprecated_apis),
    }


def _has_runtime_values(runtime_target: Dict) -> bool:
    target = runtime_target or {}
    return bool((target.get("python") or "").strip() or (target.get("packages") or {}))


def _merge_runtime_targets(validated_recommendations: Dict, requested_runtime: Dict) -> Dict:
    validated = validated_recommendations or {}
    requested = requested_runtime or {}
    merged_packages = dict(validated.get("packages") or {})
    merged_packages.update(requested.get("packages") or {})
    merged_python = (requested.get("python") or "").strip() or (validated.get("python") or "").strip()
    return {
        "python": merged_python,
        "packages": merged_packages,
    }


def _build_runtime_retrieval_query(
    framework: str,
    installed_packages: Dict[str, str],
    requested_runtime: Dict,
) -> str:
    package_text = (
        ", ".join(f"{name}={version}" for name, version in sorted(installed_packages.items()))
        if installed_packages
        else "none reported"
    )
    requested = requested_runtime or {}
    requested_python = (requested.get("python") or "").strip() or "none"
    requested_packages = requested.get("packages") or {}
    requested_package_text = (
        ", ".join(f"{name}={version}" for name, version in sorted(requested_packages.items()))
        if requested_packages
        else "none"
    )
    requested_mode = requested.get("mode") or "auto"
    return (
        f"{framework} official compatibility matrix supported python versions package version requirements "
        f"minimum version maximum version deprecations breaking changes. "
        f"Client installed packages: {package_text}. "
        f"User requested runtime mode={requested_mode}; python={requested_python}; packages={requested_package_text}. "
        "Prioritize documentation that contains explicit version numbers or version ranges."
    )


def _build_recommendation_prompt(
    framework: str,
    rag_context: str,
    client_context: Dict,
    runtime_preferences: Dict,
) -> str:
    context_json = json.dumps(client_context, sort_keys=True)
    request_json = json.dumps(runtime_preferences or {}, sort_keys=True)
    return f"""Task: infer runtime version recommendations for {framework} using ONLY the documentation context.

Documentation context:
{rag_context}

Client context:
{context_json}

User requested runtime target:
{request_json}

Return strict JSON (no markdown, no prose):
{{
  "python": "<version range or empty string>",
  "packages": {{
    "<package-name>": "<version spec>"
  }}
}}

Rules:
1. Use only versions that appear explicitly in the documentation context.
2. If the user requested explicit versions, prioritize those when supported by docs.
3. If uncertain, use empty string for python and omit that package key.
4. Keep package names lowercase.
5. Do not invent versions not present in docs.
"""


def _build_conflicts(
    python_version: Optional[str],
    installed_packages: Dict[str, str],
    recommendations: Dict,
) -> List[str]:
    conflicts: List[str] = []

    python_spec = (recommendations or {}).get("python", "")
    if python_version and python_spec:
        compatibility = _is_version_in_spec(python_version, python_spec)
        if compatibility is False:
            conflicts.append(
                f"python={python_version} is outside recommended range {python_spec}"
            )

    recommended_packages = (recommendations or {}).get("packages", {}) or {}
    for package, installed_version in sorted(installed_packages.items()):
        recommended_spec = recommended_packages.get(package)
        if not recommended_spec:
            continue
        compatibility = _is_version_in_spec(installed_version, recommended_spec)
        if compatibility is False:
            conflicts.append(
                f"{package}={installed_version} is outside recommended range {recommended_spec}"
            )

    return conflicts


def _build_rag_query_suffix(bundle: Dict, framework: str) -> str:
    recommendations = bundle.get("runtime_recommendations") or {}
    requested_runtime = bundle.get("requested_runtime") or {}
    python_target = recommendations.get("python") or "unknown"
    packages = recommendations.get("packages") or {}
    package_target = (
        "; ".join(f"{name}:{spec}" for name, spec in sorted(packages.items()))
        if packages
        else "unknown"
    )
    requested_python = (requested_runtime.get("python") or "").strip() or "none"
    requested_packages = requested_runtime.get("packages") or {}
    requested_package_text = (
        "; ".join(f"{name}:{spec}" for name, spec in sorted(requested_packages.items()))
        if requested_packages
        else "none"
    )
    installed = bundle.get("client_context", {}).get("installed_packages", {}) or {}
    installed_text = (
        "; ".join(f"{name}:{spec}" for name, spec in sorted(installed.items()))
        if installed
        else "none"
    )
    validation = bundle.get("runtime_validation", {})
    validation_state = validation.get("status", "unknown")

    return (
        f"Compatibility signals for {framework}: "
        f"requested_mode={requested_runtime.get('mode', 'auto')}; "
        f"requested_python={requested_python}; "
        f"requested_packages={requested_package_text}; "
        f"recommended_python={python_target}; "
        f"recommended_packages={package_target}; "
        f"installed_packages={installed_text}; "
        f"client_type={bundle.get('client_type')}; "
        f"validation_status={validation_state}"
    )


def _build_compatibility_context(bundle: Dict, framework: str) -> str:
    client = bundle.get("client_context", {})
    recommendations = bundle.get("runtime_recommendations") or {}
    requested_runtime = bundle.get("requested_runtime") or {}
    packages = recommendations.get("packages") or {}
    validation = bundle.get("runtime_validation", {})
    conflicts = bundle.get("version_conflicts") or []

    lines = [
        f"Framework: {framework}",
        f"Client type: {bundle.get('client_type', 'api')}",
    ]

    if client.get("python_version"):
        lines.append(f"Client reported Python version: {client.get('python_version')}")
    if client.get("client_version"):
        lines.append(f"Client version: {client.get('client_version')}")
    if client.get("extension_installed") is not None:
        lines.append(f"VS Code extension installed: {client.get('extension_installed')}")
    if client.get("extension_version"):
        lines.append(f"VS Code extension version: {client.get('extension_version')}")

    installed_packages = client.get("installed_packages") or {}
    if installed_packages:
        lines.append("Detected installed packages:")
        for package, version in sorted(installed_packages.items()):
            lines.append(f"- {package}: {version}")

    requested_python = (requested_runtime.get("python") or "").strip()
    requested_packages = requested_runtime.get("packages") or {}
    if requested_python or requested_packages:
        lines.append("User-requested runtime target:")
        lines.append(f"- mode: {requested_runtime.get('mode', 'auto')}")
        if requested_python:
            lines.append(f"- python: {requested_python}")
        for package, spec in sorted(requested_packages.items()):
            lines.append(f"- {package}: {spec}")
        lines.append(
            f"- allow_deprecated_apis: {bool(requested_runtime.get('allow_deprecated_apis', False))}"
        )

    if recommendations.get("python") or packages:
        lines.append("RAG + LLM runtime recommendations (validated against retrieved docs):")
        if recommendations.get("python"):
            lines.append(f"- python: {recommendations['python']}")
        for package, spec in sorted(packages.items()):
            lines.append(f"- {package}: {spec}")
    else:
        lines.append("No validated runtime recommendations were extracted from retrieved docs.")

    if conflicts:
        lines.append("Potential version conflicts detected:")
        for conflict in conflicts:
            lines.append(f"- {conflict}")

    lines.append(
        f"Runtime recommendation validation status: {validation.get('status', 'unknown')}"
    )
    mode = requested_runtime.get("mode", "auto")
    if mode == "legacy":
        lines.append(
            "Respect the user-requested legacy runtime target; generate APIs compatible with those versions."
        )
        lines.append(
            "Do not force modernization when it conflicts with requested runtime unless code is objectively broken."
        )
    elif mode == "modern":
        lines.append(
            "Prefer modern stable APIs unless explicit user constraints conflict."
        )
    if bundle.get("client_type") == "vscode_extension":
        lines.append(
            "Generate code compatible with validated recommendations and avoid APIs deprecated in those versions."
        )
    else:
        lines.append(
            "Include runtime recommendations only when they are validated by retrieved documentation evidence."
        )

    return "\n".join(lines)


def _build_cache_fingerprint(bundle: Dict, framework: str) -> str:
    client = bundle.get("client_context", {})
    rec = bundle.get("runtime_recommendations") or {}
    requested = bundle.get("requested_runtime") or {}
    packages = rec.get("packages") or {}
    validation = bundle.get("runtime_validation") or {}

    material = "|".join(
        [
            framework,
            bundle.get("client_type") or "",
            client.get("python_version") or "",
            client.get("client_version") or "",
            client.get("extension_version") or "",
            ",".join(f"{k}={v}" for k, v in sorted((client.get("installed_packages") or {}).items())),
            requested.get("mode") or "",
            requested.get("python") or "",
            ",".join(f"{k}:{v}" for k, v in sorted((requested.get("packages") or {}).items())),
            str(bool(requested.get("allow_deprecated_apis", False))),
            rec.get("python") or "",
            ",".join(f"{k}:{v}" for k, v in sorted(packages.items())),
            validation.get("status") or "",
            str(validation.get("documents_used", 0)),
        ]
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _status_with_runtime_request(bundle: Dict, default_status: str) -> str:
    if _has_runtime_values(bundle.get("requested_runtime", {}) or {}):
        return "requested_runtime_unverified"
    return default_status


def build_runtime_bundle(
    framework: str,
    client_context: Optional[ClientContext],
    runtime_preferences: Optional[RuntimePreferences] = None,
) -> Dict:
    """
    Build a client-aware runtime bundle without hardcoded version recommendations.
    """
    safe_framework = (framework or "").strip().lower()
    ctx = client_context or ClientContext()
    client_type = _normalize_client_type(ctx.client_type)
    requested_runtime = _normalize_runtime_preferences(
        framework=safe_framework,
        runtime_preferences=runtime_preferences,
    )

    installed_packages = {k.lower(): v for k, v in (ctx.installed_packages or {}).items() if v}
    if ctx.framework_version and safe_framework and safe_framework not in installed_packages:
        installed_packages[safe_framework] = ctx.framework_version

    bundle = {
        "client_type": client_type,
        "runtime_recommendations": {
            "python": requested_runtime.get("python", ""),
            "packages": dict(requested_runtime.get("packages", {})),
        },
        "requested_runtime": requested_runtime,
        "effective_runtime_target": {
            "python": requested_runtime.get("python", ""),
            "packages": requested_runtime.get("packages", {}),
        },
        "version_conflicts": [],
        "runtime_validation": {
            "source": "rag+llm",
            "status": "pending",
            "documents_used": 0,
            "rag_average_score": 0.0,
            "python": {},
            "packages": {},
            "raw_suggestions": {"python": "", "packages": {}},
            "requested": {},
        },
        "client_context": {
            "client_type": client_type,
            "client_version": ctx.client_version,
            "extension_installed": ctx.extension_installed,
            "extension_version": ctx.extension_version,
            "python_version": ctx.python_version,
            "framework_version": ctx.framework_version,
            "installed_packages": installed_packages,
        },
    }
    bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
    bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
    bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
    return bundle


async def build_runtime_bundle_with_rag(
    framework: str,
    client_context: Optional[ClientContext],
    runtime_preferences: Optional[RuntimePreferences] = None,
    request_source: str = "unknown",
) -> Dict:
    """
    Build runtime bundle using RAG retrieval + LLM suggestions + doc-grounded validation.
    """
    safe_framework = (framework or "").strip().lower()
    bundle = build_runtime_bundle(safe_framework, client_context, runtime_preferences=runtime_preferences)

    try:
        from services.llm_service import llm_service
        from services.rag_service import rag_service
        from services.rag_guardrails import ensure_rag_consistency
    except Exception as import_error:
        logger.warning("Runtime recommendation services unavailable: %s", import_error)
        bundle["runtime_validation"]["status"] = _status_with_runtime_request(
            bundle,
            "service_import_failed",
        )
        bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
        bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
        bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
        return bundle

    try:
        rag_results = await rag_service.retrieve_context(
            query=_build_runtime_retrieval_query(
                safe_framework,
                bundle.get("client_context", {}).get("installed_packages", {}) or {},
                bundle.get("requested_runtime", {}) or {},
            ),
            framework=safe_framework,
            top_k=6,
            request_source=f"{request_source}#runtime_recommendation",
            runtime_preferences=bundle.get("requested_runtime", {}),
            prefer_latest_version=True,
        )
        rag_error = str(rag_results.get("error") or "").strip()
        if rag_error:
            bundle["runtime_validation"]["status"] = _status_with_runtime_request(
                bundle,
                "rag_retrieval_error",
            )
            bundle["runtime_validation"]["error"] = rag_error
            bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
            bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
            bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
            return bundle
        try:
            ensure_rag_consistency(
                rag_results=rag_results,
                framework=safe_framework,
                runtime_preferences=bundle.get("requested_runtime", {}),
                prefer_latest_version=True,
                require_documents=2,
                allow_unfiltered_fallback=False,
            )
        except Exception as rag_consistency_error:
            bundle["runtime_validation"]["status"] = _status_with_runtime_request(
                bundle,
                "rag_consistency_failed",
            )
            bundle["runtime_validation"]["error"] = str(rag_consistency_error)
            bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
            bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
            bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
            return bundle

        documents = rag_results.get("documents", []) or []
        bundle["runtime_validation"]["documents_used"] = len(documents)
        bundle["runtime_validation"]["rag_average_score"] = rag_results.get("average_score", 0.0)

        rag_context = rag_results.get("context", "")
        if not rag_context.strip():
            bundle["runtime_validation"]["status"] = _status_with_runtime_request(
                bundle,
                "no_rag_context",
            )
            bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
            bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
            bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
            return bundle

        llm_response = await llm_service.generate_code(
            prompt=_build_recommendation_prompt(
                framework=safe_framework,
                rag_context=rag_context,
                client_context=bundle.get("client_context", {}),
                runtime_preferences=bundle.get("requested_runtime", {}),
            ),
            max_tokens=420,
            temperature=0.0,
        )
        raw_suggestion_text = llm_response.get("generated_text", "")
        parsed_payload = _extract_json_payload(raw_suggestion_text)
        if not parsed_payload:
            bundle["runtime_validation"]["status"] = _status_with_runtime_request(
                bundle,
                "llm_parse_failed",
            )
            bundle["runtime_validation"]["raw_llm_text"] = (raw_suggestion_text or "")[:900]
            bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
            bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
            bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
            return bundle

        raw_recommendations = _normalize_recommendations(parsed_payload)
        bundle["runtime_validation"]["raw_suggestions"] = raw_recommendations

        python_validation = _validate_suggestion_against_docs(
            target="python",
            suggestion=raw_recommendations.get("python", ""),
            documents=documents,
        )

        package_validations = {}
        validated_packages = {}
        for package, spec in sorted(raw_recommendations.get("packages", {}).items()):
            package_validation = _validate_suggestion_against_docs(
                target=package,
                suggestion=spec,
                documents=documents,
            )
            package_validations[package] = package_validation
            if package_validation.get("validated"):
                validated_packages[package] = spec

        validated_recommendations = {
            "python": raw_recommendations.get("python", "") if python_validation.get("validated") else "",
            "packages": validated_packages,
        }
        requested_runtime = bundle.get("requested_runtime", {}) or {}
        requested_validation = _validate_requested_runtime_against_docs(
            requested_runtime=requested_runtime,
            documents=documents,
        )
        effective_target = _merge_runtime_targets(
            validated_recommendations=validated_recommendations,
            requested_runtime=requested_runtime,
        )
        conflicts = _build_conflicts(
            python_version=bundle.get("client_context", {}).get("python_version"),
            installed_packages=bundle.get("client_context", {}).get("installed_packages", {}) or {},
            recommendations=effective_target,
        )
        conflicts.extend(_requested_runtime_conflicts(requested_validation))

        requested_has_values = _has_runtime_values(requested_runtime)
        any_effective = _has_runtime_values(effective_target)
        bundle["runtime_recommendations"] = effective_target
        bundle["effective_runtime_target"] = effective_target
        bundle["version_conflicts"] = conflicts
        bundle["runtime_validation"]["python"] = python_validation
        bundle["runtime_validation"]["packages"] = package_validations
        bundle["runtime_validation"]["validated_recommendations"] = validated_recommendations
        bundle["runtime_validation"]["requested"] = requested_validation
        if requested_has_values:
            if requested_validation.get("all_validated"):
                status = "requested_runtime_validated"
            elif requested_validation.get("any_validated"):
                status = "requested_runtime_partially_validated"
            else:
                status = "requested_runtime_not_validated"
        else:
            status = "validated" if any_effective else "no_validated_suggestions"
        bundle["runtime_validation"]["status"] = status
        bundle["runtime_validation"]["llm_provider"] = llm_response.get("provider")
        bundle["runtime_validation"]["llm_model"] = llm_response.get("model")
        bundle["runtime_validation"]["llm_attempt"] = llm_response.get("attempt")

    except Exception as runtime_error:
        logger.warning("Failed to build runtime recommendation from RAG/LLM: %s", runtime_error)
        bundle["runtime_validation"]["status"] = _status_with_runtime_request(
            bundle,
            "runtime_recommendation_failed",
        )
        bundle["runtime_validation"]["error"] = str(runtime_error)

    bundle["compatibility_context"] = _build_compatibility_context(bundle, safe_framework)
    bundle["rag_query_suffix"] = _build_rag_query_suffix(bundle, safe_framework)
    bundle["cache_fingerprint"] = _build_cache_fingerprint(bundle, safe_framework)
    return bundle
