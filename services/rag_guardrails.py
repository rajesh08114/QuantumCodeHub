"""
RAG retrieval guardrails for consistent, version-aware behavior.
"""
from __future__ import annotations

from typing import Dict, Optional


_FRAMEWORKS_WITH_VERSIONING = {
    "qiskit",
    "pennylane",
    "cirq",
    "torchquantum",
    "tensorflow_quantum",
}


def _normalize_framework_name(framework: str) -> str:
    return (framework or "").strip().lower()


def _extract_framework_version_spec(
    framework: str,
    runtime_preferences: Optional[Dict],
    version_constraint: Optional[str] = None,
) -> str:
    if version_constraint and str(version_constraint).strip():
        return str(version_constraint).strip()

    prefs = runtime_preferences or {}
    requested = prefs.get("packages") or prefs.get("package_versions") or {}
    safe_framework = _normalize_framework_name(framework)
    if isinstance(requested, dict):
        for package_name, spec in requested.items():
            package_key = (package_name or "").strip().lower()
            if package_key == safe_framework and str(spec or "").strip():
                return str(spec).strip()

    framework_spec = prefs.get("framework_version")
    if isinstance(framework_spec, str) and framework_spec.strip():
        return framework_spec.strip()
    return ""


def build_version_enforcement_context(
    framework: str,
    rag_results: Dict,
) -> str:
    safe_framework = _normalize_framework_name(framework)
    retrieval_metadata = rag_results.get("retrieval_metadata") or {}
    version_filter = retrieval_metadata.get("version_filter") or {}
    selected = str(version_filter.get("selected_version") or "").strip()
    latest = str(version_filter.get("latest_version") or "").strip()
    if safe_framework in _FRAMEWORKS_WITH_VERSIONING and selected:
        latest_hint = f" latest_available={latest}." if latest else ""
        return (
            f"Strict version target: use only {safe_framework} APIs compatible with version {selected}."
            f"{latest_hint} Avoid deprecated or legacy APIs outside this version."
        )
    return ""


def ensure_rag_consistency(
    rag_results: Dict,
    framework: str,
    runtime_preferences: Optional[Dict] = None,
    version_constraint: Optional[str] = None,
    prefer_latest_version: bool = True,
    require_documents: int = 1,
    allow_unfiltered_fallback: bool = False,
) -> Dict:
    safe_framework = _normalize_framework_name(framework)
    docs = rag_results.get("documents") or []
    if len(docs) < max(0, int(require_documents)):
        raise ValueError(
            f"RAG returned insufficient documents for framework '{safe_framework}': "
            f"required={require_documents}, got={len(docs)}."
        )

    retrieval_metadata = rag_results.get("retrieval_metadata") or {}
    version_filter = retrieval_metadata.get("version_filter") or {}

    info = {
        "framework": safe_framework,
        "documents": len(docs),
        "selected_version": str(version_filter.get("selected_version") or "").strip(),
        "latest_version": str(version_filter.get("latest_version") or "").strip(),
        "strategy": str(version_filter.get("strategy") or "").strip(),
        "active": bool(version_filter.get("active", False)),
        "fallback_to_unfiltered": bool(version_filter.get("fallback_to_unfiltered", False)),
    }

    if safe_framework not in _FRAMEWORKS_WITH_VERSIONING:
        return info

    if not info["active"] or not info["selected_version"]:
        raise ValueError(
            f"RAG version filter is not active for framework '{safe_framework}'. "
            f"strategy={info['strategy'] or 'unknown'}."
        )

    if info["fallback_to_unfiltered"] and not allow_unfiltered_fallback:
        raise ValueError(
            f"RAG fell back to unfiltered retrieval for framework '{safe_framework}', "
            "which is disallowed for strict accuracy mode."
        )

    explicit_spec = _extract_framework_version_spec(
        framework=safe_framework,
        runtime_preferences=runtime_preferences,
        version_constraint=version_constraint,
    )
    if not explicit_spec and prefer_latest_version:
        latest = info["latest_version"]
        selected = info["selected_version"]
        if latest and selected and selected != latest:
            raise ValueError(
                f"RAG selected version '{selected}' instead of latest '{latest}' "
                f"for framework '{safe_framework}'."
            )

    selected_version = info["selected_version"]
    if selected_version:
        mismatched = 0
        for doc in docs:
            metadata = (doc or {}).get("metadata") or {}
            doc_version = str(metadata.get("version") or "").strip()
            if doc_version and doc_version != selected_version:
                mismatched += 1
        if mismatched > 0:
            raise ValueError(
                f"RAG returned {mismatched} documents with mismatched versions for "
                f"framework '{safe_framework}' (selected={selected_version})."
            )

    return info

