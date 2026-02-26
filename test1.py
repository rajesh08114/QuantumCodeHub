#!/usr/bin/env python3
"""
Qiskit-only endpoint retrieval diagnostic.

What it does:
- tests Qiskit-related endpoints
- stores request + response for each endpoint
- extracts retrieval-related metadata (rag/runtime fields) from responses
- writes one JSON report
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


FRAMEWORK = "qiskit"


@dataclass
class EndpointCase:
    name: str
    method: str
    path: str
    json_body: Optional[Dict[str, Any]] = None
    requires_auth: bool = True
    expected_statuses: Optional[List[int]] = None
    profile: str = "default"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def detect_auth_disabled(base_url: str, timeout_seconds: float) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/auth/me", timeout=timeout_seconds)
        return resp.status_code == 200
    except Exception:
        return False


def login_for_token(base_url: str, email: str, password: str, timeout_seconds: float) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/api/auth/login"
    data = {
        "username": email,
        "password": password,
        "grant_type": "password",
    }
    try:
        resp = requests.post(url, data=data, timeout=timeout_seconds)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        token = payload.get("access_token")
        return token if isinstance(token, str) and token.strip() else None
    except Exception:
        return None


def timed_request(
    session: requests.Session,
    method: str,
    url: str,
    timeout_seconds: float,
    **kwargs: Any,
) -> Tuple[Optional[requests.Response], float, Optional[str]]:
    start = time.perf_counter()
    try:
        resp = session.request(method=method, url=url, timeout=timeout_seconds, **kwargs)
        return resp, (time.perf_counter() - start) * 1000.0, None
    except Exception as exc:
        return None, (time.perf_counter() - start) * 1000.0, str(exc)


def parse_response_body(response: Optional[requests.Response], max_chars: int) -> Any:
    if response is None:
        return None
    content_type = str(response.headers.get("content-type", "")).lower()
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            pass
    text = response.text if hasattr(response, "text") else ""
    if len(text) > max_chars:
        text = text[:max_chars] + "...<truncated>"
    return {"raw_text": text}


def runtime_preferences(profile: str) -> Dict[str, Any]:
    # Keep target explicit and modern; backend RAG now defaults to latest when not explicitly constrained.
    mode = "modern"
    if profile == "hardware":
        mode = "modern"
    return {
        "mode": mode,
        "python_version": "3.11",
        "package_versions": {
            "qiskit": "2.3.0",
        },
        "allow_deprecated_apis": False,
    }


def prompt_for(profile: str, variation: str) -> str:
    mode_text = "hardware-compatible" if profile == "hardware" else "simulator-compatible"
    if variation == "ghz":
        return f"Generate runnable qiskit GHZ-state code with measurement ({mode_text})."
    if variation == "qft2":
        return f"Generate runnable qiskit 2-qubit QFT code with measurement ({mode_text})."
    return f"Generate runnable qiskit Bell-state code with measurement ({mode_text})."


def qiskit_code(variation: str) -> str:
    if variation == "ghz":
        return (
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(3)\n"
            "qc.h(0)\n"
            "qc.cx(0,1)\n"
            "qc.cx(1,2)\n"
            "qc.measure_all()"
        )
    if variation == "qft2":
        return (
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(2)\n"
            "qc.h(1)\n"
            "qc.cp(1.5708, 0, 1)\n"
            "qc.h(0)\n"
            "qc.measure_all()"
        )
    return (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2)\n"
        "qc.h(0)\n"
        "qc.cx(0,1)\n"
        "qc.measure_all()"
    )


def broken_qiskit_code() -> Tuple[str, str]:
    return (
        "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0)",
        "TypeError: cx() missing 1 required positional argument",
    )


def completion_prefix() -> str:
    return "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc."


def transpile_source_qiskit() -> str:
    return (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2)\n"
        "qc.h(0)\n"
        "qc.cx(0, 1)\n"
        "qc.measure_all()"
    )


def _truncate_text(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _get_nested(data: Any, path: List[str]) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _normalize_doc_item(item: Any, max_chars: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"text": _truncate_text(text, max_chars)}

    if not isinstance(item, dict):
        return None

    text_value = (
        item.get("content")
        or item.get("text")
        or item.get("document")
        or item.get("chunk")
        or item.get("page_content")
        or ""
    )
    if isinstance(text_value, (list, dict)):
        text_value = json.dumps(text_value, ensure_ascii=False, default=str)
    text_value = str(text_value).strip()
    if not text_value:
        return None

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "text": _truncate_text(text_value, max_chars),
        "score": item.get("score"),
        "distance": item.get("distance"),
        "source": item.get("source") or metadata.get("source"),
        "url": item.get("url") or metadata.get("url"),
        "file_path": item.get("file_path") or metadata.get("file_path"),
        "framework": item.get("framework") or metadata.get("framework"),
        "version": item.get("version") or metadata.get("version"),
        "metadata": metadata,
    }


def _collect_retrieved_docs(body: Dict[str, Any], max_items: int, max_chars: int) -> List[Dict[str, Any]]:
    doc_paths = [
        ["documents"],
        ["retrieved_documents"],
        ["retrieval_documents"],
        ["retrieval", "documents"],
        ["retrieval", "chunks"],
        ["retrieval_metadata", "documents"],
        ["retrieval_metadata", "matched_documents"],
        ["retrieval_metadata", "chunks"],
        ["metadata", "retrieved_documents"],
        ["metadata", "retrieval_documents"],
        ["metadata", "rag_documents"],
    ]

    out: List[Dict[str, Any]] = []
    seen: set = set()

    for path in doc_paths:
        value = _get_nested(body, path)
        if not isinstance(value, list):
            continue
        for item in value:
            normalized = _normalize_doc_item(item, max_chars=max_chars)
            if not normalized:
                continue
            dedup_key = (
                normalized.get("text", ""),
                normalized.get("url", ""),
                normalized.get("file_path", ""),
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            out.append(normalized)
            if len(out) >= max_items:
                return out
    return out


def _collect_retrieval_contexts(body: Dict[str, Any], max_items: int, max_chars: int) -> List[str]:
    context_paths = [
        ["context"],
        ["rag_context"],
        ["retrieved_context"],
        ["retrieval_context"],
        ["retrieval", "context"],
        ["retrieval_metadata", "context"],
        ["metadata", "context"],
        ["metadata", "rag_context"],
        ["metadata", "retrieved_context"],
        ["metadata", "retrieval_context"],
    ]

    out: List[str] = []
    seen: set = set()
    for path in context_paths:
        value = _get_nested(body, path)
        if isinstance(value, str) and value.strip():
            text = _truncate_text(value.strip(), max_chars=max_chars)
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= max_items:
                return out
    return out


def _collect_runtime_evidence(runtime_validation: Dict[str, Any], max_chars: int) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    python_item = runtime_validation.get("python", {}) if isinstance(runtime_validation.get("python"), dict) else {}
    excerpt = python_item.get("evidence_excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        evidence.append({"target": "python", "excerpt": _truncate_text(excerpt.strip(), max_chars=max_chars)})

    packages = runtime_validation.get("packages", {}) if isinstance(runtime_validation.get("packages"), dict) else {}
    for pkg, item in packages.items():
        if not isinstance(item, dict):
            continue
        snippet = item.get("evidence_excerpt")
        if isinstance(snippet, str) and snippet.strip():
            evidence.append({"target": str(pkg), "excerpt": _truncate_text(snippet.strip(), max_chars=max_chars)})
    return evidence


def extract_retrieval_info(
    body: Any,
    max_items: int = 12,
    max_doc_chars: int = 1800,
    max_context_chars: int = 5000,
) -> Dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    metadata = body.get("metadata", {}) if isinstance(body.get("metadata"), dict) else {}
    runtime_validation = metadata.get("runtime_validation", {}) if isinstance(metadata.get("runtime_validation"), dict) else {}
    rag_version = metadata.get("rag_version", {}) if isinstance(metadata.get("rag_version"), dict) else {}
    retrieval_metadata = body.get("retrieval_metadata", {}) if isinstance(body.get("retrieval_metadata"), dict) else {}
    version_filter = retrieval_metadata.get("version_filter", {}) if isinstance(retrieval_metadata.get("version_filter"), dict) else {}
    retrieved_docs = _collect_retrieved_docs(body, max_items=max_items, max_chars=max_doc_chars)
    retrieved_contexts = _collect_retrieval_contexts(body, max_items=max_items, max_chars=max_context_chars)
    runtime_evidence = _collect_runtime_evidence(runtime_validation, max_chars=max_doc_chars)
    return {
        "rag_documents": metadata.get("rag_documents"),
        "rag_average_score": metadata.get("rag_average_score"),
        "rag_version": rag_version,
        "retrieval_version_filter": {
            "selected_version": version_filter.get("selected_version"),
            "latest_version": version_filter.get("latest_version"),
            "strategy": version_filter.get("strategy"),
            "fallback_to_unfiltered": version_filter.get("fallback_to_unfiltered"),
            "active": version_filter.get("active"),
        },
        "runtime_validation": {
            "status": runtime_validation.get("status"),
            "documents_used": runtime_validation.get("documents_used"),
            "rag_average_score": runtime_validation.get("rag_average_score"),
            "error": runtime_validation.get("error"),
        },
        "runtime_requirements": metadata.get("runtime_requirements"),
        "runtime_recommendations": metadata.get("runtime_recommendations"),
        "version_conflicts": metadata.get("version_conflicts"),
        "retrieved_content": {
            "documents": retrieved_docs,
            "contexts": retrieved_contexts,
            "runtime_evidence_excerpts": runtime_evidence,
            "documents_captured": len(retrieved_docs),
            "contexts_captured": len(retrieved_contexts),
        },
    }


def build_cases(auth_disabled: bool) -> List[EndpointCase]:
    cases: List[EndpointCase] = [
        EndpointCase(name="root", method="GET", path="/", requires_auth=False, expected_statuses=[200]),
        EndpointCase(name="health", method="GET", path="/health", requires_auth=False, expected_statuses=[200]),
        EndpointCase(
            name="supported-conversions",
            method="GET",
            path="/api/transpile/supported-conversions",
            requires_auth=False,
            expected_statuses=[200],
        ),
        EndpointCase(name="me", method="GET", path="/api/auth/me", requires_auth=(not auth_disabled), expected_statuses=[200]),
    ]

    for profile in ("simulator", "hardware"):
        prefs = runtime_preferences(profile)
        for variation in ("bell", "ghz", "qft2"):
            broken_code, broken_err = broken_qiskit_code()
            cases.extend(
                [
                    EndpointCase(
                        name=f"code-generate-{variation}-{profile}",
                        method="POST",
                        path="/api/code/generate",
                        expected_statuses=[200],
                        profile=profile,
                        json_body={
                            "prompt": prompt_for(profile, variation),
                            "framework": FRAMEWORK,
                            "include_explanation": True,
                            "runtime_preferences": prefs,
                            "client_context": {"client_type": "api"},
                        },
                    ),
                    EndpointCase(
                        name=f"completion-{variation}-{profile}",
                        method="POST",
                        path="/api/complete/suggest",
                        expected_statuses=[200],
                        profile=profile,
                        json_body={
                            "code_prefix": completion_prefix(),
                            "framework": FRAMEWORK,
                            "cursor_line": 3,
                            "cursor_column": 3,
                            "max_suggestions": 5,
                            "runtime_preferences": prefs,
                            "client_context": {"client_type": "api"},
                        },
                    ),
                    EndpointCase(
                        name=f"explain-{variation}-{profile}",
                        method="POST",
                        path="/api/explain/code",
                        expected_statuses=[200],
                        profile=profile,
                        json_body={
                            "code": qiskit_code(variation),
                            "framework": FRAMEWORK,
                            "detail_level": "intermediate",
                            "include_math": False,
                            "include_visualization": False,
                            "runtime_preferences": prefs,
                            "client_context": {"client_type": "api"},
                        },
                    ),
                    EndpointCase(
                        name=f"fix-{variation}-{profile}",
                        method="POST",
                        path="/api/fix/code",
                        expected_statuses=[200],
                        profile=profile,
                        json_body={
                            "code": broken_code,
                            "framework": FRAMEWORK,
                            "error_message": broken_err,
                            "include_explanation": True,
                            "runtime_preferences": prefs,
                            "client_context": {"client_type": "api"},
                        },
                    ),
                    EndpointCase(
                        name=f"chat-{variation}-{profile}",
                        method="POST",
                        path="/api/chat/message",
                        expected_statuses=[200],
                        profile=profile,
                        json_body={
                            "message": (
                                f"{prompt_for(profile, variation)} "
                                "Also state which qiskit version your response targets."
                            ),
                            "framework": FRAMEWORK,
                            "detail_level": "intermediate",
                            "new_session": True,
                            "include_other_session_summaries": False,
                            "runtime_preferences": prefs,
                            "client_context": {"client_type": "api"},
                        },
                    ),
                ]
            )

        for target_fw in ("pennylane", "cirq"):
            cases.append(
                EndpointCase(
                    name=f"transpile-qiskit-to-{target_fw}-{profile}",
                    method="POST",
                    path="/api/transpile/convert",
                    expected_statuses=[200],
                    profile=profile,
                    json_body={
                        "source_code": transpile_source_qiskit(),
                        "source_framework": FRAMEWORK,
                        "target_framework": target_fw,
                        "preserve_comments": True,
                        "optimize": False,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )
    return cases


def run_suite(
    base_url: str,
    token: Optional[str],
    timeout_seconds: float,
    auth_disabled: bool,
    max_response_chars: int,
) -> Dict[str, Any]:
    session = requests.Session()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    cases = build_cases(auth_disabled=auth_disabled)
    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "base_url": base_url,
        "framework_scope": "qiskit",
        "auth_disabled": auth_disabled,
        "cases_total": len(cases),
        "results": [],
        "summary": {},
    }

    latencies: List[float] = []
    passed = 0
    failed = 0

    for case in cases:
        url = f"{base_url.rstrip('/')}{case.path}"
        if case.requires_auth and not auth_disabled and not token:
            report["results"].append(
                {
                    "name": case.name,
                    "path": case.path,
                    "method": case.method,
                    "status": "SKIP",
                    "reason": "auth required but no token provided",
                    "profile": case.profile,
                    "request": {
                        "url": url,
                        "json": case.json_body,
                    },
                }
            )
            continue

        request_kwargs: Dict[str, Any] = {}
        if case.json_body is not None:
            request_kwargs["json"] = case.json_body

        response, latency_ms, error = timed_request(
            session=session,
            method=case.method,
            url=url,
            timeout_seconds=timeout_seconds,
            **request_kwargs,
        )
        status = response.status_code if response is not None else "ERROR"
        expected = case.expected_statuses or [200]
        ok = response is not None and response.status_code in expected

        response_body = parse_response_body(response, max_chars=max_response_chars)
        retrieval_info = extract_retrieval_info(response_body)

        entry = {
            "name": case.name,
            "path": case.path,
            "method": case.method,
            "profile": case.profile,
            "status": status,
            "ok": ok,
            "expected_statuses": expected,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "request": {
                "url": url,
                "json": case.json_body,
            },
            "response": {
                "headers": dict(response.headers) if response is not None else {},
                "body": response_body,
            },
            "retrieval_observed": retrieval_info,
        }
        report["results"].append(entry)

        if isinstance(latency_ms, (int, float)):
            latencies.append(float(latency_ms))
        if ok:
            passed += 1
        else:
            failed += 1

    executed = [r for r in report["results"] if r.get("status") != "SKIP"]
    report["summary"] = {
        "executed": len(executed),
        "passed": passed,
        "failed": failed,
        "success_rate": round((passed / len(executed)), 4) if executed else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Qiskit-only endpoint retrieval tester")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--auth-disabled", action="store_true")
    parser.add_argument("--max-response-chars", type=int, default=60000)
    parser.add_argument("--output", default="quantumcodehub_qiskit_retrieval_results.json")
    args = parser.parse_args()

    token = args.token.strip() or None
    if not token and args.email.strip() and args.password:
        token = login_for_token(args.base_url, args.email.strip(), args.password, args.timeout)

    detected_auth_disabled = detect_auth_disabled(args.base_url, args.timeout)
    auth_disabled = bool(args.auth_disabled or detected_auth_disabled)

    report = run_suite(
        base_url=args.base_url,
        token=token,
        timeout_seconds=args.timeout,
        auth_disabled=auth_disabled,
        max_response_chars=max(2000, int(args.max_response_chars)),
    )

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report.get("summary", {}), indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
