"""
Domain guard tester for QuantumCodeHub endpoints.

Purpose:
- validate that non-quantum requests are blocked with:
  {"detail": "not quantum domain"}
- validate chat non-quantum fallback path metadata
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class TestCase:
    name: str
    method: str
    path: str
    body: Optional[Dict[str, Any]] = None
    requires_auth: bool = True
    expected_status: int = 400
    expected_detail: Optional[str] = "not quantum domain"
    expect_chat_warning: bool = False


def detect_auth_disabled(base_url: str, timeout_s: float) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/auth/me", timeout=timeout_s)
        return response.status_code == 200
    except Exception:
        return False


def try_login(base_url: str, email: str, password: str, timeout_s: float) -> Optional[str]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/auth/login",
            data={"username": email, "password": password, "grant_type": "password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout_s,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        token = payload.get("access_token")
        return token if isinstance(token, str) and token.strip() else None
    except Exception:
        return None


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw_text": (response.text or "")[:2000]}


def build_test_cases() -> List[TestCase]:
    return [
        TestCase(
            name="code-generate-non-quantum",
            method="POST",
            path="/api/code/generate",
            body={
                "prompt": "Write a Flask API endpoint for user registration.",
                "framework": "qiskit",
                "include_explanation": False,
                "client_context": {"client_type": "api"},
            },
        ),
        TestCase(
            name="complete-non-quantum",
            method="POST",
            path="/api/complete/suggest",
            body={
                "code_prefix": "def add(a, b):\n    return a + b\n",
                "framework": "qiskit",
                "cursor_line": 2,
                "cursor_column": 4,
                "max_suggestions": 3,
                "client_context": {"client_type": "api"},
            },
        ),
        TestCase(
            name="explain-non-quantum",
            method="POST",
            path="/api/explain/code",
            body={
                "code": "def add(a, b):\n    return a + b",
                "framework": "qiskit",
                "detail_level": "intermediate",
                "include_math": False,
                "client_context": {"client_type": "api"},
            },
        ),
        TestCase(
            name="fix-non-quantum",
            method="POST",
            path="/api/fix/code",
            body={
                "code": "def add(a, b)\n    return a + b",
                "framework": "qiskit",
                "error_message": "SyntaxError: invalid syntax",
                "include_explanation": True,
                "client_context": {"client_type": "api"},
            },
        ),
        TestCase(
            name="transpile-non-quantum",
            method="POST",
            path="/api/transpile/convert",
            body={
                "source_code": "def add(a, b):\n    return a + b",
                "source_framework": "qiskit",
                "target_framework": "cirq",
                "preserve_comments": True,
                "optimize": False,
                "client_context": {"client_type": "api"},
            },
        ),
        TestCase(
            name="chat-non-quantum-fallback",
            method="POST",
            path="/api/chat/message",
            body={
                "message": "Design a PostgreSQL schema for ecommerce orders.",
                "detail_level": "advanced",
                "new_session": True,
                "client_context": {"client_type": "api"},
            },
            expected_status=200,
            expected_detail=None,
            expect_chat_warning=True,
        ),
    ]


def run_case(
    base_url: str,
    case: TestCase,
    token: Optional[str],
    auth_disabled: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    if case.requires_auth and not auth_disabled and not token:
        return {
            "name": case.name,
            "ok": False,
            "status": "SKIP",
            "error": "missing auth token",
            "response": None,
            "latency_ms": None,
        }

    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{base_url.rstrip('/')}{case.path}"
    start = time.perf_counter()
    try:
        response = requests.request(
            method=case.method,
            url=url,
            headers=headers,
            json=case.body,
            timeout=timeout_s,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        payload = _safe_json(response)

        ok = response.status_code == case.expected_status
        error = ""

        if ok and case.expected_detail is not None:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            ok = detail == case.expected_detail
            if not ok:
                error = f"expected detail '{case.expected_detail}', got '{detail}'"

        if ok and case.expect_chat_warning:
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            rag_skipped = bool(metadata.get("rag_skipped"))
            warning = metadata.get("warning")
            if (not rag_skipped) or (not isinstance(warning, str) or not warning.strip()):
                ok = False
                error = "chat non-quantum fallback metadata missing (rag_skipped/warning)"

        if not ok and not error:
            error = f"unexpected status={response.status_code}, payload={str(payload)[:500]}"

        return {
            "name": case.name,
            "ok": ok,
            "status": response.status_code,
            "error": error or None,
            "response": payload,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "name": case.name,
            "ok": False,
            "status": "ERR",
            "error": str(exc),
            "response": None,
            "latency_ms": latency_ms,
        }


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\nDomain Guard Test Results")
    print("-" * 110)
    print(f"{'NAME':36} {'STATUS':10} {'LATENCY_MS':12} RESULT  ERROR")
    print("-" * 110)
    for item in results:
        result = "PASS" if item["ok"] else "FAIL"
        latency = "-" if item["latency_ms"] is None else str(item["latency_ms"])
        error = (item.get("error") or "")[:48]
        print(f"{item['name'][:36]:36} {str(item['status']):10} {latency:12} {result:6} {error}")
    print("-" * 110)
    executed = [r for r in results if r["status"] != "SKIP"]
    passed = [r for r in executed if r["ok"]]
    failed = [r for r in executed if not r["ok"]]
    skipped = [r for r in results if r["status"] == "SKIP"]
    print(f"Executed: {len(executed)}  Passed: {len(passed)}  Failed: {len(failed)}  Skipped: {len(skipped)}")


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test non-quantum domain guard behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--token", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--output-json", default="domain_guard_test_results.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    auth_disabled = detect_auth_disabled(args.base_url, args.timeout)
    token = args.token.strip() or None
    if not token and not auth_disabled and args.email and args.password:
        token = try_login(args.base_url, args.email, args.password, args.timeout)

    cases = build_test_cases()
    started = int(time.time())
    results = [run_case(args.base_url, case, token, auth_disabled, args.timeout) for case in cases]
    print_summary(results)

    report = {
        "generated_at_epoch": started,
        "base_url": args.base_url.rstrip("/"),
        "auth_disabled": auth_disabled,
        "token_used": bool(token),
        "results": results,
    }
    output_path = Path(args.output_json).resolve()
    write_report(output_path, report)
    print(f"JSON report: {output_path}")

    failed = [item for item in results if item["status"] != "SKIP" and not item["ok"]]
    if args.fail_on_error and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
