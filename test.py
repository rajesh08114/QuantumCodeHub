"""
Endpoint smoke/performance runner for QuantumCodeHub backend.

Usage examples:
  python test.py
  python test.py --base-url http://127.0.0.1:8000 --token <JWT>
  python test.py --email test@example.com --password StrongPass123!
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class EndpointCase:
    name: str
    method: str
    path: str
    requires_auth: bool = False
    json_body: Optional[Dict] = None
    form_body: Optional[Dict] = None
    expected_statuses: Optional[List[int]] = None


def _build_cases() -> List[EndpointCase]:
    return [
        EndpointCase(name="root", method="GET", path="/", expected_statuses=[200]),
        EndpointCase(name="health", method="GET", path="/health", expected_statuses=[200]),
        EndpointCase(
            name="supported-conversions",
            method="GET",
            path="/api/transpile/supported-conversions",
            expected_statuses=[200],
        ),
        EndpointCase(name="me", method="GET", path="/api/auth/me", requires_auth=True, expected_statuses=[200]),
        EndpointCase(
            name="code-generate",
            method="POST",
            path="/api/code/generate",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "prompt": "Create a 2-qubit Bell state circuit",
                "framework": "qiskit",
                "include_explanation": False,
                "client_context": {"client_type": "api"},
            },
        ),
        EndpointCase(
            name="transpile-convert",
            method="POST",
            path="/api/transpile/convert",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "source_code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)",
                "source_framework": "qiskit",
                "target_framework": "pennylane",
                "preserve_comments": True,
                "optimize": False,
                "client_context": {"client_type": "api"},
            },
        ),
        EndpointCase(
            name="completion-suggest",
            method="POST",
            path="/api/complete/suggest",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "code_prefix": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.",
                "framework": "qiskit",
                "cursor_line": 3,
                "cursor_column": 3,
                "max_suggestions": 3,
                "client_context": {"client_type": "api"},
            },
        ),
        EndpointCase(
            name="explain-code",
            method="POST",
            path="/api/explain/code",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)",
                "framework": "qiskit",
                "detail_level": "intermediate",
                "include_math": False,
                "include_visualization": False,
                "client_context": {"client_type": "api"},
            },
        ),
        EndpointCase(
            name="fix-code",
            method="POST",
            path="/api/fix/code",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0)",
                "framework": "qiskit",
                "error_message": "TypeError: cx() missing 1 required positional argument",
                "include_explanation": True,
                "client_context": {"client_type": "api"},
            },
        ),
        EndpointCase(
            name="chat-message",
            method="POST",
            path="/api/chat/message",
            requires_auth=True,
            expected_statuses=[200],
            json_body={
                "message": "Explain Bell state in simple terms",
                "framework": "qiskit",
                "detail_level": "beginner",
                "new_session": False,
                "include_other_session_summaries": False,
                "client_context": {"client_type": "api"},
            },
        ),
    ]


def _try_login(base_url: str, email: str, password: str, timeout: float) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/api/auth/login"
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            data={"username": email, "password": password, "grant_type": "password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code != 200:
            print(f"[auth] login failed ({response.status_code}) in {elapsed_ms:.1f} ms")
            return None
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            print("[auth] login response did not include access_token")
            return None
        print(f"[auth] login success in {elapsed_ms:.1f} ms")
        return token
    except Exception as exc:
        print(f"[auth] login error: {exc}")
        return None


def _detect_auth_disabled(base_url: str, timeout: float) -> bool:
    """
    Best-effort auth mode detection.
    If /api/auth/me is reachable without token (HTTP 200), auth is treated as disabled.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/auth/me", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _run_case(
    case: EndpointCase,
    base_url: str,
    token: Optional[str],
    timeout: float,
    auth_disabled: bool,
) -> Dict:
    if case.requires_auth and not token and not auth_disabled:
        return {
            "name": case.name,
            "method": case.method,
            "path": case.path,
            "status": "SKIP",
            "time_ms": None,
            "ok": False,
            "error": "missing auth token",
        }

    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if case.form_body:
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    url = f"{base_url.rstrip('/')}{case.path}"
    start = time.perf_counter()
    try:
        response = requests.request(
            method=case.method,
            url=url,
            headers=headers,
            json=case.json_body,
            data=case.form_body,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        expected = case.expected_statuses or [200]
        ok = response.status_code in expected

        error = None
        if not ok:
            body = response.text
            error = body[:300].replace("\n", " ")

        return {
            "name": case.name,
            "method": case.method,
            "path": case.path,
            "status": response.status_code,
            "time_ms": elapsed_ms,
            "ok": ok,
            "error": error,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "name": case.name,
            "method": case.method,
            "path": case.path,
            "status": "ERR",
            "time_ms": elapsed_ms,
            "ok": False,
            "error": str(exc),
        }


def _print_results(results: List[Dict]) -> None:
    print("\nEndpoint Test Results")
    print("-" * 108)
    print(f"{'NAME':22} {'METHOD':7} {'STATUS':8} {'TIME_MS':10} PATH")
    print("-" * 108)
    for item in results:
        time_ms = "-" if item["time_ms"] is None else f"{item['time_ms']:.1f}"
        print(f"{item['name'][:22]:22} {item['method']:7} {str(item['status']):8} {time_ms:10} {item['path']}")
    print("-" * 108)

    executed = [r for r in results if r["status"] not in ("SKIP",)]
    passed = [r for r in executed if r["ok"]]
    failed = [r for r in executed if not r["ok"]]
    skipped = [r for r in results if r["status"] == "SKIP"]
    latencies = [r["time_ms"] for r in executed if isinstance(r["time_ms"], (int, float))]

    avg_ms = statistics.mean(latencies) if latencies else 0.0
    p95_ms = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)

    print(f"Executed: {len(executed)}  Passed: {len(passed)}  Failed: {len(failed)}  Skipped: {len(skipped)}")
    print(f"Avg latency: {avg_ms:.1f} ms  P95 latency: {p95_ms:.1f} ms")

    if failed:
        print("\nFailures")
        for item in failed:
            print(f"- {item['name']} ({item['status']}): {item['error'] or 'unknown error'}")

    if skipped:
        print("\nSkipped")
        for item in skipped:
            print(f"- {item['name']}: {item['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantumCodeHub endpoint tester with response times.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--token", default="", help="JWT token for protected endpoints")
    parser.add_argument("--email", default="", help="Login email (optional)")
    parser.add_argument("--password", default="", help="Login password (optional)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    parser.add_argument(
        "--auth-disabled",
        action="store_true",
        help="Force auth-disabled mode (run protected endpoints without token).",
    )
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero exit code if any endpoint fails")
    args = parser.parse_args()

    auth_disabled = args.auth_disabled
    if not auth_disabled:
        auth_disabled = _detect_auth_disabled(args.base_url, args.timeout)
        print(f"[auth] detected auth_disabled={auth_disabled}")

    token = args.token.strip() or None
    if not token and not auth_disabled and args.email and args.password:
        token = _try_login(
            base_url=args.base_url,
            email=args.email,
            password=args.password,
            timeout=args.timeout,
        )

    cases = _build_cases()
    results = [_run_case(case, args.base_url, token, args.timeout, auth_disabled) for case in cases]
    _print_results(results)

    any_failures = any((item["status"] not in ("SKIP",) and not item["ok"]) for item in results)
    if args.fail_on_error and any_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
