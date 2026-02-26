"""
Qiskit-only endpoint tester for QuantumCodeHub.

Purpose:
- hit all Qiskit-related endpoints
- detect whether an endpoint returned code
- validate returned code via subprocess compile (and optional execution)
- store full request/response + code-check results in JSON
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests


FRAMEWORK = "qiskit"
PROFILES = ["simulator", "hardware"]
DEFAULT_VARIATIONS = ["bell", "ghz", "qft2"]
QISKIT_VERSION = "2.3.0"


@dataclass
class EndpointCase:
    name: str
    method: str
    path: str
    requires_auth: bool = False
    json_body: Optional[Dict[str, Any]] = None
    form_body: Optional[Dict[str, Any]] = None
    expected_statuses: Optional[List[int]] = None
    profile: str = "default"
    variation: str = "default"
    expects_executable_code: bool = False


def _runtime_preferences() -> Dict[str, Any]:
    return {
        "mode": "modern",
        "python_version": "3.11",
        "package_versions": {"qiskit": QISKIT_VERSION},
        "allow_deprecated_apis": False,
    }


def _qiskit_prompt(variation: str, profile: str) -> str:
    mode = "simulator-compatible" if profile == "simulator" else "hardware-compatible"
    key = (variation or "").strip().lower()
    if key == "ghz":
        return f"Create runnable qiskit GHZ-state code for 3 qubits with measurement ({mode})."
    if key == "qft2":
        return f"Create runnable qiskit 2-qubit QFT code with final measurement ({mode})."
    return f"Create runnable qiskit Bell-state code with measurement ({mode})."


def _qiskit_chat_query(variation: str, profile: str) -> str:
    mode = "simulator-compatible" if profile == "simulator" else "hardware-compatible"
    key = (variation or "").strip().lower()
    if key == "ghz":
        return f"Return runnable qiskit GHZ-state code for 3 qubits with measurement ({mode})."
    if key == "qft2":
        return f"Return runnable qiskit 2-qubit QFT code with final measurement ({mode})."
    return f"Return runnable qiskit Bell-state code with measurement ({mode})."


def _qiskit_completion_prefix(variation: str) -> str:
    key = (variation or "").strip().lower()
    if key == "ghz":
        return (
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(3)\n"
            "qc.h(0)\n"
            "qc.cx(0, 1)\n"
            "qc."
        )
    if key == "qft2":
        return (
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(2)\n"
            "qc.h(0)\n"
            "qc.cp(1.5708, 1, 0)\n"
            "qc."
        )
    return "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc."


def _qiskit_explain_code(variation: str) -> str:
    key = (variation or "").strip().lower()
    if key == "ghz":
        return (
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(3)\n"
            "qc.h(0)\n"
            "qc.cx(0,1)\n"
            "qc.cx(1,2)\n"
            "qc.measure_all()"
        )
    if key == "qft2":
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
        "qc=QuantumCircuit(2)\n"
        "qc.h(0)\n"
        "qc.cx(0,1)\n"
        "qc.measure_all()"
    )


def _qiskit_broken_code(variation: str) -> tuple[str, str]:
    key = (variation or "").strip().lower()
    if key == "ghz":
        return (
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(3)\nqc.h(0)\nqc.cx(0,1)\nqc.ccx(0,1)",
            "TypeError: ccx() missing 1 required positional argument",
        )
    if key == "qft2":
        return (
            "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cp(1.5708, 0)\nqc.measure_all()",
            "TypeError: cp() missing 1 required positional argument",
        )
    return (
        "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0)",
        "TypeError: cx() missing 1 required positional argument",
    )


def _transpile_source(framework: str) -> str:
    key = (framework or "").strip().lower()
    if key == "pennylane":
        return (
            "import pennylane as qml\n"
            "dev = qml.device('default.qubit', wires=2)\n"
            "@qml.qnode(dev)\n"
            "def circuit():\n"
            "    qml.Hadamard(wires=0)\n"
            "    qml.CNOT(wires=[0,1])\n"
            "    return qml.probs(wires=[0,1])"
        )
    return (
        "import cirq\n"
        "q0, q1 = cirq.LineQubit.range(2)\n"
        "circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))"
    )


def _base_cases() -> List[EndpointCase]:
    return [
        EndpointCase(name="root", method="GET", path="/", expected_statuses=[200]),
        EndpointCase(name="health", method="GET", path="/health", expected_statuses=[200]),
        EndpointCase(name="supported-conversions", method="GET", path="/api/transpile/supported-conversions", expected_statuses=[200]),
        EndpointCase(name="me", method="GET", path="/api/auth/me", requires_auth=True, expected_statuses=[200]),
    ]


def _qiskit_cases(variations: List[str]) -> List[EndpointCase]:
    cases: List[EndpointCase] = []
    prefs = _runtime_preferences()

    for profile in PROFILES:
        for variation in variations:
            broken, err = _qiskit_broken_code(variation)
            tag = f"{variation}:{profile}"

            cases.append(
                EndpointCase(
                    name=f"code-generate-{tag}",
                    method="POST",
                    path="/api/code/generate",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=variation,
                    expects_executable_code=True,
                    json_body={
                        "prompt": _qiskit_prompt(variation, profile),
                        "framework": FRAMEWORK,
                        "include_explanation": False,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )

            cases.append(
                EndpointCase(
                    name=f"completion-{tag}",
                    method="POST",
                    path="/api/complete/suggest",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=variation,
                    expects_executable_code=False,
                    json_body={
                        "code_prefix": _qiskit_completion_prefix(variation),
                        "framework": FRAMEWORK,
                        "cursor_line": 3,
                        "cursor_column": 3,
                        "max_suggestions": 4,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )

            cases.append(
                EndpointCase(
                    name=f"explain-{tag}",
                    method="POST",
                    path="/api/explain/code",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=variation,
                    expects_executable_code=False,
                    json_body={
                        "code": _qiskit_explain_code(variation),
                        "framework": FRAMEWORK,
                        "detail_level": "intermediate",
                        "include_math": False,
                        "include_visualization": False,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )

            cases.append(
                EndpointCase(
                    name=f"fix-{tag}",
                    method="POST",
                    path="/api/fix/code",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=variation,
                    expects_executable_code=True,
                    json_body={
                        "code": broken,
                        "framework": FRAMEWORK,
                        "error_message": err,
                        "include_explanation": True,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )

            cases.append(
                EndpointCase(
                    name=f"chat-{tag}",
                    method="POST",
                    path="/api/chat/message",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=variation,
                    expects_executable_code=True,
                    json_body={
                        "message": _qiskit_chat_query(variation, profile),
                        "framework": FRAMEWORK,
                        "detail_level": "intermediate",
                        "new_session": True,
                        "include_other_session_summaries": False,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )

    for profile in PROFILES:
        for src in ["pennylane", "cirq"]:
            cases.append(
                EndpointCase(
                    name=f"transpile-{src}-to-qiskit-{profile}",
                    method="POST",
                    path="/api/transpile/convert",
                    requires_auth=True,
                    expected_statuses=[200],
                    profile=profile,
                    variation=f"{src}->qiskit",
                    expects_executable_code=True,
                    json_body={
                        "source_code": _transpile_source(src),
                        "source_framework": src,
                        "target_framework": FRAMEWORK,
                        "preserve_comments": True,
                        "optimize": False,
                        "runtime_preferences": prefs,
                        "client_context": {"client_type": "api"},
                    },
                )
            )
    return cases


def _safe_response_body(response: requests.Response) -> Any:
    content_type = str(response.headers.get("content-type", "")).lower()
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            return {"raw_text": response.text[:20000]}
    if "text/" in content_type:
        return {"raw_text": response.text[:20000]}
    return {"raw_bytes_len": len(response.content), "content_type": content_type}


def _extract_python_code(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return raw


def _extract_code_candidate(case: EndpointCase, response_body: Any) -> str:
    if case.method != "POST":
        return ""
    if not isinstance(response_body, dict):
        return ""

    path = (case.path or "").strip().lower()
    if path == "/api/code/generate":
        keys = ["code"]
    elif path == "/api/fix/code":
        keys = ["fixed_code", "code"]
    elif path == "/api/transpile/convert":
        keys = ["transpiled_code", "code"]
    elif path == "/api/chat/message":
        keys = ["code", "reply"]
    elif path == "/api/complete/suggest":
        suggestions = response_body.get("suggestions")
        if isinstance(suggestions, list) and suggestions:
            first = suggestions[0]
            if isinstance(first, dict):
                value = first.get("code")
                if isinstance(value, str) and value.strip():
                    return _extract_python_code(value)
        return ""
    else:
        return ""

    for key in keys:
        value = response_body.get(key)
        if isinstance(value, str) and value.strip():
            return _extract_python_code(value)
    return ""


def _validate_code_subprocess(
    code: str,
    execute_generated_code: bool = False,
    subprocess_timeout_seconds: float = 12.0,
) -> Dict[str, Any]:
    candidate = (code or "").strip()
    if not candidate:
        return {
            "checked": False,
            "compile_ok": None,
            "execute_ok": None,
            "error": "empty_code",
            "code_chars": 0,
        }

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(candidate)
            temp_path = handle.name

        compile_proc = subprocess.run(
            [sys.executable, "-m", "py_compile", temp_path],
            capture_output=True,
            text=True,
            timeout=subprocess_timeout_seconds,
            check=False,
        )
        if compile_proc.returncode != 0:
            return {
                "checked": True,
                "compile_ok": False,
                "execute_ok": None,
                "error": (compile_proc.stderr or compile_proc.stdout or "py_compile failed").strip()[:1200],
                "code_chars": len(candidate),
            }

        if not execute_generated_code:
            return {
                "checked": True,
                "compile_ok": True,
                "execute_ok": None,
                "error": None,
                "code_chars": len(candidate),
            }

        run_proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=subprocess_timeout_seconds,
            check=False,
        )
        execute_ok = run_proc.returncode == 0
        return {
            "checked": True,
            "compile_ok": True,
            "execute_ok": execute_ok,
            "error": None if execute_ok else (run_proc.stderr or run_proc.stdout or "execution failed").strip()[:1200],
            "code_chars": len(candidate),
        }
    except subprocess.TimeoutExpired:
        return {
            "checked": True,
            "compile_ok": False,
            "execute_ok": False if execute_generated_code else None,
            "error": f"subprocess timeout after {subprocess_timeout_seconds}s",
            "code_chars": len(candidate),
        }
    except Exception as exc:
        return {
            "checked": True,
            "compile_ok": False,
            "execute_ok": False if execute_generated_code else None,
            "error": str(exc),
            "code_chars": len(candidate),
        }
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


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
    validate_generated_code: bool,
    execute_generated_code: bool,
    subprocess_timeout_seconds: float,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if case.form_body:
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    url = f"{base_url.rstrip('/')}{case.path}"
    request_payload = {
        "method": case.method,
        "url": url,
        "path": case.path,
        "headers": headers,
        "json_body": case.json_body,
        "form_body": case.form_body,
    }
    base = {
        "name": case.name,
        "framework": FRAMEWORK,
        "profile": case.profile,
        "variation": case.variation,
        "expects_executable_code": case.expects_executable_code,
        "request": request_payload,
    }

    if case.requires_auth and not token and not auth_disabled:
        return {
            **base,
            "status": "SKIP",
            "ok": False,
            "time_ms": None,
            "error": "missing auth token",
            "response": None,
            "extracted_code": "",
            "code_returned": False,
            "code_validation": {"checked": False, "compile_ok": None, "execute_ok": None, "error": None, "code_chars": 0},
        }

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
        ok = response.status_code in (case.expected_statuses or [200])
        error = None
        body = _safe_response_body(response)

        extracted_code = _extract_code_candidate(case, body)
        code_returned = bool(extracted_code.strip())
        code_validation = {
            "checked": False,
            "compile_ok": None,
            "execute_ok": None,
            "error": None,
            "code_chars": len(extracted_code),
        }

        if validate_generated_code and code_returned:
            code_validation = _validate_code_subprocess(
                extracted_code,
                execute_generated_code=execute_generated_code,
                subprocess_timeout_seconds=subprocess_timeout_seconds,
            )

        if case.expects_executable_code:
            if not code_returned:
                ok = False
                error = "expected executable code, but no code was returned"
            elif validate_generated_code:
                compile_ok = bool(code_validation.get("compile_ok"))
                execute_ok = code_validation.get("execute_ok")
                if (not compile_ok) or (execute_generated_code and execute_ok is False):
                    ok = False
                    error = f"generated code failed subprocess checks: {code_validation.get('error') or 'unknown'}"

        if not ok and not error:
            error = str(response.text or "")[:500].replace("\n", " ")

        return {
            **base,
            "status": int(response.status_code),
            "ok": ok,
            "time_ms": float(elapsed_ms),
            "error": error,
            "response": {
                "status_code": int(response.status_code),
                "headers": dict(response.headers),
                "body": body,
                "elapsed_ms": float(elapsed_ms),
            },
            "extracted_code": extracted_code,
            "code_returned": code_returned,
            "code_validation": code_validation,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            **base,
            "status": "ERR",
            "ok": False,
            "time_ms": float(elapsed_ms),
            "error": str(exc),
            "response": None,
            "extracted_code": "",
            "code_returned": False,
            "code_validation": {"checked": False, "compile_ok": None, "execute_ok": None, "error": None, "code_chars": 0},
        }


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    executed = [r for r in results if r["status"] not in {"SKIP"}]
    passed = [r for r in executed if r["ok"]]
    failed = [r for r in executed if not r["ok"]]
    skipped = [r for r in results if r["status"] == "SKIP"]
    latencies = [float(r["time_ms"]) for r in executed if isinstance(r.get("time_ms"), (int, float))]

    expected_code_cases = [r for r in executed if r.get("expects_executable_code")]
    code_returned_count = len([r for r in executed if r.get("code_returned")])
    checks = [
        r.get("code_validation", {})
        for r in executed
        if isinstance(r.get("code_validation"), dict) and r.get("code_validation", {}).get("checked")
    ]
    compile_failures = [c for c in checks if c.get("compile_ok") is False]
    execution_failures = [c for c in checks if c.get("execute_ok") is False]

    avg_ms = statistics.mean(latencies) if latencies else 0.0
    p95_ms = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)

    return {
        "executed": len(executed),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "avg_latency_ms": float(avg_ms),
        "p95_latency_ms": float(p95_ms),
        "result": "success" if not failed else "partial_failure",
        "code_signals": {
            "expected_executable_cases": len(expected_code_cases),
            "code_returned_cases": code_returned_count,
            "checked_cases": len(checks),
            "compile_failures": len(compile_failures),
            "execution_failures": len(execution_failures),
        },
    }


def _print_results(results: List[Dict[str, Any]]) -> None:
    print("\nQiskit Endpoint Test Results")
    print("-" * 142)
    print(f"{'NAME':36} {'STATUS':8} {'TIME_MS':10} {'PROFILE':10} {'CODE':6} {'EXEC':6} PATH")
    print("-" * 142)
    for item in results:
        path = str(item.get("request", {}).get("path", ""))
        profile = str(item.get("profile", "default"))
        time_ms = "-" if item["time_ms"] is None else f"{item['time_ms']:.1f}"
        code_flag = "yes" if item.get("code_returned") else "no"
        exec_ok = item.get("code_validation", {}).get("compile_ok")
        exec_flag = "-"
        if exec_ok is True:
            exec_flag = "ok"
        elif exec_ok is False:
            exec_flag = "fail"
        print(f"{item['name'][:36]:36} {str(item['status']):8} {time_ms:10} {profile:10} {code_flag:6} {exec_flag:6} {path}")
    print("-" * 142)

    summary = _build_summary(results)
    print(f"Executed: {summary['executed']}  Passed: {summary['passed']}  Failed: {summary['failed']}  Skipped: {summary['skipped']}")
    print(f"Avg latency: {summary['avg_latency_ms']:.1f} ms  P95 latency: {summary['p95_latency_ms']:.1f} ms")
    print(
        "Code signals: "
        f"{summary['code_signals']['expected_executable_cases']} expected executable, "
        f"{summary['code_signals']['code_returned_cases']} returned code, "
        f"{summary['code_signals']['checked_cases']} checked, "
        f"{summary['code_signals']['compile_failures']} compile failures, "
        f"{summary['code_signals']['execution_failures']} execution failures"
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parse_variations(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_VARIATIONS)
    values = []
    for item in raw.split(","):
        token = item.strip().lower()
        if token and token not in values:
            values.append(token)
    return values or list(DEFAULT_VARIATIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qiskit-only endpoint tester with executable-code checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--token", default="", help="JWT token for protected endpoints")
    parser.add_argument("--email", default="", help="Login email (optional)")
    parser.add_argument("--password", default="", help="Login password (optional)")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout seconds")
    parser.add_argument("--auth-disabled", action="store_true", help="Run without auth token checks")
    parser.add_argument("--variations", default="bell,ghz,qft2", help="Comma-separated qiskit variations to test")
    parser.add_argument("--no-base-endpoints", action="store_true", help="Skip base endpoints (/ /health /api/auth/me /supported-conversions)")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero exit code if any endpoint fails")
    parser.add_argument("--skip-code-validation", action="store_true", help="Skip subprocess code validation")
    parser.add_argument("--execute-generated-code", action="store_true", help="Execute code after compile validation")
    parser.add_argument("--code-subprocess-timeout", type=float, default=12.0, help="Timeout seconds for compile/execute checks")
    parser.add_argument("--output-json", default="quantumcodehub_test_results.json", help="JSON report path")
    args = parser.parse_args()

    auth_disabled = bool(args.auth_disabled)
    if not auth_disabled:
        auth_disabled = _detect_auth_disabled(args.base_url, args.timeout)
    print(f"[auth] detected auth_disabled={auth_disabled}")

    token = args.token.strip() or None
    if not token and not auth_disabled and args.email and args.password:
        token = _try_login(args.base_url, args.email, args.password, args.timeout)

    variations = _parse_variations(args.variations)
    cases = []
    if not bool(args.no_base_endpoints):
        cases.extend(_base_cases())
    cases.extend(_qiskit_cases(variations))

    validate_code = not bool(args.skip_code_validation)
    started = time.time()
    results = [
        _run_case(
            case,
            args.base_url,
            token,
            args.timeout,
            auth_disabled,
            validate_generated_code=validate_code,
            execute_generated_code=bool(args.execute_generated_code),
            subprocess_timeout_seconds=float(args.code_subprocess_timeout),
        )
        for case in cases
    ]

    _print_results(results)
    summary = _build_summary(results)

    report = {
        "generated_at": started,
        "base_url": args.base_url.rstrip("/"),
        "scope": {
            "framework": FRAMEWORK,
            "profiles": PROFILES,
            "variations": variations,
            "include_base_endpoints": not bool(args.no_base_endpoints),
        },
        "auth": {
            "auth_disabled": auth_disabled,
            "token_used": bool(token),
            "email_login_attempted": bool(args.email and args.password),
        },
        "code_validation": {
            "enabled": validate_code,
            "execute_generated_code": bool(args.execute_generated_code),
            "subprocess_timeout_seconds": float(args.code_subprocess_timeout),
        },
        "summary": summary,
        "requests": results,
    }

    output_path = Path(args.output_json).resolve()
    _write_json(output_path, report)
    print(f"JSON report: {output_path}")

    if args.fail_on_error and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
