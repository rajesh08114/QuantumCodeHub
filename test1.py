"""
Comprehensive endpoint test matrix for QuantumCodeHub backend.

Features:
- Fixes validation metadata extraction to:
  metadata.validation_evaluation.llm.passed
- Logs full metadata blob when the expected path is missing.
- Exercises multiple scenarios across:
  /api/code/generate, /api/transpile/convert, /api/complete/suggest,
  /api/explain/code, /api/fix/code, /api/chat/message
- Aggregates grounding metrics and hallucination suppression rate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


FRAMEWORKS = ["qiskit", "pennylane", "cirq", "torchquantum"]

RUNTIME_SCENARIOS = [
    {
        "name": "legacy",
        "runtime_preferences": {
            "mode": "legacy",
            "python_version": "3.9",
            "package_versions": {},
            "allow_deprecated_apis": True,
        },
    },
    {
        "name": "modern",
        "runtime_preferences": {
            "mode": "modern",
            "python_version": "3.11",
            "package_versions": {},
            "allow_deprecated_apis": False,
        },
    },
]

GENERATION_PROMPTS = [
    "Create a 2-qubit Bell state and measure both qubits.",
    "Create a 3-qubit GHZ state with execution and histogram output.",
    "Generate a parameterized variational ansatz circuit with measurement.",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def timed_request(
    session: requests.Session,
    method: str,
    url: str,
    timeout_seconds: float,
    **kwargs,
) -> Tuple[Optional[requests.Response], float, Optional[str]]:
    start = time.perf_counter()
    try:
        response = session.request(method, url, timeout=timeout_seconds, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return response, latency_ms, None
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, latency_ms, str(exc)


def parse_json(response: Optional[requests.Response]) -> Dict[str, Any]:
    if response is None:
        return {}
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def extract_llm_eval_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    llm_eval: Dict[str, Any] = {}
    validation_passed = None
    try:
        validation_passed = payload["metadata"]["validation_evaluation"]["llm"]["passed"]
        raw_llm_eval = payload["metadata"]["validation_evaluation"]["llm"]
        if isinstance(raw_llm_eval, dict):
            llm_eval = raw_llm_eval
    except Exception:
        validation_eval = (
            metadata.get("validation_evaluation", {})
            if isinstance(metadata.get("validation_evaluation"), dict)
            else {}
        )
        llm_eval = validation_eval.get("llm", {}) if isinstance(validation_eval.get("llm"), dict) else {}

    grounding_metrics = (
        llm_eval.get("grounding_metrics")
        if isinstance(llm_eval.get("grounding_metrics"), dict)
        else {}
    )
    metadata_blob = None
    if validation_passed is None:
        metadata_blob = json.dumps(metadata, ensure_ascii=True)[:2000]

    return {
        "validation_passed": validation_passed,
        "grounding_metrics": grounding_metrics,
        "metadata_blob_if_missing": metadata_blob,
    }


def detect_auth_disabled(base_url: str, timeout_seconds: float) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/auth/me", timeout=timeout_seconds)
        return response.status_code == 200
    except Exception:
        return False


def run_tests(base_url: str, token: Optional[str], timeout_seconds: float, auth_disabled: bool) -> Dict[str, Any]:
    session = requests.Session()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "base_url": base_url,
        "auth_disabled": auth_disabled,
        "tests": [],
        "summary": {},
    }

    all_latencies: List[float] = []
    endpoint_totals: Dict[str, int] = {}
    endpoint_passed: Dict[str, int] = {}

    generation_stats: List[Dict[str, Any]] = []

    def record(result: Dict[str, Any]) -> None:
        report["tests"].append(result)
        endpoint = result.get("endpoint", "unknown")
        endpoint_totals[endpoint] = endpoint_totals.get(endpoint, 0) + 1
        if result.get("status") == 200:
            endpoint_passed[endpoint] = endpoint_passed.get(endpoint, 0) + 1
        if isinstance(result.get("latency_ms"), (int, float)):
            all_latencies.append(float(result["latency_ms"]))

    if not auth_disabled and not token:
        record(
            {
                "endpoint": "suite",
                "scenario": "precheck",
                "status": "SKIP",
                "latency_ms": None,
                "error": "auth required but token not provided",
            }
        )
        report["summary"] = {"error": "auth required but token not provided"}
        return report

    # 1) Health checks
    for endpoint in ("/", "/health", "/api/transpile/supported-conversions"):
        response, latency_ms, error = timed_request(
            session=session,
            method="GET",
            url=f"{base_url.rstrip('/')}{endpoint}",
            timeout_seconds=timeout_seconds,
        )
        record(
            {
                "endpoint": endpoint,
                "scenario": "basic",
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
            }
        )

    # 2) Code generation matrix (framework x runtime x prompt)
    for framework in FRAMEWORKS:
        for runtime in RUNTIME_SCENARIOS:
            for prompt in GENERATION_PROMPTS:
                payload = {
                    "prompt": prompt,
                    "framework": framework,
                    "include_explanation": True,
                    "include_visualization": False,
                    "runtime_preferences": runtime["runtime_preferences"],
                    "client_context": {"client_type": "test_suite"},
                }
                response, latency_ms, error = timed_request(
                    session=session,
                    method="POST",
                    url=f"{base_url.rstrip('/')}/api/code/generate",
                    timeout_seconds=timeout_seconds,
                    json=payload,
                )
                body = parse_json(response)
                eval_meta = extract_llm_eval_metadata(body)
                metadata = body.get("metadata", {}) if isinstance(body.get("metadata"), dict) else {}
                grounding = eval_meta["grounding_metrics"] or {}

                result = {
                    "endpoint": "/api/code/generate",
                    "scenario": f"{framework}:{runtime['name']}",
                    "framework": framework,
                    "runtime_mode": runtime["name"],
                    "prompt": prompt,
                    "status": response.status_code if response is not None else "ERROR",
                    "latency_ms": round(latency_ms, 2),
                    "error": error,
                    "response_validation_passed": body.get("validation_passed"),
                    "llm_validation_passed": eval_meta["validation_passed"],
                    "auto_repair_used": metadata.get("auto_repair_used"),
                    "modernization_applied": metadata.get("modernization_applied"),
                    "adaptive_chain": metadata.get("adaptive_preferred_chain"),
                    "grounding_metrics": grounding,
                    "hallucination_suppression_rate": grounding.get("hallucination_suppression_rate"),
                }
                if eval_meta["metadata_blob_if_missing"] is not None:
                    result["metadata_blob_if_missing_llm_passed"] = eval_meta["metadata_blob_if_missing"]
                record(result)

                generation_stats.append(
                    {
                        "status": response.status_code if response is not None else None,
                        "confidence_score": body.get("confidence_score"),
                        "llm_validation_passed": eval_meta["validation_passed"],
                        "hallucination_suppression_rate": grounding.get("hallucination_suppression_rate"),
                        "raw_issue_count": grounding.get("raw_issue_count"),
                        "grounded_issue_count": grounding.get("grounded_issue_count"),
                        "dropped_issue_count": grounding.get("dropped_issue_count"),
                    }
                )

    # 3) Transpile scenarios
    transpile_scenarios = [
        {
            "name": "qiskit_to_pennylane",
            "payload": {
                "source_code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nqc.measure_all()",
                "source_framework": "qiskit",
                "target_framework": "pennylane",
                "preserve_comments": True,
                "optimize": False,
                "runtime_preferences": {"mode": "modern", "python_version": "3.11", "package_versions": {}},
                "client_context": {"client_type": "test_suite"},
            },
        },
        {
            "name": "cirq_to_qiskit",
            "payload": {
                "source_code": "import cirq\nq0,q1=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit(cirq.H(q0), cirq.CNOT(q0,q1), cirq.measure(q0,q1))",
                "source_framework": "cirq",
                "target_framework": "qiskit",
                "preserve_comments": True,
                "optimize": False,
                "runtime_preferences": {"mode": "modern", "python_version": "3.11", "package_versions": {}},
                "client_context": {"client_type": "test_suite"},
            },
        },
    ]
    for scenario in transpile_scenarios:
        response, latency_ms, error = timed_request(
            session=session,
            method="POST",
            url=f"{base_url.rstrip('/')}/api/transpile/convert",
            timeout_seconds=timeout_seconds,
            json=scenario["payload"],
        )
        body = parse_json(response)
        meta = body.get("metadata", {}) if isinstance(body.get("metadata"), dict) else {}
        record(
            {
                "endpoint": "/api/transpile/convert",
                "scenario": scenario["name"],
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "validation_passed": body.get("validation_passed"),
                "llm_validation_passed": (
                    meta.get("validation_evaluation", {}).get("llm", {}).get("passed")
                    if isinstance(meta.get("validation_evaluation"), dict)
                    else None
                ),
            }
        )

    # 4) Completion scenarios
    completion_scenarios = [
        {
            "name": "qiskit_prefix",
            "payload": {
                "code_prefix": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.",
                "framework": "qiskit",
                "cursor_line": 3,
                "cursor_column": 3,
                "max_suggestions": 5,
                "client_context": {"client_type": "test_suite"},
            },
        },
        {
            "name": "pennylane_prefix",
            "payload": {
                "code_prefix": "import pennylane as qml\ndev = qml.device('default.qubit', wires=2)\n@qml.qnode(dev)\ndef circuit(theta):\n    qml.",
                "framework": "pennylane",
                "cursor_line": 5,
                "cursor_column": 8,
                "max_suggestions": 5,
                "client_context": {"client_type": "test_suite"},
            },
        },
    ]
    for scenario in completion_scenarios:
        response, latency_ms, error = timed_request(
            session=session,
            method="POST",
            url=f"{base_url.rstrip('/')}/api/complete/suggest",
            timeout_seconds=timeout_seconds,
            json=scenario["payload"],
        )
        body = parse_json(response)
        suggestions = body.get("suggestions", []) if isinstance(body.get("suggestions"), list) else []
        record(
            {
                "endpoint": "/api/complete/suggest",
                "scenario": scenario["name"],
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "suggestion_count": len(suggestions),
            }
        )

    # 5) Explain scenarios
    explain_scenarios = [
        {
            "name": "bell_beginner",
            "payload": {
                "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nqc.measure_all()",
                "framework": "qiskit",
                "detail_level": "beginner",
                "include_math": False,
                "include_visualization": False,
                "client_context": {"client_type": "test_suite"},
            },
        },
        {
            "name": "ghz_advanced_math",
            "payload": {
                "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(3)\nqc.h(0)\nqc.cx(0,1)\nqc.cx(0,2)\nqc.measure_all()",
                "framework": "qiskit",
                "detail_level": "advanced",
                "include_math": True,
                "include_visualization": False,
                "client_context": {"client_type": "test_suite"},
            },
        },
    ]
    for scenario in explain_scenarios:
        response, latency_ms, error = timed_request(
            session=session,
            method="POST",
            url=f"{base_url.rstrip('/')}/api/explain/code",
            timeout_seconds=timeout_seconds,
            json=scenario["payload"],
        )
        body = parse_json(response)
        record(
            {
                "endpoint": "/api/explain/code",
                "scenario": scenario["name"],
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "has_overview": bool(body.get("overview")),
                "has_math": bool(body.get("mathematics")),
            }
        )

    # 6) Fix scenarios
    fix_scenarios = [
        {
            "name": "qiskit_missing_arg",
            "payload": {
                "code": "from qiskit import QuantumCircuit\nqc=QuantumCircuit(2)\nqc.h(0)\nqc.cx(0)",
                "framework": "qiskit",
                "error_message": "TypeError: cx() missing 1 required positional argument",
                "include_explanation": True,
                "client_context": {"client_type": "test_suite"},
            },
        },
        {
            "name": "cirq_bad_gate_usage",
            "payload": {
                "code": "import cirq\nq=cirq.LineQubit(0)\ncircuit=cirq.Circuit()\ncircuit.append(cirq.CNOT(q))",
                "framework": "cirq",
                "error_message": "TypeError: CNOT expects two qubits",
                "include_explanation": True,
                "client_context": {"client_type": "test_suite"},
            },
        },
    ]
    for scenario in fix_scenarios:
        response, latency_ms, error = timed_request(
            session=session,
            method="POST",
            url=f"{base_url.rstrip('/')}/api/fix/code",
            timeout_seconds=timeout_seconds,
            json=scenario["payload"],
        )
        body = parse_json(response)
        meta = body.get("metadata", {}) if isinstance(body.get("metadata"), dict) else {}
        record(
            {
                "endpoint": "/api/fix/code",
                "scenario": scenario["name"],
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "validation_passed": meta.get("validation_passed"),
                "llm_validation_passed": (
                    meta.get("validation_evaluation", {}).get("llm", {}).get("passed")
                    if isinstance(meta.get("validation_evaluation"), dict)
                    else None
                ),
            }
        )

    # 7) Chat scenarios (with session continuation)
    chat_scenarios = [
        {
            "name": "conceptual_intro",
            "payload": {
                "message": "What is quantum superposition? Explain simply.",
                "framework": "qiskit",
                "detail_level": "beginner",
                "new_session": True,
                "include_other_session_summaries": False,
                "client_context": {"client_type": "test_suite"},
            },
            "capture_session": True,
        },
        {
            "name": "code_followup",
            "payload": {
                "message": "Now give runnable qiskit code for a Bell state with measurement.",
                "framework": "qiskit",
                "detail_level": "intermediate",
                "new_session": False,
                "include_other_session_summaries": False,
                "client_context": {"client_type": "test_suite"},
            },
            "use_captured_session": True,
        },
        {
            "name": "math_focus",
            "payload": {
                "message": "Give matrix-level explanation of GHZ state and measurement probabilities.",
                "framework": "qiskit",
                "detail_level": "advanced",
                "new_session": False,
                "include_other_session_summaries": False,
                "client_context": {"client_type": "test_suite"},
            },
            "use_captured_session": True,
        },
        {
            "name": "legacy_version_chat",
            "payload": {
                "message": "Show qiskit code compatible with qiskit 0.45 style APIs.",
                "framework": "qiskit",
                "detail_level": "intermediate",
                "new_session": False,
                "include_other_session_summaries": False,
                "runtime_preferences": {
                    "mode": "legacy",
                    "python_version": "3.9",
                    "package_versions": {"qiskit": "0.45.*"},
                    "allow_deprecated_apis": True,
                },
                "client_context": {"client_type": "test_suite"},
            },
            "use_captured_session": True,
        },
    ]
    captured_session_id: Optional[str] = None
    for scenario in chat_scenarios:
        payload = dict(scenario["payload"])
        if scenario.get("use_captured_session") and captured_session_id:
            payload["session_id"] = captured_session_id
        response, latency_ms, error = timed_request(
            session=session,
            method="POST",
            url=f"{base_url.rstrip('/')}/api/chat/message",
            timeout_seconds=timeout_seconds,
            json=payload,
        )
        body = parse_json(response)
        returned_session_id = body.get("session_id")
        if scenario.get("capture_session") and isinstance(returned_session_id, str) and returned_session_id:
            captured_session_id = returned_session_id
        record(
            {
                "endpoint": "/api/chat/message",
                "scenario": scenario["name"],
                "status": response.status_code if response is not None else "ERROR",
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "intent": body.get("intent"),
                "has_reply": bool(body.get("reply")),
                "has_code_block": bool(body.get("code")),
                "session_id": returned_session_id,
            }
        )

    # Summary
    executed = [t for t in report["tests"] if t.get("status") not in ("SKIP",)]
    passed = [t for t in executed if t.get("status") == 200]
    failed = [t for t in executed if t.get("status") != 200]

    generation_success = [g for g in generation_stats if g.get("status") == 200]
    llm_pass_observed = [g["llm_validation_passed"] for g in generation_success if isinstance(g.get("llm_validation_passed"), bool)]
    hallucination_rates = [
        float(g["hallucination_suppression_rate"])
        for g in generation_success
        if isinstance(g.get("hallucination_suppression_rate"), (int, float))
    ]
    raw_issue_counts = [
        int(g["raw_issue_count"])
        for g in generation_success
        if isinstance(g.get("raw_issue_count"), int)
    ]
    grounded_issue_counts = [
        int(g["grounded_issue_count"])
        for g in generation_success
        if isinstance(g.get("grounded_issue_count"), int)
    ]
    dropped_issue_counts = [
        int(g["dropped_issue_count"])
        for g in generation_success
        if isinstance(g.get("dropped_issue_count"), int)
    ]
    confidence_scores = [
        float(g["confidence_score"])
        for g in generation_success
        if isinstance(g.get("confidence_score"), (int, float))
    ]

    endpoint_summary: Dict[str, Dict[str, Any]] = {}
    for endpoint, total in endpoint_totals.items():
        ok = endpoint_passed.get(endpoint, 0)
        endpoint_summary[endpoint] = {
            "total": total,
            "passed": ok,
            "success_rate": round(float(ok) / float(total), 4) if total else 0.0,
        }

    report["summary"] = {
        "total_requests": len(executed),
        "passed_requests": len(passed),
        "failed_requests": len(failed),
        "success_rate": round(float(len(passed)) / float(len(executed)), 4) if executed else 0.0,
        "avg_latency_ms": round(statistics.mean(all_latencies), 2) if all_latencies else 0.0,
        "p95_latency_ms": round(percentile(all_latencies, 0.95), 2) if all_latencies else 0.0,
        "endpoint_summary": endpoint_summary,
        "generation_metrics": {
            "total_generation_requests": len(generation_stats),
            "successful_generation_requests": len(generation_success),
            "avg_confidence_score": round(statistics.mean(confidence_scores), 4) if confidence_scores else None,
            "llm_validation_pass_rate": (
                round(sum(1 for v in llm_pass_observed if v) / len(llm_pass_observed), 4)
                if llm_pass_observed
                else None
            ),
            "avg_hallucination_suppression_rate": (
                round(statistics.mean(hallucination_rates), 4) if hallucination_rates else None
            ),
            "avg_raw_issue_count": round(statistics.mean(raw_issue_counts), 4) if raw_issue_counts else None,
            "avg_grounded_issue_count": (
                round(statistics.mean(grounded_issue_counts), 4) if grounded_issue_counts else None
            ),
            "avg_dropped_issue_count": (
                round(statistics.mean(dropped_issue_counts), 4) if dropped_issue_counts else None
            ),
        },
        "failures": [
            {
                "endpoint": item.get("endpoint"),
                "scenario": item.get("scenario"),
                "status": item.get("status"),
                "error": item.get("error"),
            }
            for item in failed
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantumCodeHub comprehensive endpoint test matrix")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="quantumcodehub_test_results.json")
    parser.add_argument(
        "--auth-disabled",
        action="store_true",
        help="Force running without auth token checks",
    )
    args = parser.parse_args()

    detected_auth_disabled = detect_auth_disabled(args.base_url, args.timeout)
    auth_disabled = bool(args.auth_disabled or detected_auth_disabled)

    report = run_tests(
        base_url=args.base_url,
        token=args.token.strip() or None,
        timeout_seconds=args.timeout,
        auth_disabled=auth_disabled,
    )

    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    print(json.dumps(report.get("summary", {}), indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
