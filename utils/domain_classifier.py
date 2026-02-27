"""
Shared quantum-domain classification helpers.
"""
import re

from scripts.quantum_regex import scan_text

_FALLBACK_TOKENS = (
    "qubit",
    "quantum",
    "qiskit",
    "pennylane",
    "cirq",
    "torchquantum",
    "openqasm",
    "hadamard",
    "cnot",
    "grover",
    "shor",
    "vqe",
    "qaoa",
)

_STRONG_QUANTUM_PATTERNS = (
    r"\bquantum\s+(?:computing|circuit|algorithm|gate|state|error|simulation|annealing)\b",
    r"\b(?:qiskit|pennylane|cirq|torchquantum|openqasm|qasm)\b",
    r"\b(?:vqe|qaoa|qft|qpe|grover|shor|deutsch[-_\s]?jozsa|simon(?:'s)?\s+algorithm)\b",
    r"\b(?:cnot|ccx|toffoli|fredkin|hadamard)\b",
    r"\bq\[\d+\]\b",
)

_WEAK_SCAN_CATEGORIES = {
    "Single Qubit Gates",
    "Measurements",
    "Classical Registers",
}

_SINGLE_GATE_WITH_CONTEXT = re.compile(
    r"\b(?:h|x|y|z|rx|ry|rz)\s*q\[\d+\]",
    re.IGNORECASE,
)


def is_quantum_domain_text(text: str) -> bool:
    """
    Return True when input appears to be in quantum-computing domain.
    """
    value = (text or "").strip()
    if not value:
        return False

    lowered = value.lower()
    if any(token in lowered for token in _FALLBACK_TOKENS):
        return True

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _STRONG_QUANTUM_PATTERNS):
        return True

    if _SINGLE_GATE_WITH_CONTEXT.search(value):
        return True

    try:
        hits = scan_text(value)
        for category, matches in hits.items():
            if category in _WEAK_SCAN_CATEGORIES:
                continue
            if matches:
                return True
    except Exception:
        # Fail-closed to False for domain gating on malformed regex payloads.
        return False

    return False
