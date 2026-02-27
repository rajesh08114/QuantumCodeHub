"""
Shared quantum-domain classification helpers.
"""
from scripts.quantum_regex import is_quantum_text

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


def is_quantum_domain_text(text: str) -> bool:
    """
    Return True when input appears to be in quantum-computing domain.
    """
    value = (text or "").strip()
    if not value:
        return False

    try:
        if is_quantum_text(value):
            return True
    except Exception:
        # Fall through to lexical backup checks.
        pass

    lowered = value.lower()
    return any(token in lowered for token in _FALLBACK_TOKENS)
