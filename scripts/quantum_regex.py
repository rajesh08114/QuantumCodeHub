"""
╔══════════════════════════════════════════════════════════════════════╗
║          QUANTUM COMPUTING - COMPREHENSIVE REGEX PATTERN FILE        ║
║         Covers: Gates, Circuits, Algorithms, States, Notations       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────
# 1. QUANTUM GATES
# ─────────────────────────────────────────────────────────────

# Single Qubit Gates: H, X, Y, Z, T, S, I, Rx, Ry, Rz
SINGLE_QUBIT_GATE = re.compile(
    r'\b(H|X|Y|Z|T|S|I|Rx|Ry|Rz|U1|U2|U3|P|SX|SX†|T†|S†)\b'
)

# Multi Qubit Gates: CNOT, CX, CZ, SWAP, Toffoli, Fredkin, CCX, CY
MULTI_QUBIT_GATE = re.compile(
    r'\b(CNOT|CX|CY|CZ|CH|CP|SWAP|iSWAP|CSWAP|CCX|CCZ|Toffoli|Fredkin|'
    r'RCCX|RC3X|C3X|C4X)\b',
    re.IGNORECASE
)

# All quantum gates combined
ALL_GATES = re.compile(
    r'\b(H|X|Y|Z|T|S|I|Rx|Ry|Rz|U1|U2|U3|P|SX|'
    r'CNOT|CX|CY|CZ|CH|CP|SWAP|iSWAP|CSWAP|CCX|CCZ|'
    r'Toffoli|Fredkin|RCCX|RC3X|C3X|C4X)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 2. QUANTUM CIRCUIT NOTATION
# ─────────────────────────────────────────────────────────────

# Circuit wire notation e.g., q[0], q[1], qubit[2]
QUBIT_REGISTER = re.compile(
    r'\b(q|qubit|qreg|anc|ancilla)\s*\[\s*\d+\s*\]',
    re.IGNORECASE
)

# Classical register notation e.g., c[0], creg[1]
CLASSICAL_REGISTER = re.compile(
    r'\b(c|creg|cbits?|meas)\s*\[\s*\d+\s*\]',
    re.IGNORECASE
)

# QASM-style gate application e.g., h q[0]; cx q[0], q[1];
QASM_GATE_INSTRUCTION = re.compile(
    r'\b(h|x|y|z|t|s|cx|cz|ccx|swap|rx|ry|rz|u1|u2|u3|measure|reset|barrier)\s+'
    r'(q|qubit)\s*\[\s*\d+\s*\](\s*,\s*(q|qubit)\s*\[\s*\d+\s*\])*\s*;',
    re.IGNORECASE
)

# Measurement instruction
MEASUREMENT = re.compile(
    r'\b(measure|meas)\s+\w+\s*(->|→)\s*\w+\b',
    re.IGNORECASE
)

# Barrier instruction
BARRIER = re.compile(r'\bbarrier\s+[\w\s,\[\]]+;', re.IGNORECASE)

# Circuit depth/width notation
CIRCUIT_PARAMS = re.compile(
    r'\b(depth|width|size|num_qubits?|num_clbits?)\s*[=:]\s*\d+',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 3. QUANTUM STATE NOTATIONS (DIRAC / BRA-KET)
# ─────────────────────────────────────────────────────────────

# Ket notation: |0>, |1>, |+>, |->, |ψ>, |Φ+>, etc.
KET_NOTATION = re.compile(
    r'\|[01+\-ψφΨΦβΒ↑↓→←⊕⊗01][0-9±+\-*/]*[\+\-]?\s*(?:⟩|>)',
    re.UNICODE
)

# Bra notation: <0|, <1|, <ψ|
BRA_NOTATION = re.compile(
    r'(?:⟨|<)\s*[01+\-ψφΨΦβΒ↑↓→←⊕⊗01][0-9±+\-]?\s*\|',
    re.UNICODE
)

# Bell States: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
BELL_STATES = re.compile(
    r'\|(?:Φ[+\-]|Ψ[+\-]|phi[_\-]?(?:plus|minus)|psi[_\-]?(?:plus|minus))\s*(?:⟩|>)',
    re.IGNORECASE | re.UNICODE
)

# GHZ / W states
GHZ_W_STATES = re.compile(
    r'\b(GHZ|W[_\-]?state|W\s*state)\b',
    re.IGNORECASE
)

# Superposition notation
SUPERPOSITION = re.compile(
    r'(?:\d+/\d+|\d*\.\d+)?\s*\|[01]\s*(?:⟩|>)\s*[+\-]\s*(?:\d+/\d+|\d*\.\d+)?\s*\|[01]\s*(?:⟩|>)'
)

# Density matrix / state vector
DENSITY_MATRIX = re.compile(
    r'\b(density[_\s]?matrix|rho|ρ|state[_\s]?vector|bloch[_\s]?vector)\b',
    re.IGNORECASE | re.UNICODE
)

# ─────────────────────────────────────────────────────────────
# 4. QUANTUM ALGORITHMS
# ─────────────────────────────────────────────────────────────

QUANTUM_ALGORITHMS = re.compile(
    r'\b('
    r'Shor[\'s]*\s*[Aa]lgorithm|'
    r'Grover[\'s]*\s*[Aa]lgorithm|'
    r'Deutsch[_\-]Jozsa|'
    r'Simon[\'s]*\s*[Aa]lgorithm|'
    r'Bernstein[_\-]Vazirani|'
    r'QFT|Quantum\s*Fourier\s*Transform|'
    r'QPE|Quantum\s*Phase\s*Estimation|'
    r'HHL\s*[Aa]lgorithm|'
    r'VQE|Variational\s*Quantum\s*Eigensolver|'
    r'QAOA|Quantum\s*Approximate\s*Optimization\s*Algorithm|'
    r'QSVM|Quantum\s*Support\s*Vector\s*Machine|'
    r'QNN|Quantum\s*Neural\s*Network|'
    r'QPCA|Quantum\s*Principal\s*Component\s*Analysis|'
    r'Quantum\s*Walk|'
    r'Amplitude\s*[Ee]stimation|'
    r'Amplitude\s*[Aa]mplification|'
    r'Quantum\s*[Aa]nnealing|'
    r'SWAP\s*[Tt]est'
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 5. QUANTUM ERROR CORRECTION CODES
# ─────────────────────────────────────────────────────────────

QUANTUM_ERROR_CORRECTION = re.compile(
    r'\b('
    r'Shor\s*[Cc]ode|'
    r'Steane\s*[Cc]ode|'
    r'Surface\s*[Cc]ode|'
    r'Toric\s*[Cc]ode|'
    r'Repetition\s*[Cc]ode|'
    r'CSS\s*[Cc]ode|'
    r'Calderbank[_\-]Shor[_\-]Steane|'
    r'Stabilizer\s*[Cc]ode|'
    r'Color\s*[Cc]ode|'
    r'Bosonic\s*[Cc]ode|'
    r'Cat\s*[Cc]ode|'
    r'Bacon[_\-]Shor\s*[Cc]ode|'
    r'Reed[_\-]Muller\s*[Cc]ode|'
    r'Topological\s*[Cc]ode|'
    r'Quantum\s*LDPC\s*[Cc]ode|'
    r'\[\[(\d+),\s*(\d+),\s*(\d+)\]\]'   # [[n,k,d]] notation
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 6. QUANTUM HARDWARE & TECHNOLOGY
# ─────────────────────────────────────────────────────────────

QUANTUM_HARDWARE = re.compile(
    r'\b('
    r'superconducting\s*qubit|transmon|flux\s*qubit|charge\s*qubit|'
    r'trapped\s*ion|ion\s*trap|'
    r'photonic\s*qubit|optical\s*qubit|'
    r'topological\s*qubit|Majorana\s*qubit|'
    r'spin\s*qubit|nitrogen[_\-]vacancy|NV[_\-]center|'
    r'quantum\s*dot|'
    r'neutral\s*atom|Rydberg\s*atom|'
    r'quantum\s*processor|QPU|'
    r'cryostat|dilution\s*refrigerator|'
    r'qubit\s*coherence|T1|T2|T2\*|'
    r'gate\s*fidelity|readout\s*fidelity'
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 7. QUANTUM PROGRAMMING & FRAMEWORKS
# ─────────────────────────────────────────────────────────────

QUANTUM_FRAMEWORKS = re.compile(
    r'\b('
    r'Qiskit|Cirq|Q#|QDK|Quil|PyQuil|'
    r'PennyLane|Braket|OpenQASM|QASM|'
    r'ProjectQ|Strawberry\s*Fields|'
    r'Quirk|tket|pytket|'
    r'QuTiP|Qibo|'
    r'QuantumCircuit|QuantumRegister|ClassicalRegister'
    r')\b',
    re.IGNORECASE
)

# OpenQASM version header
OPENQASM_HEADER = re.compile(
    r'OPENQASM\s+\d+(\.\d+)?\s*;',
    re.IGNORECASE
)

# Include statement (Qiskit/QASM)
QASM_INCLUDE = re.compile(
    r'include\s+"[\w.]+"\s*;',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 8. QUANTUM CRYPTOGRAPHY & COMMUNICATION
# ─────────────────────────────────────────────────────────────

QUANTUM_CRYPTO = re.compile(
    r'\b('
    r'QKD|Quantum\s*Key\s*Distribution|'
    r'BB84|B92|E91|BBM92|'
    r'Quantum\s*Teleportation|'
    r'Quantum\s*Entanglement|entangled\s*pair|EPR\s*pair|'
    r'Quantum\s*Repeater|'
    r'Quantum\s*Memory|'
    r'Quantum\s*Network|'
    r'Quantum\s*Internet|'
    r'Quantum\s*Channel|'
    r'No[_\-]Cloning\s*Theorem|'
    r'Quantum\s*Supremacy|Quantum\s*Advantage|'
    r'Post[_\-]Quantum\s*Cryptography|PQC|'
    r'Lattice[_\-]Based\s*Cryptography'
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 9. QUANTUM PHYSICS TERMS
# ─────────────────────────────────────────────────────────────

QUANTUM_PHYSICS_TERMS = re.compile(
    r'\b('
    r'superposition|entanglement|decoherence|interference|'
    r'wave\s*function|wavefunction|collapse|measurement|'
    r'Hamiltonian|Hilbert\s*Space|eigenvalue|eigenvector|eigenstate|'
    r'unitary|Hermitian|observable|operator|'
    r'tensor\s*product|Kronecker\s*product|'
    r'Pauli\s*(?:matrix|matrices|X|Y|Z)|'
    r'Hadamard\s*(?:gate|transform|matrix)|'
    r'rotation\s*(?:gate|operator)|'
    r'phase\s*(?:gate|kickback|shift)|'
    r'quantum\s*noise|depolarizing\s*noise|'
    r'bit\s*flip|phase\s*flip|'
    r'quantum\s*channel|Kraus\s*operator|'
    r'fidelity|purity|trace|partial\s*trace|'
    r'von\s*Neumann\s*entropy|entanglement\s*entropy'
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# 10. QUANTUM COMPUTING METRICS & NOTATIONS
# ─────────────────────────────────────────────────────────────

# Qubit counts: n-qubit, 50-qubit, etc.
QUBIT_COUNT = re.compile(r'\b(\d+)[_\-\s]?qubit(?:s)?\b', re.IGNORECASE)

# Big-O quantum complexity
QUANTUM_COMPLEXITY = re.compile(
    r'O\(\s*(?:N|n|2\^n|sqrt\(n\)|log\(n\)|n\^2|poly\(n\))\s*\)',
    re.IGNORECASE
)

# Matrix/vector size for quantum (2^n)
QUANTUM_MATRIX_SIZE = re.compile(r'\b2\s*\^\s*\d+\b')

# ─────────────────────────────────────────────────────────────
# 11. QASM CODE BLOCK
# ─────────────────────────────────────────────────────────────

FULL_QASM_CODE = re.compile(
    r'OPENQASM\s+\d+(\.\d+)?\s*;'       # version
    r'(?:\s*include\s+"[\w.]+"\s*;)*'    # optional includes
    r'(?:\s*qreg\s+\w+\s*\[\s*\d+\s*\]\s*;)*'   # qreg declarations
    r'(?:\s*creg\s+\w+\s*\[\s*\d+\s*\]\s*;)*'   # creg declarations
    r'(?:\s*[\w\d]+\s+[\w\d\[\],\s]+\s*;)*',    # gate instructions
    re.IGNORECASE | re.DOTALL
)

# ─────────────────────────────────────────────────────────────
# 12. MASTER QUANTUM DOMAIN PATTERN (catch-all)
# ─────────────────────────────────────────────────────────────

QUANTUM_DOMAIN_MASTER = re.compile(
    r'\b('
    r'qubit|qubits|quantum|entangle|superpose|decohere|'
    r'teleport|quantum\s*gate|quantum\s*circuit|quantum\s*computer|'
    r'quantum\s*algorithm|quantum\s*error|quantum\s*noise|'
    r'quantum\s*simulation|quantum\s*machine\s*learning|QML|'
    r'variational|ansatz|NISQ|fault[_\-]tolerant|'
    r'quantum\s*volume|randomized\s*benchmarking|'
    r'Clifford\s*group|Clifford\s*circuit|T\s*gate|'
    r'magic\s*state|resource\s*state|cluster\s*state|'
    r'measurement[_\-]based\s*quantum\s*computation|MBQC|'
    r'adiabatic\s*quantum\s*computation|AQC'
    r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def scan_text(text: str) -> dict:
    """
    Scan a given text/code for all quantum-related patterns.
    Returns a dictionary with category -> list of matches.
    """
    results = {
        "Single Qubit Gates":       SINGLE_QUBIT_GATE.findall(text),
        "Multi Qubit Gates":        MULTI_QUBIT_GATE.findall(text),
        "Qubit Registers":          QUBIT_REGISTER.findall(text),
        "Classical Registers":      CLASSICAL_REGISTER.findall(text),
        "QASM Instructions":        QASM_GATE_INSTRUCTION.findall(text),
        "Measurements":             MEASUREMENT.findall(text),
        "Ket Notations":            KET_NOTATION.findall(text),
        "Bra Notations":            BRA_NOTATION.findall(text),
        "Bell States":              BELL_STATES.findall(text),
        "Quantum Algorithms":       QUANTUM_ALGORITHMS.findall(text),
        "Error Correction Codes":   QUANTUM_ERROR_CORRECTION.findall(text),
        "Hardware Terms":           QUANTUM_HARDWARE.findall(text),
        "Frameworks & Tools":       QUANTUM_FRAMEWORKS.findall(text),
        "Crypto & Communication":   QUANTUM_CRYPTO.findall(text),
        "Physics Terms":            QUANTUM_PHYSICS_TERMS.findall(text),
        "Qubit Counts":             QUBIT_COUNT.findall(text),
        "General Quantum Terms":    QUANTUM_DOMAIN_MASTER.findall(text),
    }
    # Remove empty categories
    return {k: v for k, v in results.items() if v}


def is_quantum_text(text: str) -> bool:
    """Returns True if the text contains any quantum domain content."""
    return bool(QUANTUM_DOMAIN_MASTER.search(text) or ALL_GATES.search(text))


def extract_qasm_circuits(text: str) -> list:
    """Extract all OpenQASM circuit blocks from a text."""
    return FULL_QASM_CODE.findall(text)


def highlight_matches(text: str, pattern: re.Pattern, tag: str = "**") -> str:
    """Wrap all matches of a pattern in the text with a tag."""
    return pattern.sub(lambda m: f"{tag}{m.group()}{tag}", text)


# ─────────────────────────────────────────────────────────────
# DEMO / TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];

    // Create Bell State
    h q[0];
    cx q[0], q[1];

    // Toffoli Gate
    ccx q[0], q[1], q[2];

    // Measure
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """

    sample_text = """
    The Grover's Algorithm provides a quadratic speedup O(sqrt(N)) over classical search.
    Using a 5-qubit transmon processor with surface code error correction, 
    we prepare the Bell state |Φ+⟩ via the Hadamard gate followed by CNOT.
    Quantum Key Distribution (QKD) using the BB84 protocol ensures secure communication.
    The VQE algorithm with an ansatz circuit is used on NISQ devices.
    """

    print("=" * 60)
    print("  QUANTUM REGEX SCANNER — DEMO OUTPUT")
    print("=" * 60)

    print("\n[1] Scanning QASM Code:")
    for category, matches in scan_text(sample_code).items():
        print(f"  ► {category}: {matches}")

    print("\n[2] Scanning Quantum Text:")
    for category, matches in scan_text(sample_text).items():
        print(f"  ► {category}: {matches}")

    print("\n[3] Is quantum text? →", is_quantum_text(sample_text))
    print("\n[4] Qubit counts found:", QUBIT_COUNT.findall(sample_text))
    print("\n[5] Algorithms found:", QUANTUM_ALGORITHMS.findall(sample_text))
    print("=" * 60)
