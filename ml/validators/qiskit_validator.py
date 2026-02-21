"""
Qiskit validator.
"""
from ml.validators.base_validator import BaseValidator
import re

class QiskitValidator(BaseValidator):
    """Qiskit-specific code validator"""
    
    def __init__(self):
        super().__init__("qiskit")
        
        # Qiskit API patterns
        self.valid_gates = [
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'p', 'u',
            'cx', 'cy', 'cz', 'swap', 'ccx', 'ccz',
            'measure', 'barrier', 'reset'
        ]
    
    def _get_required_imports(self) -> list:
        return ['qiskit', 'QuantumCircuit']
    
    def _validate_framework_specific(self, code: str):
        """Qiskit-specific validation"""
        
        # 1. Check for deprecated APIs
        deprecated = {
            'execute(': 'Use transpile() + backend.run() instead of execute()',
            'IBMQ.load_account': 'Use QiskitRuntimeService instead',
            'Aer.get_backend': 'Use AerSimulator instead'
        }
        
        for pattern, message in deprecated.items():
            if pattern in code:
                self.warnings.append(f"Deprecated: {message}")
        
        # 2. Validate gate usage
        gate_pattern = r'qc\.(\w+)\('
        used_gates = re.findall(gate_pattern, code)
        
        for gate in used_gates:
            if gate not in self.valid_gates:
                self.warnings.append(f"Unknown or unsupported gate: {gate}")
        
        # 3. Check for common mistakes
        if 'QuantumCircuit(' in code:
            # Ensure measurement register is proper size
            if 'measure_all()' not in code and 'measure(' in code:
                measure_calls = len(re.findall(r'\.measure\(', code))
                self.warnings.append(
                    f"Found {measure_calls} individual measure() calls. "
                    "Consider using measure_all() instead."
                )
        
        # 4. Check parameter passing
        if 'Parameter(' in code and 'bind_parameters' not in code:
            self.warnings.append(
                "Parameters defined but bind_parameters() not found"
            )