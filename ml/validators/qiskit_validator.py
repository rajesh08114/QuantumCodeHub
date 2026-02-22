"""
Qiskit validator.
"""
import ast
from typing import Dict, List, Optional, Tuple

from ml.validators.base_validator import BaseValidator


class _QiskitSemanticAnalyzer(ast.NodeVisitor):
    """Lightweight AST-based semantic checks for Qiskit circuit indexing."""

    _SINGLE_QUBIT_GATES = {
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "id",
        "i",
        "p",
        "u",
        "rx",
        "ry",
        "rz",
        "reset",
    }
    _TWO_QUBIT_GATES = {
        "cx",
        "cy",
        "cz",
        "swap",
        "ch",
        "cp",
        "crx",
        "cry",
        "crz",
        "rxx",
        "ryy",
        "rzz",
        "rzx",
        "ecr",
    }
    _THREE_QUBIT_GATES = {
        "ccx",
        "ccz",
        "cswap",
    }

    def __init__(self):
        self.constants: Dict[str, int] = {}
        self.quantum_registers: Dict[str, int] = {}
        self.circuits: Dict[str, int] = {}
        self.loop_bounds_stack: List[Dict[str, Tuple[int, int]]] = []
        self.errors: List[str] = []

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _resolve_int(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._resolve_int(node.operand)
            return -value if value is not None else None
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = self._resolve_int(node.left)
            right = self._resolve_int(node.right)
            if left is None or right is None:
                return None
            return left + right if isinstance(node.op, ast.Add) else left - right
        return None

    def _current_loop_bounds(self) -> Dict[str, Tuple[int, int]]:
        merged: Dict[str, Tuple[int, int]] = {}
        for frame in self.loop_bounds_stack:
            merged.update(frame)
        return merged

    def _resolve_index_upper_bound(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            const_value = self.constants.get(node.id)
            if const_value is not None:
                return const_value
            loop_bound = self._current_loop_bounds().get(node.id)
            if loop_bound is not None:
                return loop_bound[1]
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._resolve_index_upper_bound(node.operand)
            return -value if value is not None else None
        if isinstance(node, ast.Subscript):
            target = node.slice
            if isinstance(target, ast.Index):  # pragma: no cover - py<3.9 compat
                target = target.value
            return self._resolve_index_upper_bound(target)
        return self._resolve_int(node)

    def _collect_indices(self, node: ast.AST) -> List[int]:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values: List[int] = []
            for item in node.elts:
                values.extend(self._collect_indices(item))
            return values
        value = self._resolve_index_upper_bound(node)
        return [value] if value is not None else []

    def _resolve_range_bounds(self, node: ast.AST) -> Optional[Tuple[int, int]]:
        if not isinstance(node, ast.Call):
            return None
        if self._call_name(node.func) != "range":
            return None

        if len(node.args) == 1:
            start, stop, step = 0, self._resolve_int(node.args[0]), 1
        elif len(node.args) >= 2:
            start = self._resolve_int(node.args[0])
            stop = self._resolve_int(node.args[1])
            step = self._resolve_int(node.args[2]) if len(node.args) > 2 else 1
        else:
            return None

        if start is None or stop is None or step is None or step == 0:
            return None
        if step < 0:
            # Conservative for descending ranges.
            return None
        if stop <= start:
            return (start, start)
        return (start, stop - 1)

    def _register_circuit(self, name: str, call: ast.Call):
        size: Optional[int] = None

        for keyword in call.keywords or []:
            if keyword.arg in {"num_qubits", "n_qubits"}:
                size = self._resolve_int(keyword.value)
                break

        if size is None and call.args:
            first_arg = call.args[0]
            first_resolved = self._resolve_int(first_arg)
            if first_resolved is not None and first_resolved > 0:
                # QuantumCircuit(num_qubits, num_clbits, ...) -> first positional is qubit count.
                size = first_resolved
            else:
                # QuantumCircuit(qreg1, qreg2, ...) -> sum declared quantum registers.
                register_total = 0
                used_register = False
                for arg in call.args:
                    if isinstance(arg, ast.Name):
                        qreg_size = self.quantum_registers.get(arg.id)
                        if qreg_size is not None:
                            register_total += qreg_size
                            used_register = True
                if used_register and register_total > 0:
                    size = register_total

        if size is not None and size > 0:
            self.circuits[name] = size

    def visit_Assign(self, node: ast.Assign):
        value = node.value

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id

            constant = self._resolve_int(value)
            if constant is not None:
                self.constants[name] = constant
                continue

            if isinstance(value, ast.Call):
                constructor = self._call_name(value.func)
                if constructor == "QuantumRegister" and value.args:
                    size = self._resolve_int(value.args[0])
                    if size is not None and size > 0:
                        self.quantum_registers[name] = size
                elif constructor == "QuantumCircuit":
                    self._register_circuit(name, value)

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        pushed = False
        if isinstance(node.target, ast.Name):
            bounds = self._resolve_range_bounds(node.iter)
            if bounds is not None:
                self.loop_bounds_stack.append({node.target.id: bounds})
                pushed = True
        self.generic_visit(node)
        if pushed:
            self.loop_bounds_stack.pop()

    def _check_index_bounds(self, method: str, circuit_name: str, circuit_size: int, indices: List[int], lineno: int):
        for index in indices:
            if index < 0 or index >= circuit_size:
                self.errors.append(
                    f"Qubit index {index} out of range for circuit '{circuit_name}' of size {circuit_size} "
                    f"(method {method}, line {lineno})"
                )

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return
        if not isinstance(node.func.value, ast.Name):
            self.generic_visit(node)
            return

        circuit_name = node.func.value.id
        circuit_size = self.circuits.get(circuit_name)
        if circuit_size is None:
            self.generic_visit(node)
            return

        method = (node.func.attr or "").lower()
        args = list(node.args or [])
        indices: List[int] = []

        if method in self._SINGLE_QUBIT_GATES and args:
            indices = self._collect_indices(args[0])
        elif method in self._TWO_QUBIT_GATES and len(args) >= 2:
            indices = self._collect_indices(args[0]) + self._collect_indices(args[1])
        elif method in self._THREE_QUBIT_GATES and len(args) >= 3:
            indices = (
                self._collect_indices(args[0])
                + self._collect_indices(args[1])
                + self._collect_indices(args[2])
            )
        elif method == "measure" and args:
            indices = self._collect_indices(args[0])
        elif method == "barrier":
            for arg in args:
                indices.extend(self._collect_indices(arg))

        if indices:
            self._check_index_bounds(method, circuit_name, circuit_size, indices, getattr(node, "lineno", 0))

        self.generic_visit(node)


class QiskitValidator(BaseValidator):
    """Qiskit-specific code validator."""

    def __init__(self):
        super().__init__("qiskit")

        self.valid_gates = {
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sdg",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "u",
            "cx",
            "cy",
            "cz",
            "swap",
            "ccx",
            "ccz",
            "measure",
            "barrier",
            "reset",
            "sx",
            "sxdg",
            "cp",
            "crx",
            "cry",
            "crz",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ecr",
        }

    def _get_required_imports(self) -> list:
        return ["qiskit", "QuantumCircuit"]

    def _validate_framework_specific(self, code: str):
        """Qiskit-specific validation."""

        deprecated = {
            "execute(": "Use transpile() + backend.run() instead of execute()",
            "IBMQ.load_account": "Use QiskitRuntimeService instead",
            "Aer.get_backend": "Use AerSimulator instead",
        }
        for pattern, message in deprecated.items():
            if pattern in code:
                self.warnings.append(f"Deprecated: {message}")

        try:
            tree = ast.parse(code)
            analyzer = _QiskitSemanticAnalyzer()
            analyzer.visit(tree)
            self.errors.extend(analyzer.errors)
        except SyntaxError:
            # Syntax errors are already handled by BaseValidator.
            return
