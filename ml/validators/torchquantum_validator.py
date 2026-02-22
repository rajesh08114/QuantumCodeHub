"""
TorchQuantum validator.
"""
import ast
from typing import Dict, List, Optional, Tuple

from ml.validators.base_validator import BaseValidator


class _TorchQuantumAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.constants: Dict[str, int] = {}
        self.devices: Dict[str, int] = {}
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
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        return None

    def _resolve_range_bounds(self, node: ast.AST) -> Optional[Tuple[int, int]]:
        if not isinstance(node, ast.Call):
            return None
        if self._call_name(node.func) != "range":
            return None
        if len(node.args) == 1:
            start = 0
            stop = self._resolve_int(node.args[0])
        elif len(node.args) >= 2:
            start = self._resolve_int(node.args[0])
            stop = self._resolve_int(node.args[1])
        else:
            return None
        if start is None or stop is None:
            return None
        if stop <= start:
            return (start, start)
        return (start, stop - 1)

    def _resolve_wire_indices(self, node: ast.AST) -> List[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return [int(node.value)]
        if isinstance(node, ast.Name):
            const = self.constants.get(node.id)
            if const is not None:
                return [const]
            for frame in reversed(self.loop_bounds_stack):
                if node.id in frame:
                    return [frame[node.id][1]]
            return []
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values: List[int] = []
            for item in node.elts:
                values.extend(self._resolve_wire_indices(item))
            return values
        return []

    def _wire_budget(self) -> Optional[int]:
        if not self.devices:
            return None
        return max(self.devices.values())

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            resolved = self._resolve_int(node.value)
            if resolved is not None:
                self.constants[name] = resolved
                continue
            if isinstance(node.value, ast.Call):
                constructor = self._call_name(node.value.func)
                if constructor in {"QuantumDevice", "QuantumModuleFromOps"}:
                    wires = None
                    for keyword in node.value.keywords or []:
                        if keyword.arg in {"n_wires", "wires"}:
                            wires = self._resolve_int(keyword.value)
                    if wires is None and node.value.args:
                        wires = self._resolve_int(node.value.args[0])
                    if wires is not None and wires > 0:
                        self.devices[name] = wires
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

    def visit_Call(self, node: ast.Call):
        wire_budget = self._wire_budget()
        if wire_budget is not None:
            for keyword in node.keywords or []:
                if keyword.arg == "wires":
                    indices = self._resolve_wire_indices(keyword.value)
                    for index in indices:
                        if index < 0 or index >= wire_budget:
                            self.errors.append(
                                f"TorchQuantum wire index {index} out of range for n_wires={wire_budget} "
                                f"(line {getattr(node, 'lineno', 0)})"
                            )
        self.generic_visit(node)


class TorchQuantumValidator(BaseValidator):
    def __init__(self):
        super().__init__("torchquantum")

    def _get_required_imports(self) -> list:
        return ["torchquantum"]

    def _validate_framework_specific(self, code: str):
        try:
            tree = ast.parse(code)
            analyzer = _TorchQuantumAnalyzer()
            analyzer.visit(tree)
            self.errors.extend(analyzer.errors)
            if not analyzer.devices:
                self.warnings.append("No QuantumDevice(...) detected; wire-size checks are limited.")
        except SyntaxError:
            return
