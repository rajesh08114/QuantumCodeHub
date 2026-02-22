"""
Base validator.
"""
import ast
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class BaseValidator:
    """Base class for framework-specific validators"""
    
    def __init__(self, framework: str):
        self.framework = framework
        self.errors = []
        self.warnings = []
    
    async def validate(self, code: str) -> Dict:
        """
        Main validation method
        
        Returns:
            Dict with validation results
        """
        self.errors = []
        self.warnings = []
        
        try:
            # 1. Syntax validation
            self._validate_syntax(code)
            
            # 2. Import validation
            self._validate_imports(code)
            
            # 3. Framework-specific validation (override in subclasses)
            self._validate_framework_specific(code)
            
            # 4. Quantum-specific checks
            self._validate_quantum_rules(code)
            
            return {
                "passed": len(self.errors) == 0,
                "errors": self.errors,
                "warnings": self.warnings,
                "framework": self.framework
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "passed": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "framework": self.framework
            }
    
    def _validate_syntax(self, code: str):
        """Check Python syntax correctness"""
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
    
    def _validate_imports(self, code: str):
        """Validate required imports are present"""
        required_imports = self._get_required_imports()
        
        for package in required_imports:
            if package not in code:
                self.warnings.append(f"Missing import: {package}")
    
    def _validate_framework_specific(self, code: str):
        """Framework-specific validation (override in subclasses)"""
        pass
    
    def _validate_quantum_rules(self, code: str):
        """Validate general quantum computing rules"""
        # Keep base checks conservative to avoid false positives.
        # Framework-specific validators should implement deeper semantic checks.

        # Measurement before reading results.
        if 'measure' in code and '.result()' in code:
            measure_pos = code.find('measure')
            result_pos = code.find('.result()')
            if result_pos < measure_pos:
                self.warnings.append(
                    "Calling .result() before measurement may cause issues"
                )
    
    def _get_required_imports(self) -> List[str]:
        """Get list of required imports for framework (override in subclasses)"""
        return []
