"""
Code validation service.
"""
from ml.validators.qiskit_validator import QiskitValidator
from ml.validators.base_validator import BaseValidator
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ValidatorService:
    """Service for code validation across frameworks"""
    
    def __init__(self):
        self.validators = {
            "qiskit": QiskitValidator(),
            # Add other framework validators
            # "pennylane": PennyLaneValidator(),
            # "cirq": CirqValidator(),
        }
    
    async def validate(self, code: str, framework: str) -> Dict:
        """
        Validate code for specific framework
        
        Args:
            code: Code to validate
            framework: Framework name
            
        Returns:
            Validation results
        """
        try:
            validator = self.validators.get(framework)
            
            if not validator:
                logger.warning(f"No validator for framework: {framework}")
                # Use base validator as fallback
                validator = BaseValidator(framework)
            
            result = await validator.validate(code)
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "passed": False,
                "errors": [str(e)],
                "warnings": [],
                "framework": framework
            }

# Singleton instance
validator_service = ValidatorService()