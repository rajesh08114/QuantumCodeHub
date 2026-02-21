"""
Custom exceptions.
"""
from fastapi import HTTPException, status

class QuantumCodeHubException(Exception):
    """Base exception for QuantumCodeHub"""
    pass

class ValidationError(QuantumCodeHubException):
    """Raised when code validation fails"""
    def __init__(self, message: str, errors: list):
        self.message = message
        self.errors = errors
        super().__init__(message)

class TranspilationError(QuantumCodeHubException):
    """Raised when code transpilation fails"""
    pass

class RAGError(QuantumCodeHubException):
    """Raised when RAG retrieval fails"""
    pass

class LLMError(QuantumCodeHubException):
    """Raised when LLM generation fails"""
    pass

class QuotaExceededError(QuantumCodeHubException):
    """Raised when user quota is exceeded"""
    def __init__(self, message: str, quota_status: dict):
        self.message = message
        self.quota_status = quota_status
        super().__init__(message)

class FrameworkNotSupportedError(QuantumCodeHubException):
    """Raised when unsupported framework is requested"""
    pass

# HTTP Exception helpers
def not_found_exception(detail: str):
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=detail
    )

def bad_request_exception(detail: str):
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail
    )

def unauthorized_exception(detail: str = "Authentication required"):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"}
    )

def forbidden_exception(detail: str = "Access forbidden"):
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail
    )

def internal_server_exception(detail: str = "Internal server error"):
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail
    )
