"""
Structured logging utilities.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
import json

class StructuredLogger:
    """Structured logging for better observability"""
    
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def log_request(self, endpoint: str, user_id: str, method: str, **kwargs):
        """Log API request"""
        log_data = {
            "type": "api_request",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "user_id": user_id,
            "method": method,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error with context"""
        log_data = {
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.logger.error(json.dumps(log_data))
    
    def log_performance(self, operation: str, duration_ms: int, **kwargs):
        """Log performance metrics"""
        log_data = {
            "type": "performance",
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))

# Create default logger
app_logger = StructuredLogger("quantumcodehub", "logs/app.log")
