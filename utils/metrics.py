"""
Performance metrics utilities.
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Callable

# Define metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['endpoint', 'method']
)

LLM_GENERATION_TIME = Histogram(
    'llm_generation_time_seconds',
    'LLM generation time',
    ['framework']
)

RAG_RETRIEVAL_TIME = Histogram(
    'rag_retrieval_time_seconds',
    'RAG retrieval time',
    ['framework']
)

VALIDATION_TIME = Histogram(
    'validation_time_seconds',
    'Code validation time',
    ['framework']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

def track_time(metric: Histogram, **labels):
    """Decorator to track execution time"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        return wrapper
    return decorator

def increment_counter(counter: Counter, **labels):
    """Increment a counter metric"""
    counter.labels(**labels).inc()

class MetricsCollector:
    """Collect and export metrics"""
    
    @staticmethod
    def record_request(endpoint: str, method: str, status: int):
        """Record API request"""
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
    
    @staticmethod
    def record_latency(endpoint: str, method: str, duration: float):
        """Record request latency"""
        REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    @staticmethod
    def record_llm_generation(framework: str, duration: float):
        """Record LLM generation time"""
        LLM_GENERATION_TIME.labels(framework=framework).observe(duration)
    
    @staticmethod
    def record_rag_retrieval(framework: str, duration: float):
        """Record RAG retrieval time"""
        RAG_RETRIEVAL_TIME.labels(framework=framework).observe(duration)
    
    @staticmethod
    def record_cache_hit(cache_type: str):
        """Record cache hit"""
        CACHE_HITS.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_cache_miss(cache_type: str):
        """Record cache miss"""
        CACHE_MISSES.labels(cache_type=cache_type).inc()

metrics = MetricsCollector()
