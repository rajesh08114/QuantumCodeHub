from fastapi import Request, status
from fastapi.responses import JSONResponse
from core.config import settings
from core.security import decode_access_token
from services.quota_service import quota_service
import json
import logging
import time
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)

_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key"}
_SENSITIVE_FIELDS = {"password", "token", "access_token", "refresh_token", "secret", "api_key"}


def _masked_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for key, value in headers.items():
        normalized = key.lower()
        if normalized in _SENSITIVE_HEADERS:
            sanitized[key] = _masked_value(value)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: Dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in _SENSITIVE_FIELDS:
                sanitized[key] = "***"
            else:
                sanitized[key] = _sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [_sanitize_payload(item) for item in payload]
    return payload


def _preview_body(raw_body: bytes) -> str:
    if not raw_body:
        return ""

    max_chars = max(200, int(settings.LOG_HTTP_BODY_MAX_CHARS or 2000))
    text = raw_body.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        return ""

    preview = text
    try:
        parsed = json.loads(text)
        preview = json.dumps(_sanitize_payload(parsed), ensure_ascii=True)
    except Exception:
        # Non-JSON payload.
        pass

    if len(preview) <= max_chars:
        return preview
    return f"{preview[:max_chars]}..."


def _excluded_path(path: str) -> bool:
    excluded = (settings.LOG_HTTP_EXCLUDE_PATHS or "").split(",")
    excludes = {item.strip() for item in excluded if item.strip()}
    return path in excludes


async def request_response_logging_middleware(request: Request, call_next):
    """
    Global HTTP request/response logging middleware with request correlation IDs.
    """
    if _excluded_path(request.url.path):
        return await call_next(request)

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    body_preview = ""
    if settings.LOG_HTTP_REQUEST_BODY:
        try:
            raw_body = await request.body()
            body_preview = _preview_body(raw_body)
            # Preserve body for downstream handlers after logging middleware reads it.
            async def _receive():
                return {"type": "http.request", "body": raw_body, "more_body": False}

            request = Request(request.scope, _receive)
        except Exception as e:
            body_preview = f"<request-body-read-error: {e}>"

    logger.info(
        "HTTP request request_id=%s method=%s path=%s query=%s client=%s headers=%s body=%s",
        request_id,
        request.method,
        request.url.path,
        request.url.query or "",
        request.client.host if request.client else "unknown",
        _sanitize_headers(dict(request.headers.items())),
        body_preview,
    )

    try:
        response = await call_next(request)
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "HTTP request failed request_id=%s method=%s path=%s latency_ms=%s error=%s",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
            e,
        )
        raise

    latency_ms = int((time.perf_counter() - start) * 1000)
    response_body_preview = ""
    if settings.LOG_HTTP_RESPONSE_BODY:
        try:
            raw = getattr(response, "body", b"")
            if isinstance(raw, (bytes, bytearray)):
                response_body_preview = _preview_body(bytes(raw))
        except Exception as e:
            response_body_preview = f"<response-body-read-error: {e}>"

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "HTTP response request_id=%s method=%s path=%s status_code=%s latency_ms=%s body=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        response_body_preview,
    )
    return response


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware for quota and endpoint rate checks on authenticated users.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    if not settings.ENABLE_RATE_LIMITING:
        return await call_next(request)

    # Skip auth endpoints
    if request.url.path.startswith("/api/auth"):
        return await call_next(request)

    # Get user from token
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return await call_next(request)

    try:
        token = auth_header.split(" ")[1]
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        subscription_tier = payload.get("subscription_tier", "free")

        # Check quota
        quota_status = await quota_service.check_quota(user_id, subscription_tier)
        if not quota_status["allowed"]:
            logger.warning(
                "Quota denied request_id=%s user_id=%s endpoint=%s reason=%s",
                request_id,
                user_id,
                request.url.path,
                quota_status.get("reason", "unknown"),
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": quota_status["reason"],
                    "quota_status": quota_status,
                },
            )

        endpoint = request.url.path
        rate_limit_ok = await quota_service.check_rate_limit(
            user_id=user_id,
            endpoint=endpoint,
            max_requests=20,
            window_seconds=60,
        )
        if not rate_limit_ok:
            logger.warning(
                "Rate limit denied request_id=%s user_id=%s endpoint=%s",
                request_id,
                user_id,
                endpoint,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please slow down.", "retry_after": 60},
            )

        response = await call_next(request)

        if 200 <= response.status_code < 300:
            await quota_service.increment_usage(user_id)

        # Add quota headers to response
        response.headers["X-Daily-Limit"] = str(quota_status.get("daily_limit", -1))
        response.headers["X-Daily-Remaining"] = str(quota_status.get("daily_remaining", -1))
        response.headers["X-Monthly-Limit"] = str(quota_status.get("monthly_limit", -1))
        response.headers["X-Monthly-Remaining"] = str(quota_status.get("monthly_remaining", -1))

        return response
    except Exception as e:
        logger.error("Rate limit middleware error request_id=%s error=%s", request_id, e)
        return await call_next(request)
