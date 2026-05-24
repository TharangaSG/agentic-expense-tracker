from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from src.settings import settings
from src.utils.logging_config import get_logger

if settings.LANGFUSE_PUBLIC_KEY:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)

if settings.LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)

if settings.LANGFUSE_BASE_URL:
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.LANGFUSE_BASE_URL)

if settings.LANGFUSE_TRACING_ENVIRONMENT:
    os.environ.setdefault(
        "LANGFUSE_TRACING_ENVIRONMENT",
        settings.LANGFUSE_TRACING_ENVIRONMENT,
    )

from langfuse import get_client, observe, propagate_attributes

logger = get_logger(__name__)


def is_langfuse_enabled() -> bool:
    return settings.LANGFUSE_ENABLED and bool(
        settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY
    )


def sanitize_langfuse_value(value: Any) -> str | None:
    if value is None:
        return None

    sanitized = str(value).strip()
    if not sanitized:
        return None

    return sanitized[:200]


def build_trace_metadata(**metadata: Any) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        sanitized_value = sanitize_langfuse_value(value)
        if sanitized_value is not None:
            cleaned[key] = sanitized_value
    return cleaned


@contextmanager
def trace_attributes(
    *,
    user_id: Any = None,
    session_id: Any = None,
    metadata: dict[str, Any] | None = None,
):
    if not is_langfuse_enabled():
        yield
        return

    cleaned_metadata = build_trace_metadata(**(metadata or {}))
    cleaned_user_id = sanitize_langfuse_value(user_id)
    cleaned_session_id = sanitize_langfuse_value(session_id)

    if not cleaned_metadata and not cleaned_user_id and not cleaned_session_id:
        yield
        return

    with propagate_attributes(
        user_id=cleaned_user_id,
        session_id=cleaned_session_id,
        metadata=cleaned_metadata or None,
    ):
        yield


def trace_url() -> str | None:
    if not is_langfuse_enabled():
        return None

    try:
        return get_client().get_trace_url()
    except Exception as exc:
        logger.debug(f"Unable to resolve Langfuse trace URL: {exc}")
        return None


__all__ = [
    "build_trace_metadata",
    "get_client",
    "is_langfuse_enabled",
    "observe",
    "trace_attributes",
    "trace_url",
]
