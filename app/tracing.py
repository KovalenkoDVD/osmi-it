"""Optional Langfuse tracing. Without LANGFUSE_* keys every call is a no-op."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        return None


class Tracing:
    def __init__(self, settings: Settings) -> None:
        self.client = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                log.info("Langfuse tracing enabled (%s)", settings.langfuse_host)
            except Exception:  # tracing must never break the bot
                log.exception("Langfuse init failed, tracing disabled")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def trace(self, name: str, *, user_id: str, session_id: str, input: Any) -> Iterator[Any]:
        if not self.client:
            yield _NoopObservation()
            return
        from langfuse import propagate_attributes

        with (
            propagate_attributes(user_id=user_id, session_id=session_id, trace_name=name),
            self.client.start_as_current_observation(name=name, as_type="chain", input=input) as span,
        ):
            yield span

    @contextmanager
    def observation(self, name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
        if not self.client:
            yield _NoopObservation()
            return
        with self.client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
            yield obs

    def flush(self) -> None:
        if self.client:
            self.client.flush()
