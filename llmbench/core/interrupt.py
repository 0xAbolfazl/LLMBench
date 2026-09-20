"""Cooperative interruption for long-running benchmark operations.

A benchmark can run long enough that a user wants to stop it mid-flight. It is
interrupted through a signal passed into the operation:

    signal = StopSignal()
    ...
    execute_matrix(plan, prompts, interrupt=signal)

    # from another thread
    signal.request_stop()

The operation checks the signal at every point where it can stop safely and
raises :class:`RunInterrupted`. Interrupting is cooperative and therefore never
leaves a half-written file behind: the operation cleans up its own partial
work before the exception propagates, and records an entry in the event log so
the interruption is visible afterwards.
"""

from __future__ import annotations

import threading

from .eventlog import record_event

# How often a blocking wait re-checks the signal.
POLL_SECONDS = 0.2


class RunInterrupted(Exception):
    """Raised by a long-running operation when its stop signal is set."""


class StopSignal:
    """Thread-safe stop flag shared with a running operation.

    The operation and the caller requesting the stop are always on different
    threads, so the flag is a :class:`threading.Event`: setting it is atomic
    and a waiter can block on it instead of sleeping in a loop.
    """

    def __init__(self) -> None:
        """Create a signal that has not been tripped."""
        self._event = threading.Event()
        self._reason: str | None = None
        self._lock = threading.Lock()

    def request_stop(self, reason: str | None = None) -> None:
        """Request interruption.

        Safe to call more than once; the first reason given is kept.

        Args:
            reason: Optional explanation recorded with the interruption.
        """
        with self._lock:
            if self._reason is None:
                self._reason = reason or "Stopped by request"

        self._event.set()

    @property
    def stopped(self) -> bool:
        """bool: Whether a stop has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """str | None: Why the operation was stopped, once it has been."""
        with self._lock:
            return self._reason

    def raise_if_stopped(self) -> None:
        """Stop the operation if interruption has been requested.

        Call this at every point where the operation can stop without leaving
        partial work behind.

        Raises:
            RunInterrupted: If the signal has been tripped.
        """
        if self._event.is_set():
            raise RunInterrupted(self.reason or "Stopped by request")

    def wait(self, timeout: float) -> bool:
        """Sleep, but wake immediately if a stop is requested.

        Args:
            timeout: Seconds to wait at most.

        Returns:
            bool: True if the signal was tripped during the wait.
        """
        return self._event.wait(timeout)


def record_interruption(
    component: str,
    action: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Record that an operation was interrupted.

    Interrupting is meant to leave nothing behind except this entry, so every
    operation calls it on the way out — that log line is the only lasting
    trace an interrupted operation is allowed to leave.

    Args:
        component: Component the operation belongs to.
        action: Action that was interrupted.
        message: Human-readable description of the interruption.
        details: Optional metadata, for example what was cleaned up.
    """
    record_event(
        level="WARNING",
        component=component,
        action=action,
        message=message,
        details=details or {},
    )
