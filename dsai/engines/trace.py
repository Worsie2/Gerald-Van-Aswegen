"""The live analysis trace.

Shows what the platform is doing, step by step, in the user's terms. It reports
conclusions and progress — never private reasoning — so the run is auditable
without pretending to expose a thought process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from dsai.core.schema import JsonMixin, TraceEvent

TraceCallback = Callable[[TraceEvent], None]


@dataclass
class AnalysisTrace(JsonMixin):
    events: list[TraceEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    callback: TraceCallback | None = None

    def add(self, step: str, status: str = "done", detail: str = "",
            elapsed_s: float | None = None) -> TraceEvent:
        event = TraceEvent(step=step, status=status, detail=detail, elapsed_s=elapsed_s)
        self.events.append(event)
        if self.callback:
            try:
                self.callback(event)
            except Exception:
                pass  # a broken UI callback must never stop the analysis
        return event

    def start(self, step: str, detail: str = "") -> "TraceTimer":
        return TraceTimer(self, step, detail)

    def render(self) -> str:
        symbols = {"done": "✓", "running": "…", "warning": "!", "failed": "✗", "skipped": "–"}
        lines = []
        for event in self.events:
            timing = f"  ({event.elapsed_s:.2f}s)" if event.elapsed_s else ""
            detail = f" — {event.detail}" if event.detail else ""
            lines.append(f"{symbols.get(event.status, '·')} {event.step}{detail}{timing}")
        return "\n".join(lines)

    @property
    def total_seconds(self) -> float:
        return round(time.time() - self.started_at, 2)

    def failures(self) -> list[TraceEvent]:
        return [e for e in self.events if e.status == "failed"]

    def warnings(self) -> list[TraceEvent]:
        return [e for e in self.events if e.status == "warning"]


class TraceTimer:
    """Context manager that records a step and how long it took."""

    def __init__(self, trace: AnalysisTrace, step: str, detail: str = "") -> None:
        self.trace = trace
        self.step = step
        self.detail = detail
        self.started = 0.0
        self.event: TraceEvent | None = None

    def __enter__(self) -> "TraceTimer":
        self.started = time.perf_counter()
        self.event = self.trace.add(self.step, status="running", detail=self.detail)
        return self

    def update(self, detail: str) -> None:
        self.detail = detail
        if self.event:
            self.event.detail = detail

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = round(time.perf_counter() - self.started, 3)
        if self.event:
            self.event.elapsed_s = elapsed
            if exc_type is not None:
                self.event.status = "failed"
                self.event.detail = f"{self.detail} — {exc_type.__name__}: {exc}".strip(" —")
            else:
                self.event.status = "done"
                self.event.detail = self.detail
            if self.trace.callback:
                try:
                    self.trace.callback(self.event)
                except Exception:
                    pass
        return False  # never swallow the exception
