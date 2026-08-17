"""Event trace: the harness's only output channel for "what happened".

Everything the agent does — a model call, a tool call, a node entry, a stop-rule
firing — becomes an Event. Nothing in this package prints to stdout; the CLI
renders the trace instead. That separation is what makes runs replayable and
testable: a run is its event log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


class Trace:
    """Append-only event log, optionally mirrored to a JSONL file.

    `clock` is injectable so tests get deterministic timestamps.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._events: list[Event] = []
        self._clock = clock
        self._path = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("")

    def emit(self, type: str, **payload: Any) -> Event:
        event = Event(seq=len(self._events), ts=self._clock(), type=type, payload=payload)
        self._events.append(event)
        if self._path is not None:
            with self._path.open("a") as fh:
                fh.write(event.to_json() + "\n")
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, type: str) -> list[Event]:
        return [e for e in self._events if e.type == type]

    def types(self) -> list[str]:
        return [e.type for e in self._events]

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


def read_trace(path: str | Path) -> list[Event]:
    """Load a JSONL trace back into Event objects."""
    events: list[Event] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        events.append(Event(seq=raw["seq"], ts=raw["ts"], type=raw["type"], payload=raw["payload"]))
    return events
