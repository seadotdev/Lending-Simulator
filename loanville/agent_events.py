"""JSONL event logger for agent sim runs."""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventLogger:
    """Append-only JSONL event logger."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a")
        self._start = time.time()

    def log(self, event_type: str, **data: object) -> None:
        record = {
            "t": round(time.time() - self._start, 3),
            "ts": time.time(),
            "type": event_type,
            **data,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        events = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events
