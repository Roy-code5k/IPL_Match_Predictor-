"""Prediction log (Phase 7).

Every prediction the API makes (single, live or batch row) is appended as one JSON
line to logs/predictions.jsonl: when, request id, inputs, outputs, model release,
response time, and the error code if it failed.

The API only uses `write` and `read_all`, so a database backend (for example MongoDB)
can replace `JsonlLog` without touching the routes.

Settings (environment variables):
    IPL_LOG_DIR   folder for predictions.jsonl (default: <project>/logs)
    IPL_LOG=off   turn logging off
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]


class PredictionLog(Protocol):
    def write(self, record: dict) -> None: ...
    def read_all(self) -> list[dict]: ...


class JsonlLog:
    def __init__(self, folder: Path):
        self.path = Path(folder) / "predictions.jsonl"
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partly written line is skipped, never fatal
        return records


class NullLog:
    def write(self, record: dict) -> None:
        pass

    def read_all(self) -> list[dict]:
        return []


def open_log() -> PredictionLog:
    if os.environ.get("IPL_LOG", "").lower() == "off":
        return NullLog()
    return JsonlLog(Path(os.environ.get("IPL_LOG_DIR", ROOT / "logs")))
