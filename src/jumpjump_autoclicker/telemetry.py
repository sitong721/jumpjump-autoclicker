from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class JumpTelemetry:
    """Append-only JSONL logger for jump calibration data."""

    def __init__(self, output_path: Path, enabled: bool = True) -> None:
        self.output_path = output_path
        self.enabled = enabled
        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return

        data = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._json_safe(data), ensure_ascii=False) + "\n")

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def recent_events(self, event: str, limit: int) -> list[dict[str, Any]]:
        if not self.output_path.exists() or limit <= 0:
            return []

        events: list[dict[str, Any]] = []
        with self.output_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("event") == event:
                    events.append(data)

        return events[-limit:]
