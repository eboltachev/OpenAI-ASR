from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar


class JsonFormatter(logging.Formatter):
    _reserved: ClassVar[set[str]] = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in self._reserved and key not in {"message", "asctime"}:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class AsyncLogging:
    def __init__(
        self,
        *,
        level: str,
        file_path: Path,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        self._listener: logging.handlers.QueueListener | None = None
        self._level = getattr(logging, level, logging.INFO)
        self._file_path = file_path
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._listener is not None:
                return
            formatter = JsonFormatter()
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(formatter)
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                self._file_path,
                maxBytes=self._max_bytes,
                backupCount=self._backup_count,
                encoding="utf-8",
            )
            rotating.setFormatter(formatter)
            self._listener = logging.handlers.QueueListener(
                self._queue, stream, rotating, respect_handler_level=True
            )
            root = logging.getLogger()
            root.handlers.clear()
            root.setLevel(self._level)
            root.addHandler(logging.handlers.QueueHandler(self._queue))
            self._listener.start()

    def stop(self) -> None:
        with self._lock:
            if self._listener is not None:
                self._listener.stop()
                self._listener = None
