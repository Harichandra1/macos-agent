"""
logging.py — minimal structured (JSON-line) logging.

One JSON object per line so logs are grep-able and ingestible by a log platform
without a parser. Used to record each /chat request: session, latency, retrieval
path (KB vs web fallback), hit count, and errors.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if isinstance(getattr(record, "extra_fields", None), dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "macos_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, msg: str, **fields):
    logger.info(msg, extra={"extra_fields": fields})


@contextmanager
def timed(logger: logging.Logger, msg: str, **fields):
    """Log `msg` with elapsed_ms (and any fields) when the block exits."""
    start = time.perf_counter()
    error = None
    try:
        yield
    except Exception as e:  # noqa: BLE001
        error = repr(e)
        raise
    finally:
        fields["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
        if error:
            fields["error"] = error
        log_event(logger, msg, **fields)
