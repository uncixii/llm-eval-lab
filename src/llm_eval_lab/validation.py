"""Shared fail-closed validation and stable experiment identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass


def number(
    value: object, name: str, minimum: float = 0, maximum: float | None = None
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def unique_ids(values, name: str) -> None:
    values = list(values)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must be nonempty")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


def fingerprint(value: object) -> str:
    data = json.dumps(
        value,
        default=lambda v: asdict(v) if is_dataclass(v) else _unsupported(v),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(data.encode()).hexdigest()


def _unsupported(value):
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
