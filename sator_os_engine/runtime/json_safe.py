"""Make a job result safe to return as JSON."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def sanitize_for_json(value: Any) -> Any:
    """Replace non-finite floats with null and reject values JSON cannot encode."""
    cleaned = _clean(value)
    try:
        json.dumps(cleaned, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"result is not JSON serialisable: {exc}") from exc
    return cleaned


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
