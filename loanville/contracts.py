"""Contract helpers for integration artifacts and APR normalization."""

from __future__ import annotations

import math
from typing import Any, Mapping


MAX_APR_DECIMAL = 0.50


class APRNormalizationError(ValueError):
    """Raised when APR cannot be normalized deterministically."""


def _ensure_numeric(value: Any, field: str = "apr") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APRNormalizationError(f"{field} must be numeric, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise APRNormalizationError(f"{field} must be finite")
    return numeric


def normalize_apr_to_decimal(raw_apr: Any) -> float:
    """Normalize APR to decimal representation.

    Conversion contract:
    - value <= 1.0 is treated as decimal
    - value > 1.0 is treated as percentage and converted to decimal
    """
    apr = _ensure_numeric(raw_apr)
    if apr < 0:
        raise APRNormalizationError("apr must be >= 0")
    if apr <= 1.0:
        return apr
    return apr / 100.0


def apr_within_sanity_limit(apr_decimal: float, max_apr_decimal: float = MAX_APR_DECIMAL) -> bool:
    """Return True if APR is within policy sanity bounds."""
    apr = _ensure_numeric(apr_decimal, field="apr_decimal")
    return 0.0 <= apr <= max_apr_decimal


def decimal_apr_to_percentage(apr_decimal: Any) -> float:
    """Convert decimal APR to percentage scale."""
    apr = normalize_apr_to_decimal(apr_decimal)
    return apr * 100.0


def extract_apr_from_payload(payload: Mapping[str, Any]) -> float:
    """Extract and normalize APR from provider payload variants.

    Supported keys (top-level or under ``terms``):
    - apr
    - interest_rate
    - annual_percentage_rate

    Optional key ``apr_scale`` may be ``decimal`` or ``percent`` for explicit scale.
    """
    if not isinstance(payload, Mapping):
        raise APRNormalizationError("payload must be a mapping")

    sources = [payload]
    terms = payload.get("terms")
    if isinstance(terms, Mapping):
        sources.append(terms)

    raw = None
    for source in sources:
        for key in ("apr", "interest_rate", "annual_percentage_rate"):
            if key in source:
                raw = source[key]
                break
        if raw is not None:
            break

    if raw is None:
        raise APRNormalizationError("payload does not contain a recognized APR field")

    scale = payload.get("apr_scale")
    if scale is None and isinstance(terms, Mapping):
        scale = terms.get("apr_scale")

    if scale is None:
        return normalize_apr_to_decimal(raw)

    if not isinstance(scale, str):
        raise APRNormalizationError("apr_scale must be a string when provided")

    normalized_scale = scale.strip().lower()
    numeric = _ensure_numeric(raw)

    if normalized_scale == "decimal":
        if numeric < 0:
            raise APRNormalizationError("apr must be >= 0")
        return numeric

    if normalized_scale == "percent":
        if numeric < 0:
            raise APRNormalizationError("apr must be >= 0")
        return numeric / 100.0

    raise APRNormalizationError(f"unsupported apr_scale '{scale}'")
