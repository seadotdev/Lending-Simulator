"""
LLM response cache — content-addressable storage for underwriting decisions.

Caches LLM responses keyed by a hash of (model, dossier_content, policy_params).
When the same model evaluates the same borrower with the same policy, the cached
response is returned instead of making an API call.

This is the single biggest token-efficiency improvement: a typical 10-week season
with 5 borrowers/week and 3 lenders generates 150 LLM evaluations.  If you re-run
the same scenario to test a scoring change, the cache saves 100% of those tokens.

Usage::

    cache = ResponseCache("cache/responses")
    key = cache.make_key(model="claude-3.5-sonnet", dossier=dossier_dict, policy={})

    hit = cache.get(key)
    if hit is not None:
        decision = hit  # Skip LLM call
    else:
        decision = call_llm(...)
        cache.put(key, decision)

Integration with SimulationEngine::

    engine = SimulationEngine(borrowers, lenders, mock=True, response_cache="cache/")
    # Second run with same borrowers+lenders: all responses served from cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def _stable_hash(data: str) -> str:
    """SHA-256 hash of string data, returned as hex."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def make_cache_key(
    model: str,
    borrower_id: str,
    dossier: dict[str, Any],
    policy_params: Optional[dict[str, Any]] = None,
    data_mode: str = "full",
) -> str:
    """Build a content-addressable cache key.

    The key is a hash of the inputs that determine the LLM response:
    model identity, borrower dossier content, policy parameters, and
    data presentation mode.

    Changing any of these invalidates the cache for this entry.
    """
    # Normalize: sort keys for deterministic serialization
    key_parts = {
        "model": model,
        "borrower_id": borrower_id,
        "data_mode": data_mode,
    }
    if policy_params:
        key_parts["policy"] = policy_params

    # Use a subset of dossier fields that are decision-relevant
    # (exclude formatting artifacts that don't change the decision)
    dossier_signal = {
        k: dossier.get(k)
        for k in sorted(dossier.keys())
        if dossier.get(k) is not None
    }
    key_parts["dossier"] = dossier_signal

    canonical = json.dumps(key_parts, sort_keys=True, default=str)
    return _stable_hash(canonical)


@dataclass
class CacheEntry:
    """A cached LLM response."""
    key: str
    model: str
    borrower_id: str
    decision: str  # "APPROVE" | "REJECT" | "PASS"
    reasoning: str = ""
    term_sheet: Optional[dict] = None
    # Metadata
    hits: int = 0
    created_from: str = ""  # "mock" | "los" | "manual"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "model": self.model,
            "borrower_id": self.borrower_id,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "term_sheet": self.term_sheet,
            "hits": self.hits,
            "created_from": self.created_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry:
        return cls(
            key=data["key"],
            model=data.get("model", ""),
            borrower_id=data.get("borrower_id", ""),
            decision=data.get("decision", "REJECT"),
            reasoning=data.get("reasoning", ""),
            term_sheet=data.get("term_sheet"),
            hits=data.get("hits", 0),
            created_from=data.get("created_from", ""),
        )


class ResponseCache:
    """Content-addressable LLM response cache.

    Stores responses as individual JSON files keyed by content hash.
    Designed for fast lookup and easy inspection.

    Directory structure::

        cache_dir/
        ├── ab/
        │   └── ab3def...json
        ├── cd/
        │   └── cd9876...json
        └── stats.json
    """

    def __init__(self, cache_dir: str | Path = "cache/responses") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key_path(self, key: str) -> Path:
        """Two-level directory structure to avoid huge flat dirs."""
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[CacheEntry]:
        """Look up a cached response. Returns None on miss."""
        path = self._key_path(key)
        if not path.exists():
            self._misses += 1
            return None

        data = json.loads(path.read_text())
        entry = CacheEntry.from_dict(data)
        entry.hits += 1

        # Update hit count on disk
        path.write_text(json.dumps(entry.to_dict(), indent=2))

        self._hits += 1
        return entry

    def put(
        self,
        key: str,
        model: str,
        borrower_id: str,
        decision: str,
        reasoning: str = "",
        term_sheet: Optional[dict] = None,
        created_from: str = "",
    ) -> CacheEntry:
        """Store a response in the cache."""
        entry = CacheEntry(
            key=key,
            model=model,
            borrower_id=borrower_id,
            decision=decision,
            reasoning=reasoning,
            term_sheet=term_sheet,
            created_from=created_from,
        )

        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry.to_dict(), indent=2))
        return entry

    def has(self, key: str) -> bool:
        """Check if a key exists without loading."""
        return self._key_path(key).exists()

    def clear(self) -> int:
        """Remove all cached entries. Returns count removed."""
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for f in subdir.glob("*.json"):
                    f.unlink()
                    count += 1
                subdir.rmdir()
        return count

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics for current session."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }

    def size(self) -> int:
        """Count total cached entries."""
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                count += sum(1 for _ in subdir.glob("*.json"))
        return count
