"""
Preset loading for lender/model assignments and scenario bundles.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .models import LenderConfig

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_ROOT = _PROJECT_ROOT / "configs"
_LENDERS_DIR = _CONFIGS_ROOT / "lenders"
_SCENARIOS_DIR = _CONFIGS_ROOT / "scenarios"
_PRESET_EXTS = (".yaml", ".yml", ".json", ".toml")

_SCENARIO_KEYS = {
    "weeks",
    "cohort_size",
    "months_per_week",
    "season_mix",
    "seed",
    "arrival_phases",
    "deep_uw_slots_per_week",
    "speed_scoring",
    "custom_tools",
    "info_asymmetry",
    "data_mode",
}
_SEASON_MIX_VALUES = {"gentle", "realistic", "adversarial", "stress", "escalating"}
_INFO_ASYMMETRY_VALUES = {"none", "partial_statements", "redacted"}
_DATA_MODE_VALUES = {"full", "quarterly_only", "aggregate_only", "statements_inline", "lite"}
_SCENARIO_KEY_ALIASES = {
    "mix": "season_mix",
    "deep_uw_slots": "deep_uw_slots_per_week",
}
_SCENARIO_IGNORED_KEYS = {"name", "description"}


def list_lender_presets() -> list[str]:
    return _list_preset_names(_LENDERS_DIR)


def list_scenario_presets() -> list[str]:
    return _list_preset_names(_SCENARIOS_DIR)


def _list_preset_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    names = {
        p.stem
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _PRESET_EXTS
    }
    return sorted(names)


def load_lender_preset(spec: str) -> tuple[list[dict[str, str]], Path]:
    """Load a lender preset from <name> or a file path."""
    path = _resolve_preset_path(spec, kind="lenders")
    payload = _load_structured_config(path)

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("lenders")
    else:
        entries = None

    if not isinstance(entries, list):
        raise ValueError(
            f"Lender preset '{path}' must be a list or contain a top-level 'lenders' list."
        )

    assignments: list[dict[str, str]] = []
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Lender preset '{path}' entry #{idx} must be a mapping.")

        model = str(entry.get("model", "")).strip()
        if not model:
            raise ValueError(f"Lender preset '{path}' entry #{idx} is missing 'model'.")

        lender_id = str(entry.get("id", "")).strip()
        lender_name = str(entry.get("name", "")).strip()
        display_name = str(entry.get("display_name", "")).strip()

        if not lender_id and not lender_name:
            raise ValueError(
                f"Lender preset '{path}' entry #{idx} must include either 'id' or 'name'."
            )

        normalized: dict[str, str] = {"model": model}
        if lender_id:
            normalized["id"] = lender_id
        if lender_name:
            normalized["name"] = lender_name
        if display_name:
            normalized["display_name"] = display_name
        assignments.append(normalized)

    if not assignments:
        raise ValueError(f"Lender preset '{path}' has no lender entries.")

    return assignments, path


def apply_lender_preset(
    base_lenders: list[LenderConfig],
    assignments: list[dict[str, str]],
) -> list[LenderConfig]:
    """Select lender characters and assign models from the preset entries."""
    by_id = {l.id: l for l in base_lenders}
    by_name = {l.name.lower(): l for l in base_lenders}

    selected: list[LenderConfig] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(assignments, start=1):
        lender_id = entry.get("id", "")
        lender_name = entry.get("name", "")
        model = entry["model"]
        display_name = entry.get("display_name", "")

        lender = None
        if lender_id:
            lender = by_id.get(lender_id)
            if lender is None:
                raise ValueError(f"Lender preset entry #{idx} references unknown id '{lender_id}'.")
        if lender_name:
            by_name_match = by_name.get(lender_name.lower())
            if by_name_match is None:
                raise ValueError(
                    f"Lender preset entry #{idx} references unknown name '{lender_name}'."
                )
            if lender is not None and lender.id != by_name_match.id:
                raise ValueError(
                    f"Lender preset entry #{idx} id/name mismatch: '{lender_id}' vs '{lender_name}'."
                )
            lender = by_name_match
        if lender is None:  # defensive; validated in load_lender_preset
            raise ValueError(f"Lender preset entry #{idx} did not resolve a lender.")

        if lender.id in seen_ids:
            raise ValueError(
                f"Lender preset includes lender '{lender.name}' ({lender.id}) more than once."
            )

        configured = copy.deepcopy(lender)
        configured.model = model
        if display_name:
            configured.name = display_name
        selected.append(configured)
        seen_ids.add(lender.id)

    return selected


def load_scenario_preset(spec: str) -> tuple[dict[str, Any], Path]:
    """Load and normalize a scenario preset from <name> or file path."""
    path = _resolve_preset_path(spec, kind="scenarios")
    payload = _load_structured_config(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario preset '{path}' must be a mapping/object.")

    if isinstance(payload.get("scenario"), dict):
        payload = payload["scenario"]

    normalized: dict[str, Any] = {}
    unknown: list[str] = []

    for raw_key, value in payload.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if key in _SCENARIO_IGNORED_KEYS:
            continue
        key = _SCENARIO_KEY_ALIASES.get(key, key)

        if key not in _SCENARIO_KEYS:
            unknown.append(str(raw_key))
            continue

        if key in {
            "weeks",
            "cohort_size",
            "months_per_week",
            "seed",
            "arrival_phases",
            "deep_uw_slots_per_week",
        }:
            normalized[key] = _to_int(key, value, path)
        elif key in {"speed_scoring", "custom_tools"}:
            normalized[key] = _to_bool(key, value, path)
        elif key == "season_mix":
            mix = str(value).strip()
            if mix not in _SEASON_MIX_VALUES:
                options = ", ".join(sorted(_SEASON_MIX_VALUES))
                raise ValueError(
                    f"Scenario preset '{path}' key 'season_mix' must be one of: {options}"
                )
            normalized[key] = mix
        elif key == "info_asymmetry":
            asym = str(value).strip()
            if asym not in _INFO_ASYMMETRY_VALUES:
                options = ", ".join(sorted(_INFO_ASYMMETRY_VALUES))
                raise ValueError(
                    f"Scenario preset '{path}' key 'info_asymmetry' must be one of: {options}"
                )
            normalized[key] = asym
        elif key == "data_mode":
            mode = str(value).strip()
            if mode not in _DATA_MODE_VALUES:
                options = ", ".join(sorted(_DATA_MODE_VALUES))
                raise ValueError(
                    f"Scenario preset '{path}' key 'data_mode' must be one of: {options}"
                )
            normalized[key] = mode
        else:
            normalized[key] = str(value).strip()

    if unknown:
        raise ValueError(
            f"Scenario preset '{path}' has unknown keys: {', '.join(sorted(unknown))}."
        )

    return normalized, path


def _to_int(key: str, value: Any, path: Path) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Scenario preset '{path}' key '{key}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Scenario preset '{path}' key '{key}' must be an integer."
            ) from exc
    raise ValueError(f"Scenario preset '{path}' key '{key}' must be an integer.")


def _to_bool(key: str, value: Any, path: Path) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Scenario preset '{path}' key '{key}' must be a boolean.")


def _resolve_preset_path(spec: str, kind: str) -> Path:
    raw = Path(spec).expanduser()
    if raw.exists():
        if not raw.is_file():
            raise ValueError(f"Preset path '{raw}' is not a file.")
        return raw.resolve()

    if raw.suffix:
        raise ValueError(f"Preset file not found: '{spec}'.")

    default_dir = _LENDERS_DIR if kind == "lenders" else _SCENARIOS_DIR
    search_roots = [default_dir, Path.cwd()]

    candidates: list[Path] = []
    for root in search_roots:
        for ext in _PRESET_EXTS:
            candidates.append(root / f"{spec}{ext}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    available = list_lender_presets() if kind == "lenders" else list_scenario_presets()
    available_text = ", ".join(available) if available else "(none found)"
    raise ValueError(
        f"Unknown {kind.rstrip('s')} preset '{spec}'. "
        f"Looked in '{default_dir}' and current working directory. "
        f"Available presets: {available_text}"
    )


def _load_structured_config(path: Path) -> Any:
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".toml":
        if tomllib is None:  # pragma: no cover
            raise RuntimeError("TOML parser is unavailable in this Python runtime.")
        with open(path, "rb") as f:
            return tomllib.load(f)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "YAML preset support requires PyYAML. Install with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    exts = ", ".join(_PRESET_EXTS)
    raise ValueError(f"Unsupported preset extension '{suffix}'. Use one of: {exts}")
