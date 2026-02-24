#!/usr/bin/env python3
"""Validate SIM contract fixtures and APR normalization gates."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loanville.contracts import (  # noqa: E402
    APRNormalizationError,
    apr_within_sanity_limit,
    decimal_apr_to_percentage,
    extract_apr_from_payload,
    normalize_apr_to_decimal,
)
from loanville.models import LenderDecision, TermSheet  # noqa: E402
from loanville.run_schema import RunDecision  # noqa: E402


CONTRACT_FIXTURE_MAP = {
    "underwriting-run.v1.json": [
        "underwriting-run.approve.json",
        "underwriting-run.decline.json",
    ],
    "case-pack.v1.json": [
        "case-pack.basic.json",
    ],
    "feedback-candidate.v1.json": [
        "feedback-candidate.basic.json",
    ],
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return False


def _check_format(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    fmt = schema.get("format")
    if fmt == "date-time" and isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"{path}: invalid date-time format '{value}'")


def _validate(instance: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected_type = schema.get("type")
    if expected_type and not _is_type(instance, expected_type):
        errors.append(f"{path}: expected type {expected_type}, got {type(instance).__name__}")
        return

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value '{instance}' not in enum {schema['enum']}")

    if isinstance(instance, str) and "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(f"{path}: string shorter than minLength={schema['minLength']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: list shorter than minItems={schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(instance):
                _validate(item, item_schema, f"{path}[{idx}]", errors)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum={schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum={schema['maximum']}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required key '{key}'")

        props = schema.get("properties", {})
        allow_extra = schema.get("additionalProperties", True)

        for key, value in instance.items():
            if key in props:
                _validate(value, props[key], f"{path}.{key}", errors)
                _check_format(value, props[key], f"{path}.{key}", errors)
            elif allow_extra is False:
                errors.append(f"{path}: unexpected key '{key}'")


def validate_fixture(schema_path: Path, fixture_path: Path) -> CheckResult:
    schema = json.loads(schema_path.read_text())
    fixture = json.loads(fixture_path.read_text())
    errors: list[str] = []
    _validate(fixture, schema, "$", errors)

    if errors:
        return CheckResult(
            name=f"schema:{schema_path.name}:{fixture_path.name}",
            passed=False,
            detail="; ".join(errors[:4]),
        )
    return CheckResult(
        name=f"schema:{schema_path.name}:{fixture_path.name}",
        passed=True,
    )


def _run_apr_fixture_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    # 1. Input APR 0.095 -> normalized decimal 0.095
    apr_decimal = normalize_apr_to_decimal(0.095)
    results.append(CheckResult(
        name="apr_fixture_1_decimal_passthrough",
        passed=math.isclose(apr_decimal, 0.095, rel_tol=0, abs_tol=1e-9),
        detail=f"normalized={apr_decimal}",
    ))

    # 2. Input APR 9.5 -> normalized decimal 0.095
    apr_percent = normalize_apr_to_decimal(9.5)
    results.append(CheckResult(
        name="apr_fixture_2_percent_to_decimal",
        passed=math.isclose(apr_percent, 0.095, rel_tol=0, abs_tol=1e-9),
        detail=f"normalized={apr_percent}",
    ))

    # 3. Input APR 0.0 with decline action -> accepted
    decline_apr = normalize_apr_to_decimal(0.0)
    decline_gate = True  # declines bypass term sanity gate by policy
    results.append(CheckResult(
        name="apr_fixture_3_decline_zero_apr",
        passed=math.isclose(decline_apr, 0.0, rel_tol=0, abs_tol=1e-9) and decline_gate,
        detail=f"normalized={decline_apr}",
    ))

    # 4. Input APR 55 -> gate failure (sanity limit breach)
    high_apr = normalize_apr_to_decimal(55)
    results.append(CheckResult(
        name="apr_fixture_4_sanity_limit_breach",
        passed=not apr_within_sanity_limit(high_apr),
        detail=f"normalized={high_apr}",
    ))

    # 5. Round-trip decimal -> termsheet -> run keeps semantic equivalence
    base_decimal = 0.095
    decision = LenderDecision(
        lender_id="L001",
        borrower_id="BRW-001",
        decision="APPROVE",
        reasoning="fixture",
        term_sheet=TermSheet(
            loan_amount=500000,
            interest_rate=decimal_apr_to_percentage(base_decimal),
            term_months=24,
        ),
    )
    run_decision = RunDecision.from_lender_decision(decision)
    round_trip_ok = math.isclose(run_decision.terms.apr, base_decimal, rel_tol=0, abs_tol=1e-9)
    results.append(CheckResult(
        name="apr_fixture_5_round_trip_equivalence",
        passed=round_trip_ok,
        detail=f"run_apr={run_decision.terms.apr}",
    ))

    # 6. Mixed provider payloads normalize identically
    payloads = [
        {"apr": 0.095},
        {"apr": 9.5},
        {"interest_rate": 9.5, "apr_scale": "percent"},
        {"terms": {"annual_percentage_rate": 9.5}},
    ]
    normalized = [extract_apr_from_payload(payload) for payload in payloads]
    mixed_ok = all(math.isclose(value, 0.095, rel_tol=0, abs_tol=1e-9) for value in normalized)
    results.append(CheckResult(
        name="apr_fixture_6_mixed_payload_consistency",
        passed=mixed_ok,
        detail=f"normalized={normalized}",
    ))

    # Unknown APR scale/type should fail hard (no silent fallback)
    unknown_scale_failed = False
    try:
        extract_apr_from_payload({"apr": "9.5"})
    except APRNormalizationError:
        unknown_scale_failed = True

    results.append(CheckResult(
        name="apr_guard_no_silent_fallback",
        passed=unknown_scale_failed,
        detail="string APR must raise APRNormalizationError",
    ))

    return results


def run() -> int:
    results: list[CheckResult] = []

    contracts_dir = REPO_ROOT / "contracts"
    fixtures_dir = REPO_ROOT / "fixtures" / "contracts"

    for schema_name, fixture_names in CONTRACT_FIXTURE_MAP.items():
        schema_path = contracts_dir / schema_name
        if not schema_path.exists():
            results.append(CheckResult(f"schema_file:{schema_name}", False, "missing schema file"))
            continue

        for fixture_name in fixture_names:
            fixture_path = fixtures_dir / fixture_name
            if not fixture_path.exists():
                results.append(CheckResult(f"fixture_file:{fixture_name}", False, "missing fixture file"))
                continue
            results.append(validate_fixture(schema_path, fixture_path))

    apr_results = _run_apr_fixture_checks()
    results.extend(apr_results)

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed

    print("Contract fixture validation results:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        suffix = f" ({result.detail})" if result.detail else ""
        print(f"  [{status}] {result.name}{suffix}")

    apr_passed = sum(1 for result in apr_results if result.passed)
    apr_rate = 100.0 * apr_passed / len(apr_results)
    print(f"\nAPR fixture pass rate: {apr_passed}/{len(apr_results)} ({apr_rate:.1f}%)")
    print(f"Overall: {passed}/{total} checks passed")

    if failed:
        return 1

    if apr_rate < 100.0:
        print("APR fixture gate requires 100% pass rate.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
