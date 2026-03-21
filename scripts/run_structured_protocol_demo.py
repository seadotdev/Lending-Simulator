from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path("artifacts/pre_submission_experiments")


def validate_request(message: dict) -> list[str]:
    errors: list[str] = []
    required = {
        "protocol",
        "requested_fields",
        "acceptable_evidence",
        "disclosure_boundary",
        "retention",
    }
    missing = required - set(message)
    if missing:
        errors.append(f"missing_request_fields:{sorted(missing)}")
    if message.get("protocol") != "bounded_confidential_disclosure":
        errors.append("unsupported_protocol")
    return errors


def validate_response(request: dict, message: dict) -> list[str]:
    errors: list[str] = []
    required = {
        "provided",
        "withheld",
        "evidence_source",
        "disclosure_boundary_acknowledged",
    }
    missing = required - set(message)
    if missing:
        errors.append(f"missing_response_fields:{sorted(missing)}")
        return errors

    provided_keys = set(message["provided"])
    allowed_keys = set(request["requested_fields"])

    if not provided_keys.issubset(allowed_keys):
        extra = sorted(provided_keys - allowed_keys)
        errors.append(f"field_whitelist_violation:{extra}")

    if request["disclosure_boundary"] == "no_client_names" and "client_names" in provided_keys:
        errors.append("boundary_violation:client_names")

    if message["evidence_source"] not in request["acceptable_evidence"]:
        errors.append("evidence_source_nonconformant")

    if not message["disclosure_boundary_acknowledged"]:
        errors.append("boundary_not_acknowledged")

    return errors


def run_episode(episode_id: str, request: dict, borrower_messages: list[dict]) -> dict:
    trace = {
        "episode_id": episode_id,
        "request": request,
        "events": [],
        "final_status": "",
    }

    request_errors = validate_request(request)
    if request_errors:
        trace["events"].append(
            {
                "actor": "enforcer",
                "type": "reject_request",
                "errors": request_errors,
            }
        )
        trace["final_status"] = "invalid_request"
        return trace

    trace["events"].append(
        {
            "actor": "enforcer",
            "type": "accept_request",
            "requested_fields": request["requested_fields"],
        }
    )

    for idx, message in enumerate(borrower_messages, start=1):
        errors = validate_response(request, message)
        if errors:
            trace["events"].append(
                {
                    "actor": "enforcer",
                    "type": "reject_response",
                    "attempt": idx,
                    "errors": errors,
                    "blocked_payload_keys": sorted(message.get("provided", {}).keys()),
                }
            )
            continue

        trace["events"].append(
            {
                "actor": "enforcer",
                "type": "forward_response",
                "attempt": idx,
                "forwarded_keys": sorted(message["provided"].keys()),
                "evidence_source": message["evidence_source"],
            }
        )
        trace["final_status"] = "conformant_exchange"
        return trace

    trace["final_status"] = "escalation_required"
    return trace


def build_demo() -> dict:
    request = {
        "protocol": "bounded_confidential_disclosure",
        "requested_fields": ["bank_feed_12m", "tax_filing_latest"],
        "acceptable_evidence": ["verified_feed", "hmrc_confirmation"],
        "disclosure_boundary": "no_client_names",
        "retention": "session_only",
    }

    episodes = [
        run_episode(
            "clean_exchange",
            request,
            [
                {
                    "provided": {
                        "bank_feed_12m": "<aggregated verified feed>",
                        "tax_filing_latest": "<hmrc confirmation>",
                    },
                    "withheld": [],
                    "evidence_source": "verified_feed",
                    "disclosure_boundary_acknowledged": True,
                }
            ],
        ),
        run_episode(
            "boundary_violation_caught",
            request,
            [
                {
                    "provided": {
                        "bank_feed_12m": "<aggregated verified feed>",
                        "tax_filing_latest": "<hmrc confirmation>",
                        "client_names": ["Grandview Weddings", "Metro Convention Center"],
                    },
                    "withheld": [],
                    "evidence_source": "verified_feed",
                    "disclosure_boundary_acknowledged": True,
                },
                {
                    "provided": {
                        "bank_feed_12m": "<aggregated verified feed>",
                        "tax_filing_latest": "<hmrc confirmation>",
                    },
                    "withheld": ["client_names"],
                    "evidence_source": "verified_feed",
                    "disclosure_boundary_acknowledged": True,
                },
            ],
        ),
        run_episode(
            "missing_evidence_escalation",
            request,
            [
                {
                    "provided": {
                        "bank_feed_12m": "<csv export>",
                    },
                    "withheld": ["tax_filing_latest"],
                    "evidence_source": "manual_upload",
                    "disclosure_boundary_acknowledged": True,
                }
            ],
        ),
    ]

    total_rejections = sum(
        1 for episode in episodes for event in episode["events"] if event["type"] == "reject_response"
    )
    forwarded_boundary_violations = 0
    for episode in episodes:
        for event in episode["events"]:
            if event["type"] == "forward_response" and "client_names" in event["forwarded_keys"]:
                forwarded_boundary_violations += 1

    summary = {
        "episodes": len(episodes),
        "conformant_after_mediation": sum(1 for e in episodes if e["final_status"] == "conformant_exchange"),
        "escalations_required": sum(1 for e in episodes if e["final_status"] == "escalation_required"),
        "rejected_nonconformant_messages": total_rejections,
        "forwarded_boundary_violations": forwarded_boundary_violations,
    }

    return {
        "summary": summary,
        "episodes": episodes,
    }


def write_markdown(result: dict, path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Structured Disclosure Protocol Demo",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- Conformant after mediation: `{summary['conformant_after_mediation']}`",
        f"- Escalations required: `{summary['escalations_required']}`",
        f"- Rejected non-conformant messages: `{summary['rejected_nonconformant_messages']}`",
        f"- Forwarded boundary violations: `{summary['forwarded_boundary_violations']}`",
        "",
    ]
    for episode in result["episodes"]:
        lines.append(f"## {episode['episode_id']}")
        lines.append(f"- Final status: `{episode['final_status']}`")
        for event in episode["events"]:
            lines.append(f"- {event['actor']}::{event['type']} {json.dumps(event, sort_keys=True)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = build_demo()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "experiment3_structured_protocol_demo.json"
    md_path = OUTPUT_DIR / "experiment3_structured_protocol_demo.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
