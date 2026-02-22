"""
Benchmark Items — gold-labeled cases that the evaluation harness scores against.

Each benchmark item is a small bundle:
  - case definition (borrower data, docs)
  - gold label (correct action, terms ranges, required conditions)
  - rubric (what the trace must include, penalty schedule)

Benchmark items are generated from:
  1. Hand-crafted expert cases (initial seed set)
  2. Disagreement mining (flywheel Step D)
  3. Simulator counterfactuals

The benchmark evolves from static Q&A into a living evaluation suite
as the LOS traces reveal new edge cases and label ambiguities.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .models import Borrower
from .run_schema import RunCase, UnderwritingRun


@dataclass
class GoldLabel:
    """The correct answer for a benchmark case."""
    action: str  # "approve" | "decline" | "counter" | "refer"
    risk_grade: str = ""
    terms: dict = field(default_factory=dict)  # {max_amount, apr_range: [lo,hi], tenor_months}
    required_conditions: list[str] = field(default_factory=list)
    required_covenants: list[str] = field(default_factory=list)
    true_outcome: str = ""  # "good" | "bad" | "fraud"
    months_before_default: Optional[int] = None
    # Optional: probability-based labels for cases where binary is too simplistic
    prob_default_12m: Optional[float] = None


@dataclass
class Rubric:
    """What the underwriting trace must include and penalty schedule."""
    must_include: list[str] = field(default_factory=list)  # e.g. ["dscr_calc", "cashflow_comment"]
    penalties: dict = field(default_factory=dict)  # {missing_required_doc: 10, math_error: 25}
    bonus: dict = field(default_factory=dict)  # {identified_fraud_pattern: 15}


@dataclass
class BenchmarkItem:
    """A single benchmark case with gold label and rubric."""
    case_id: str
    segment: str  # e.g. "smb_term_loan"
    category: str  # "clean" | "messy" | "missing_docs" | "borderline_dscr" | "sector_quirk" | "fraud" | "over_leverage"
    docs: list[str] = field(default_factory=list)  # doc references
    task: str = "Decide approve/decline/counter and propose terms."
    gold: GoldLabel = field(default_factory=lambda: GoldLabel(action="decline"))
    rubric: Rubric = field(default_factory=Rubric)
    source: str = "manual"  # "manual" | "mined" | "simulator"
    difficulty: str = "medium"  # "easy" | "medium" | "hard"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_gold_dict(self) -> dict:
        """Convert gold label to the format expected by scorecard."""
        d = {
            "correct_action": self.gold.action,
            "true_outcome": self.gold.true_outcome,
        }
        if self.gold.terms:
            d["terms"] = self.gold.terms
        if self.gold.months_before_default is not None:
            d["months_before_default"] = self.gold.months_before_default
        if self.gold.prob_default_12m is not None:
            d["prob_default_actual"] = self.gold.prob_default_12m
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkItem:
        item = cls(
            case_id=data["case_id"],
            segment=data.get("segment", "smb_term_loan"),
            category=data.get("category", "unknown"),
            docs=data.get("docs", []),
            task=data.get("task", "Decide approve/decline/counter and propose terms."),
            source=data.get("source", "manual"),
            difficulty=data.get("difficulty", "medium"),
        )
        if "gold" in data:
            g = data["gold"]
            item.gold = GoldLabel(
                action=g.get("action", "decline"),
                risk_grade=g.get("risk_grade", ""),
                terms=g.get("terms", {}),
                required_conditions=g.get("required_conditions", []),
                required_covenants=g.get("required_covenants", []),
                true_outcome=g.get("true_outcome", ""),
                months_before_default=g.get("months_before_default"),
                prob_default_12m=g.get("prob_default_12m"),
            )
        if "rubric" in data:
            r = data["rubric"]
            item.rubric = Rubric(
                must_include=r.get("must_include", []),
                penalties=r.get("penalties", {}),
                bonus=r.get("bonus", {}),
            )
        return item


# ---------------------------------------------------------------------------
# Benchmark Suite — a collection of items
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSuite:
    """A versioned collection of benchmark items."""
    suite_id: str
    version: str = "1.0"
    items: list[BenchmarkItem] = field(default_factory=list)

    def save(self, path: str) -> None:
        data = {
            "suite_id": self.suite_id,
            "version": self.version,
            "n_items": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> BenchmarkSuite:
        data = json.loads(Path(path).read_text())
        suite = cls(
            suite_id=data["suite_id"],
            version=data.get("version", "1.0"),
        )
        suite.items = [BenchmarkItem.from_dict(item) for item in data.get("items", [])]
        return suite

    def by_category(self) -> dict[str, list[BenchmarkItem]]:
        """Group items by category."""
        groups: dict[str, list[BenchmarkItem]] = {}
        for item in self.items:
            groups.setdefault(item.category, []).append(item)
        return groups

    def print_summary(self) -> None:
        cats = self.by_category()
        print(f"\n  Benchmark Suite: {self.suite_id} v{self.version}")
        print(f"  Total items: {len(self.items)}")
        for cat, items in sorted(cats.items()):
            approve = sum(1 for i in items if i.gold.action == "approve")
            decline = len(items) - approve
            print(f"    {cat}: {len(items)} ({approve} approve, {decline} decline)")


# ---------------------------------------------------------------------------
# Generate seed benchmark from existing Loanville borrowers
# ---------------------------------------------------------------------------

def generate_seed_benchmark(borrowers: list[Borrower]) -> BenchmarkSuite:
    """Generate the initial benchmark suite from the existing borrower dataset.

    This produces the Phase 2 "minimum benchmark": high-signal cases
    that cover the key failure modes.
    """
    items = []

    for b in borrowers:
        d = b.dossier

        # Determine category
        if b.true_outcome == "fraud":
            category = "fraud"
        elif b.true_outcome == "bad":
            # Check if it's over-leverage vs fundamental
            debt_service = d.loan_request_amount / 24 * 12  # annual proxy
            dscr = d.net_income / debt_service if debt_service > 0 else 0
            if dscr < 1.0:
                category = "over_leverage"
            else:
                category = "fundamental_weakness"
        else:
            # Good borrower — categorize by cleanness
            margin = d.net_income / d.annual_revenue if d.annual_revenue > 0 else 0
            if margin > 0.15:
                category = "clean"
            else:
                category = "borderline_dscr"

        # Determine correct action
        if b.true_outcome in ("bad", "fraud"):
            correct_action = "decline"
        else:
            correct_action = "approve"

        # Build gold terms for approvals
        terms = {}
        if correct_action == "approve":
            terms = {
                "max_amount": d.loan_request_amount,
                "apr_range": [8.0, 18.0],  # reasonable range
                "tenor_months": 24,
            }

        # Build rubric
        must_include = ["dscr_calc"]
        penalties = {"math_error": 25, "missing_required_doc": 10}
        bonus = {}

        if b.true_outcome == "fraud":
            must_include.append("fraud_flag")
            bonus["identified_fraud_pattern"] = 15
        if b.true_outcome == "bad":
            must_include.append("risk_factor_identification")

        item = BenchmarkItem(
            case_id=b.id,
            segment=f"{d.sector.lower().replace(' ', '_').replace('-', '_')}_term_loan",
            category=category,
            docs=["quarterly_income_4q", "bank_statements_12m"],
            gold=GoldLabel(
                action=correct_action,
                true_outcome=b.true_outcome,
                terms=terms,
                months_before_default=b.months_before_default,
            ),
            rubric=Rubric(
                must_include=must_include,
                penalties=penalties,
                bonus=bonus,
            ),
            source="seed",
            difficulty="easy" if b.true_outcome == "good" and category == "clean" else "medium",
        )
        items.append(item)

    suite = BenchmarkSuite(
        suite_id="loanville_seed_v1",
        version="1.0",
        items=items,
    )
    return suite
