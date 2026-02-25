"""
Generate population tensors for the AgentTorch lending simulation.

Converts Loanville2's borrower generation (both static and procedural) into
PyTorch tensors suitable for AgentTorch's state initialization.

Two modes:
  1. from_loanville() — use Loanville2's 36 static borrowers (small scale, validation)
  2. generate_population() — procedural generation at arbitrary scale (10K-1M+)
"""

import sys
import os
import random
from pathlib import Path

import torch

# Add project root to path for loanville imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from loanville.data import SECTORS, get_borrowers
from loanville.borrower_gen import (
    _generate_good_borrower,
    _generate_bad_borrower,
    _generate_fraud_borrower,
    _unique_company_name,
    SEASON_MIX,
)

# Sector name -> index mapping
SECTOR_INDEX = {name: i for i, name in enumerate(SECTORS)}


def _borrower_to_tensors(borrower) -> dict:
    """Extract tensor-friendly fields from a Loanville2 Borrower object."""
    d = borrower.dossier
    sector_idx = SECTOR_INDEX.get(d.sector, 0)
    net_margin = d.net_income / d.annual_revenue if d.annual_revenue > 0 else 0.0

    # Map outcome to probabilities
    if borrower.true_outcome == "good":
        default_prob = 0.0
        fraud_prob = 0.0
        months_to_default = 24.0  # won't trigger (no default_prob)
    elif borrower.true_outcome == "bad":
        default_prob = 1.0
        fraud_prob = 0.0
        months_to_default = float(borrower.months_before_default or 8)
    else:  # fraud
        default_prob = 1.0
        fraud_prob = 1.0
        months_to_default = 0.0  # immediate

    return {
        "sector": sector_idx,
        "annual_revenue": d.annual_revenue,
        "net_income": d.net_income,
        "net_margin": net_margin,
        "loan_request": d.loan_request_amount,
        "years_in_business": float(d.years_in_business),
        "default_prob": default_prob,
        "months_to_default": months_to_default,
        "fraud_prob": fraud_prob,
    }


def from_loanville(mix: str = "all") -> dict[str, torch.Tensor]:
    """Create population tensors from Loanville2's static borrower pool.

    Returns a dict of {property_name: tensor} for all borrower properties.
    Useful for validation against the original Loanville2 simulation.
    """
    borrowers = get_borrowers(mix)
    n = len(borrowers)

    # Collect per-borrower data
    records = [_borrower_to_tensors(b) for b in borrowers]

    return _records_to_tensors(records, n)


def generate_population(
    num_borrowers: int = 10000,
    mix: str = "realistic",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Procedurally generate a large borrower population.

    Uses Loanville2's borrower generation templates scaled to arbitrary size.
    Mix ratios control the good/bad/fraud distribution.

    Args:
        num_borrowers: Total number of borrower agents.
        mix: Distribution preset — "gentle", "realistic", or "adversarial".
        seed: Random seed for reproducibility.

    Returns:
        Dict of {property_name: tensor} for all borrower properties.
    """
    rng = random.Random(seed)
    mix_ratios = SEASON_MIX.get(mix, SEASON_MIX["realistic"])

    # Compute counts using largest-remainder method
    n_good = int(num_borrowers * mix_ratios["good"])
    n_bad = int(num_borrowers * mix_ratios["bad"])
    n_fraud = num_borrowers - n_good - n_bad  # remainder to fraud

    records = []

    for i in range(n_good):
        bid = f"AT-G-{i:06d}"
        sector = rng.choice(SECTORS)
        b = _generate_good_borrower(bid, rng, sector)
        records.append(_borrower_to_tensors(b))

    for i in range(n_bad):
        bid = f"AT-B-{i:06d}"
        sector = rng.choice(SECTORS)
        b = _generate_bad_borrower(bid, rng, sector)
        records.append(_borrower_to_tensors(b))

    for i in range(n_fraud):
        bid = f"AT-F-{i:06d}"
        sector = rng.choice(SECTORS)
        b = _generate_fraud_borrower(bid, rng, sector)
        records.append(_borrower_to_tensors(b))

    # Shuffle so outcome order isn't predictable
    rng.shuffle(records)

    return _records_to_tensors(records, num_borrowers)


def generate_lender_tensors(
    num_lenders: int = 10,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Generate lender population tensors with diversified profiles.

    Creates lenders with varying capital levels, risk tolerances, and target yields
    to simulate a realistic competitive lending market.
    """
    rng = random.Random(seed)

    # Lender archetype distributions
    archetypes = [
        # (name_prefix, capital_range, target_yield_range, risk_tolerance_range, max_loan_frac)
        ("community_bank", (1_500_000, 3_000_000), (8.0, 10.0), (0.3, 0.5), 0.15),
        ("regional_bank", (3_000_000, 5_000_000), (9.0, 12.0), (0.4, 0.6), 0.18),
        ("national_bank", (5_000_000, 10_000_000), (7.0, 9.0), (0.3, 0.45), 0.12),
        ("fintech", (2_000_000, 4_000_000), (12.0, 16.0), (0.6, 0.8), 0.25),
        ("credit_union", (1_000_000, 2_500_000), (7.0, 9.0), (0.25, 0.4), 0.20),
    ]

    capitals = []
    target_yields = []
    risk_tolerances = []
    max_single_loans = []

    for i in range(num_lenders):
        archetype = archetypes[i % len(archetypes)]
        _, cap_range, yield_range, risk_range, max_frac = archetype

        cap = rng.uniform(*cap_range)
        capitals.append(cap)
        target_yields.append(rng.uniform(*yield_range))
        risk_tolerances.append(rng.uniform(*risk_range))
        max_single_loans.append(cap * max_frac)

    return {
        "total_capital": torch.tensor(capitals, dtype=torch.float32).unsqueeze(1),
        "available_capital": torch.tensor(capitals, dtype=torch.float32).unsqueeze(1),
        "target_yield": torch.tensor(target_yields, dtype=torch.float32).unsqueeze(1),
        "risk_tolerance": torch.tensor(risk_tolerances, dtype=torch.float32).unsqueeze(1),
        "max_single_loan": torch.tensor(max_single_loans, dtype=torch.float32).unsqueeze(1),
    }


def _records_to_tensors(records: list[dict], n: int) -> dict[str, torch.Tensor]:
    """Convert a list of per-borrower dicts to a dict of tensors."""
    fields = [
        "sector", "annual_revenue", "net_income", "net_margin",
        "loan_request", "years_in_business",
        "default_prob", "months_to_default", "fraud_prob",
    ]

    result = {}
    for field in fields:
        values = [r[field] for r in records]
        dtype = torch.long if field == "sector" else torch.float32
        result[field] = torch.tensor(values, dtype=dtype).unsqueeze(1)

    return result
