"""
Procedural borrower generation + hybrid pool management for season mode.

Generates weekly cohorts using a mix of static borrowers (from data.py)
and procedurally generated ones from templates.
"""

import random

from .data import (
    MONTHS,
    SECTORS,
    build_quarterly_income,
    generate_statements,
    get_borrowers,
)
from .models import (
    Borrower,
    FinancialDossier,
    SeasonConfig,
)

# ---------------------------------------------------------------------------
# Season mixes: good/bad/fraud fractions
# ---------------------------------------------------------------------------

SEASON_MIX = {
    "gentle":      {"good": 0.95, "bad": 0.05, "fraud": 0.00},
    "realistic":   {"good": 0.85, "bad": 0.10, "fraud": 0.05},
    "adversarial": {"good": 0.55, "bad": 0.30, "fraud": 0.15},
    "stress":      {"good": 0.40, "bad": 0.35, "fraud": 0.25},
}

ESCALATING_MIXES = [
    # (fraction_of_season, mix)  — boundaries computed from config.weeks
    (0.30, {"good": 0.90, "bad": 0.10, "fraud": 0.00}),
    (0.60, {"good": 0.80, "bad": 0.15, "fraud": 0.05}),
    (1.00, {"good": 0.60, "bad": 0.25, "fraud": 0.15}),
]


def _get_week_mix(week: int, season_mix: str, total_weeks: int = 10) -> dict[str, float]:
    if season_mix == "escalating":
        for cutoff_frac, mix in ESCALATING_MIXES:
            if week <= max(1, round(total_weeks * cutoff_frac)):
                return mix
        return ESCALATING_MIXES[-1][1]
    return SEASON_MIX.get(season_mix, SEASON_MIX["realistic"])


# ---------------------------------------------------------------------------
# Name / narrative generation helpers
# ---------------------------------------------------------------------------

_COMPANY_PREFIXES = [
    "Apex", "Summit", "Meridian", "Nova", "Atlas", "Pinnacle", "Vanguard",
    "Zenith", "Catalyst", "Forge", "Nexus", "Prime", "Sterling", "Crest",
    "Horizon", "Vertex", "Ember", "Iron", "Pacific", "Azure",
]

_COMPANY_SUFFIXES = [
    "Solutions", "Systems", "Labs", "Industries", "Works", "Corp",
    "Technologies", "Services", "Group", "Partners", "Dynamics",
    "Innovations", "Enterprises", "Holdings", "Collective",
]

_CUSTOMER_NAMES = [
    "GlobalTech Inc", "Unified Supply Co", "Metro Distributors", "Coastal Trading LLC",
    "Summit Sourcing", "Bridgewater Corp", "Eastline Partners", "Northwind Logistics",
    "SilverOak Industries", "RedRiver Supply", "Bluesky Ventures", "Irongate Holdings",
    "Pacifica Group", "Harborview LLC", "Ridgeline Services",
]

_VENDOR_NAMES = [
    "RawMat Supply", "TechParts Wholesale", "IndustrialBase Co", "QuickShip Logistics",
    "FuelCore Supply", "MaintPro Services", "EquipLease Inc", "PackagePrime",
]

_GOOD_NARRATIVES = [
    "{name} is a {years}-year-old {sector} company with steady revenue growth and "
    "healthy margins. The business has {employees} employees and a diversified client "
    "base. Management is seeking ${amount:,.0f} to {purpose}.",
    "{name} has operated in the {sector} space for {years} years, building a reliable "
    "revenue stream. With consistent profitability and {employees} staff, the company "
    "is requesting ${amount:,.0f} for {purpose}.",
]

_BAD_NARRATIVES = [
    "{name} operates in {sector} and has been in business for {years} years. "
    "The company has {employees} employees and reports annual revenue of ${revenue:,.0f}. "
    "Management is requesting ${amount:,.0f} to {purpose}. The company notes some recent "
    "challenges but believes growth will resume.",
    "{name} is a {years}-year-old {sector} business seeking ${amount:,.0f} for {purpose}. "
    "While the company has experienced some revenue pressure, management cites a strong "
    "pipeline of new contracts expected to materialize.",
]

_FRAUD_NARRATIVES = [
    "{name} is a rapidly growing {sector} company founded {years} years ago. "
    "With {employees} employees and impressive revenue of ${revenue:,.0f}, the company "
    "is seeking ${amount:,.0f} to {purpose}. Management emphasizes their fast growth trajectory.",
    "{name} operates in {sector} and has shown remarkable growth over {years} years. "
    "The company reports ${revenue:,.0f} in annual revenue with {employees} staff and "
    "is requesting ${amount:,.0f} for {purpose}.",
]

_GOOD_PURPOSES = [
    "expand production capacity", "acquire new equipment",
    "open a second location", "hire additional staff for growth",
    "upgrade technology infrastructure", "fund inventory for peak season",
]

_BAD_PURPOSES = [
    "refinance existing obligations", "bridge a temporary cash shortfall",
    "fund operational improvements", "consolidate vendor payables",
    "invest in turnaround initiatives",
]

_FRAUD_PURPOSES = [
    "fund rapid expansion", "acquire a competitor",
    "invest in new technology platform", "capitalize on market opportunity",
]

# Fraud deposit type labels
_FRAUD_TYPES = ["round_numbers", "circular", "fabricated", "structured"]

# ---------------------------------------------------------------------------
# Procedural borrower generation
# ---------------------------------------------------------------------------

def _generate_company_name(rng: random.Random) -> str:
    return f"{rng.choice(_COMPANY_PREFIXES)} {rng.choice(_COMPANY_SUFFIXES)}"


def _generate_good_borrower(borrower_id: str, rng: random.Random, sector: str) -> Borrower:
    """Generate a good borrower with stable/growing financials."""
    base_revenue = rng.uniform(80000, 250000)
    growth = rng.uniform(1.01, 1.04)  # 1-4% monthly growth
    margin = rng.uniform(0.22, 0.40)

    monthly = []
    rev = base_revenue
    for _ in range(12):
        exp = rev * (1.0 - margin) + rng.uniform(-2000, 2000)
        monthly.append((round(rev, 2), round(max(exp, rev * 0.5), 2)))
        rev *= growth

    annual_revenue = sum(m[0] for m in monthly)
    annual_expenses = sum(m[1] for m in monthly)
    net_income = annual_revenue - annual_expenses
    employees = rng.randint(10, 80)
    years = rng.randint(3, 15)
    name = _unique_company_name(rng)
    loan_amount = round(rng.uniform(150000, 600000) / 10000) * 10000
    purpose = rng.choice(_GOOD_PURPOSES)
    opening_balance = round(rng.uniform(50000, 200000), 2)

    customers = rng.sample(_CUSTOMER_NAMES, min(4, len(_CUSTOMER_NAMES)))
    vendors = rng.sample(_VENDOR_NAMES, min(3, len(_VENDOR_NAMES)))

    narrative = rng.choice(_GOOD_NARRATIVES).format(
        name=name, years=years, sector=sector, employees=employees,
        amount=loan_amount, purpose=purpose, revenue=annual_revenue,
    )

    seed = rng.randint(10000, 99999)
    return Borrower(
        id=borrower_id,
        dossier=FinancialDossier(
            company_name=name,
            sector=sector,
            years_in_business=years,
            annual_revenue=round(annual_revenue, 2),
            annual_expenses=round(annual_expenses, 2),
            net_income=round(net_income, 2),
            employee_count=employees,
            bank_statements=generate_statements(
                monthly, opening_balance, customers, vendors, seed=seed,
            ),
            quarterly_income=build_quarterly_income(monthly),
            narrative=narrative,
            loan_request_amount=loan_amount,
            loan_purpose=purpose,
        ),
        true_outcome="good",
    )


def _generate_bad_borrower(borrower_id: str, rng: random.Random, sector: str) -> Borrower:
    """Generate a bad borrower with declining financials or thin margins."""
    base_revenue = rng.uniform(60000, 200000)
    # Declining or flat revenue
    growth = rng.uniform(0.95, 1.01)
    margin = rng.uniform(0.05, 0.18)  # thin margins

    monthly = []
    rev = base_revenue
    for _ in range(12):
        exp = rev * (1.0 - margin) + rng.uniform(-1000, 3000)
        monthly.append((round(rev, 2), round(min(exp, rev * 1.1), 2)))
        rev *= growth

    annual_revenue = sum(m[0] for m in monthly)
    annual_expenses = sum(m[1] for m in monthly)
    net_income = annual_revenue - annual_expenses
    employees = rng.randint(5, 50)
    years = rng.randint(2, 10)
    name = _unique_company_name(rng)
    loan_amount = round(rng.uniform(100000, 500000) / 10000) * 10000
    purpose = rng.choice(_BAD_PURPOSES)
    opening_balance = round(rng.uniform(20000, 80000), 2)
    months_before_default = rng.randint(3, 12)

    customers = rng.sample(_CUSTOMER_NAMES, min(4, len(_CUSTOMER_NAMES)))
    vendors = rng.sample(_VENDOR_NAMES, min(3, len(_VENDOR_NAMES)))

    narrative = rng.choice(_BAD_NARRATIVES).format(
        name=name, years=years, sector=sector, employees=employees,
        amount=loan_amount, purpose=purpose, revenue=annual_revenue,
    )

    seed = rng.randint(10000, 99999)
    return Borrower(
        id=borrower_id,
        dossier=FinancialDossier(
            company_name=name,
            sector=sector,
            years_in_business=years,
            annual_revenue=round(annual_revenue, 2),
            annual_expenses=round(annual_expenses, 2),
            net_income=round(net_income, 2),
            employee_count=employees,
            bank_statements=generate_statements(
                monthly, opening_balance, customers, vendors, seed=seed,
            ),
            quarterly_income=build_quarterly_income(monthly),
            narrative=narrative,
            loan_request_amount=loan_amount,
            loan_purpose=purpose,
        ),
        true_outcome="bad",
        months_before_default=months_before_default,
    )


def _generate_fraud_borrower(borrower_id: str, rng: random.Random, sector: str) -> Borrower:
    """Generate a fraudulent borrower with suspicious financial patterns."""
    # Inflated revenue to make the business look bigger than it is
    base_revenue = rng.uniform(100000, 300000)
    growth = rng.uniform(1.02, 1.06)
    margin = rng.uniform(0.25, 0.45)  # suspiciously healthy

    monthly = []
    rev = base_revenue
    for _ in range(12):
        exp = rev * (1.0 - margin) + rng.uniform(-500, 500)
        monthly.append((round(rev, 2), round(max(exp, rev * 0.4), 2)))
        rev *= growth

    annual_revenue = sum(m[0] for m in monthly)
    annual_expenses = sum(m[1] for m in monthly)
    net_income = annual_revenue - annual_expenses
    employees = rng.randint(8, 40)
    years = rng.randint(1, 5)
    name = _unique_company_name(rng)
    loan_amount = round(rng.uniform(200000, 700000) / 10000) * 10000
    purpose = rng.choice(_FRAUD_PURPOSES)
    opening_balance = round(rng.uniform(30000, 150000), 2)
    fraud_type = rng.choice(_FRAUD_TYPES)

    customers = rng.sample(_CUSTOMER_NAMES, min(4, len(_CUSTOMER_NAMES)))
    vendors = rng.sample(_VENDOR_NAMES, min(3, len(_VENDOR_NAMES)))

    narrative = rng.choice(_FRAUD_NARRATIVES).format(
        name=name, years=years, sector=sector, employees=employees,
        amount=loan_amount, purpose=purpose, revenue=annual_revenue,
    )

    seed = rng.randint(10000, 99999)
    return Borrower(
        id=borrower_id,
        dossier=FinancialDossier(
            company_name=name,
            sector=sector,
            years_in_business=years,
            annual_revenue=round(annual_revenue, 2),
            annual_expenses=round(annual_expenses, 2),
            net_income=round(net_income, 2),
            employee_count=employees,
            bank_statements=generate_statements(
                monthly, opening_balance, customers, vendors,
                seed=seed, fraud_type=fraud_type,
            ),
            quarterly_income=build_quarterly_income(monthly),
            narrative=narrative,
            loan_request_amount=loan_amount,
            loan_purpose=purpose,
        ),
        true_outcome="fraud",
    )


# ---------------------------------------------------------------------------
# Mix allocation helper
# ---------------------------------------------------------------------------

def _split_counts(total: int, mix: dict[str, float]) -> tuple[int, int, int]:
    """Split total into (n_good, n_bad, n_fraud) respecting mix fractions.

    Uses largest-remainder method so counts always sum to total and every
    non-zero fraction gets at least 1 when total is large enough.
    """
    raw_good = total * mix["good"]
    raw_bad = total * mix["bad"]
    raw_fraud = total * mix["fraud"]

    n_good = int(raw_good)
    n_bad = int(raw_bad)
    n_fraud = int(raw_fraud)

    # Distribute remainders to the categories with largest fractional parts
    remainders = [
        (raw_good - n_good, "good"),
        (raw_bad - n_bad, "bad"),
        (raw_fraud - n_fraud, "fraud"),
    ]
    remainders.sort(key=lambda x: -x[0])

    shortfall = total - (n_good + n_bad + n_fraud)
    for _, category in remainders:
        if shortfall <= 0:
            break
        if category == "good":
            n_good += 1
        elif category == "bad":
            n_bad += 1
        else:
            n_fraud += 1
        shortfall -= 1

    return n_good, n_bad, n_fraud


# ---------------------------------------------------------------------------
# Static pool cache (drawn from data.py)
# ---------------------------------------------------------------------------

_static_pool_cache: list[Borrower] | None = None


def _get_static_pool() -> list[Borrower]:
    """Get all static borrowers from data.py (cached)."""
    global _static_pool_cache
    if _static_pool_cache is None:
        _static_pool_cache = get_borrowers("all")
    return list(_static_pool_cache)


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def generate_cohort(
    week: int,
    config: SeasonConfig,
    used_static_ids: set[str] | None = None,
) -> list[Borrower]:
    """Generate a week's cohort. Deterministic via seed.

    Hybrid strategy:
    - Weeks 1-2: draw from static pool (existing borrowers) shuffled by seed
    - Weeks 3+: procedurally generated from templates
    - Static borrowers not reused across weeks (tracked by used_static_ids)

    Returns a list of Borrower objects for the week.
    """
    if used_static_ids is None:
        used_static_ids = set()

    rng = random.Random(config.seed * 1000 + week)
    mix = _get_week_mix(week, config.season_mix, total_weeks=config.weeks)
    n_good, n_bad, n_fraud = _split_counts(config.cohort_size, mix)

    cohort: list[Borrower] = []

    if week <= 2:
        # Draw from static pool
        pool = _get_static_pool()
        rng.shuffle(pool)
        available = [b for b in pool if b.id not in used_static_ids]

        got = {"good": 0, "bad": 0, "fraud": 0}
        for outcome, count in [("good", n_good), ("bad", n_bad), ("fraud", n_fraud)]:
            matching = [b for b in available if b.true_outcome == outcome]
            drawn = matching[:count]
            cohort.extend(drawn)
            got[outcome] = len(drawn)
            for b in drawn:
                used_static_ids.add(b.id)
                available.remove(b)

        # Fill specific missing outcomes with procedural borrowers
        need_good = n_good - got["good"]
        need_bad = n_bad - got["bad"]
        need_fraud = n_fraud - got["fraud"]
        if need_good + need_bad + need_fraud > 0:
            cohort.extend(_generate_exact(
                need_good, need_bad, need_fraud, week, len(cohort), rng,
            ))
    else:
        # Fully procedural
        cohort = _generate_exact(n_good, n_bad, n_fraud, week, 0, rng)

    return cohort


_used_names: set[str] = set()


def _unique_company_name(rng: random.Random) -> str:
    """Generate a company name that hasn't been used this session."""
    for _ in range(50):
        name = _generate_company_name(rng)
        if name not in _used_names:
            _used_names.add(name)
            return name
    # Fallback: append a number
    name = _generate_company_name(rng)
    suffix = rng.randint(100, 999)
    unique = f"{name} {suffix}"
    _used_names.add(unique)
    return unique


def _generate_exact(
    n_good: int,
    n_bad: int,
    n_fraud: int,
    week: int,
    offset: int,
    rng: random.Random,
) -> list[Borrower]:
    """Generate exactly n_good + n_bad + n_fraud procedural borrowers."""
    borrowers: list[Borrower] = []
    idx = offset

    for i in range(n_good):
        bid = f"BRW-S{week:02d}-{idx + i + 1:03d}"
        sector = rng.choice(SECTORS)
        borrowers.append(_generate_good_borrower(bid, rng, sector))

    idx += n_good
    for i in range(n_bad):
        bid = f"BRW-S{week:02d}-{idx + i + 1:03d}"
        sector = rng.choice(SECTORS)
        borrowers.append(_generate_bad_borrower(bid, rng, sector))

    idx += n_bad
    for i in range(n_fraud):
        bid = f"BRW-S{week:02d}-{idx + i + 1:03d}"
        sector = rng.choice(SECTORS)
        borrowers.append(_generate_fraud_borrower(bid, rng, sector))

    # Shuffle so outcome order isn't predictable
    rng.shuffle(borrowers)
    return borrowers
