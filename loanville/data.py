"""
Hard-coded borrower dataset and lender configurations.

Contains 12 businesses: 5 good, 4 bad, 3 fraudulent.
Contains 3 lender personas with distinct risk profiles.
"""

import random
from .models import (
    Borrower,
    ExistingLoan,
    FinancialDossier,
    LenderConfig,
    MonthlyStatement,
    QuarterlyIncome,
    Transaction,
)

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

SECTORS = [
    "Aero-Logistics",
    "Bio-Synthetics",
    "Quantum Computing",
    "Green Energy",
    "Urban Agriculture",
    "Digital Media",
]


# ---------------------------------------------------------------------------
# Statement generation helpers
# ---------------------------------------------------------------------------

def _split_amount(total: float, labels: list[str], rng: random.Random) -> list[tuple[str, float]]:
    """Split a total amount among labeled items with random proportions."""
    n = len(labels)
    weights = [rng.random() + 0.1 for _ in range(n)]
    weight_sum = sum(weights)
    parts = []
    running = 0.0
    for i, label in enumerate(labels):
        if i == n - 1:
            amt = round(total - running, 2)
        else:
            amt = round(total * weights[i] / weight_sum, 2)
        running += amt
        parts.append((label, amt))
    return parts


def _generate_statements(
    monthly_figures: list[tuple[float, float]],
    opening_balance: float,
    customers: list[str],
    vendors: list[str],
    seed: int,
    fraud_type: str | None = None,
) -> list[MonthlyStatement]:
    """Generate 12 monthly bank statements from summary figures."""
    rng = random.Random(seed)
    statements = []
    balance = opening_balance

    for i, (dep_total, wd_total) in enumerate(monthly_figures):
        opening = round(balance, 2)

        # Generate deposit transactions
        if fraud_type == "round_numbers":
            # Fraud: all deposits are suspiciously round
            dep_parts = _make_round_deposits(dep_total, customers, rng)
        elif fraud_type == "circular":
            dep_parts = _make_circular_deposits(dep_total, customers, rng, i)
        elif fraud_type == "fabricated":
            dep_parts = _make_fabricated_deposits(dep_total, customers, rng)
        elif fraud_type == "structured":
            dep_parts = _make_structured_deposits(dep_total, customers, rng)
        else:
            dep_parts = _split_amount(dep_total, customers, rng)

        deposits = []
        for label, amt in dep_parts:
            day = rng.randint(1, 28)
            deposits.append(Transaction(
                date=f"{i+1:02d}/{day:02d}/2025",
                description=label,
                amount=round(amt, 2),
            ))

        # Generate withdrawal transactions
        standard_expenses = ["Payroll", "Rent/Lease", "Utilities", "Insurance"]
        wd_labels = vendors + standard_expenses
        wd_parts = _split_amount(wd_total, wd_labels, rng)
        withdrawals = []
        for label, amt in wd_parts:
            day = rng.randint(1, 28)
            withdrawals.append(Transaction(
                date=f"{i+1:02d}/{day:02d}/2025",
                description=label,
                amount=round(amt, 2),
            ))

        balance = opening + dep_total - wd_total
        statements.append(MonthlyStatement(
            month=f"{MONTHS[i]} 2025",
            opening_balance=opening,
            deposits=deposits,
            withdrawals=withdrawals,
            ending_balance=round(balance, 2),
        ))

    return statements


def _make_round_deposits(total, customers, rng):
    """Fraud: deposits are suspiciously round numbers."""
    round_amounts = [50000, 75000, 100000, 25000, 150000, 200000]
    parts = []
    remaining = total
    for i, c in enumerate(customers):
        if i == len(customers) - 1:
            amt = remaining
        else:
            amt = rng.choice(round_amounts)
            amt = min(amt, remaining - 25000 * (len(customers) - i - 1))
        parts.append((c, round(amt, 2)))
        remaining -= amt
    return parts


def _make_circular_deposits(total, customers, rng, month_idx):
    """Fraud: circular transfers with a related entity."""
    real_portion = total * 0.3
    circular_portion = total - real_portion
    parts = _split_amount(real_portion, customers[:2], rng)
    parts.append(("Transfer from BioGenesis Holdings LLC", round(circular_portion * 0.6, 2)))
    parts.append(("Transfer from BGH Capital Partners", round(circular_portion * 0.4, 2)))
    return parts


def _make_fabricated_deposits(total, customers, rng):
    """Fraud: deposits are nearly identical every month (unnaturally consistent)."""
    n = len(customers)
    base = total / n
    parts = []
    for i, c in enumerate(customers):
        # Tiny variation to look slightly different but suspiciously consistent
        variation = rng.uniform(-200, 200)
        parts.append((c, round(base + variation, 2)))
    return parts


def _make_structured_deposits(total, customers, rng):
    """Fraud: deposits split into many small amounts just under $10,000.

    This mimics 'structuring' (aka 'smurfing') — breaking large sums into
    sub-$10K transactions to avoid Currency Transaction Report thresholds.
    A legitimate business with $130K/month revenue would have a handful of
    large client payments, not 15+ tiny deposits.
    """
    parts = []
    remaining = total
    while remaining > 0:
        customer = rng.choice(customers)
        if remaining < 5000:
            parts.append((customer, round(remaining, 2)))
            remaining = 0
        else:
            # Amount between $7,000 and $9,950, always under $10,000
            amt = round(rng.uniform(7000, 9950), 2)
            amt = min(amt, remaining)
            parts.append((customer, round(amt, 2)))
            remaining -= amt
    return parts


def _build_quarterly_income(monthly: list[tuple[float, float]]) -> list[QuarterlyIncome]:
    """Aggregate monthly (deposits, withdrawals) into 4 quarterly income statements."""
    quarters = []
    labels = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
    for q in range(4):
        start = q * 3
        rev = sum(m[0] for m in monthly[start:start + 3])
        exp = sum(m[1] for m in monthly[start:start + 3])
        gp = rev - exp
        gm = (gp / rev * 100) if rev else 0.0
        ni = gp  # simplified: gross profit = net income at this level
        nm = (ni / rev * 100) if rev else 0.0
        quarters.append(QuarterlyIncome(
            quarter=labels[q],
            revenue=round(rev, 2),
            expenses=round(exp, 2),
            gross_profit=round(gp, 2),
            gross_margin_pct=round(gm, 1),
            net_income=round(ni, 2),
            net_margin_pct=round(nm, 1),
        ))
    return quarters


# ---------------------------------------------------------------------------
# Borrower definitions
# ---------------------------------------------------------------------------

def _build_borrowers() -> list[Borrower]:
    borrowers = []

    # ===== GOOD BUSINESSES (5) =====

    # GOOD 1: SkyFreight Solutions - steady growth in cargo logistics
    monthly = [
        (185000, 138000), (188000, 140000), (192000, 141000), (195000, 143000),
        (198000, 145000), (202000, 148000), (205000, 150000), (210000, 152000),
        (212000, 154000), (215000, 155000), (218000, 157000), (222000, 160000),
    ]
    borrowers.append(Borrower(
        id="BRW-001",
        dossier=FinancialDossier(
            company_name="SkyFreight Solutions",
            sector="Aero-Logistics",
            years_in_business=8,
            annual_revenue=2442000,
            annual_expenses=1783000,
            net_income=659000,
            employee_count=45,
            bank_statements=_generate_statements(
                monthly, 180000,
                ["Meridian Airways", "TransGlobal Shipping", "Pacific Routes Inc", "Nordic Air Cargo"],
                ["FuelTech Supply", "AeroMaint Services", "Harbor Logistics"],
                seed=1001,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "SkyFreight Solutions is an established cargo logistics company operating across "
                "the western seaboard. Founded in 2017, the company has grown steadily by serving "
                "mid-market air and ground freight clients. Revenue has grown 8% YoY with a "
                "diversified client base. The company maintains a strong cash position and is "
                "seeking capital to expand its fleet by 3 cargo vehicles to meet growing demand. "
                "Management has 20+ years of combined logistics experience."
            ),
            loan_request_amount=500000,
            loan_purpose="Fleet expansion - acquisition of 3 new cargo vehicles",
        ),
        true_outcome="good",
    ))

    # GOOD 2: NovaBio Labs - growing biotech startup
    monthly = [
        (92000, 67000), (96000, 69000), (100000, 72000), (105000, 74000),
        (108000, 76000), (112000, 78000), (115000, 80000), (118000, 82000),
        (122000, 84000), (125000, 86000), (128000, 88000), (132000, 90000),
    ]
    borrowers.append(Borrower(
        id="BRW-002",
        dossier=FinancialDossier(
            company_name="NovaBio Labs",
            sector="Bio-Synthetics",
            years_in_business=4,
            annual_revenue=1353000,
            annual_expenses=946000,
            net_income=407000,
            employee_count=22,
            bank_statements=_generate_statements(
                monthly, 95000,
                ["MedSupply Corp", "PharmaDist Inc", "ClinicalPath Labs"],
                ["LabEquip Wholesale", "ChemSource Ltd"],
                seed=1002,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "NovaBio Labs develops and manufactures synthetic biological compounds for "
                "pharmaceutical intermediaries. The company has tripled revenue since founding "
                "in 2021 and recently secured two multi-year supply contracts. Cash flow has "
                "been consistently positive with 30%+ net margins. The loan will fund a new "
                "cleanroom production line to fulfill growing contract obligations."
            ),
            loan_request_amount=300000,
            loan_purpose="New cleanroom production line installation",
        ),
        true_outcome="good",
    ))

    # GOOD 3: Helios Solar Works - seasonal but strong green energy
    monthly = [
        (210000, 165000), (215000, 168000), (240000, 175000), (270000, 185000),
        (310000, 200000), (340000, 215000), (350000, 220000), (335000, 210000),
        (290000, 195000), (250000, 180000), (225000, 170000), (218000, 167000),
    ]
    borrowers.append(Borrower(
        id="BRW-003",
        dossier=FinancialDossier(
            company_name="Helios Solar Works",
            sector="Green Energy",
            years_in_business=6,
            annual_revenue=3253000,
            annual_expenses=2350000,
            net_income=903000,
            employee_count=58,
            bank_statements=_generate_statements(
                monthly, 220000,
                ["SunState Utilities", "GreenGrid Power", "EcoHome Builders", "Municipal Energy Co-op"],
                ["SolarPanel Direct", "Inverter Solutions Inc", "CopperWire Supply"],
                seed=1003,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "Helios Solar Works installs commercial and municipal solar arrays. Revenue is "
                "seasonal with a strong Q2-Q3 peak driven by construction weather windows. Despite "
                "seasonality, the company has maintained positive cash flow every month for 3 "
                "consecutive years. The company has a $2.1M contracted backlog and seeks capital "
                "to purchase bulk panel inventory at favorable pricing."
            ),
            loan_request_amount=750000,
            loan_purpose="Bulk solar panel inventory purchase for contracted backlog",
        ),
        true_outcome="good",
    ))

    # GOOD 4: UrbanGrow Collective - small but consistent urban farm
    monthly = [
        (55000, 41000), (56000, 41500), (58000, 42000), (60000, 43000),
        (62000, 44000), (65000, 45000), (67000, 46000), (68000, 46500),
        (69000, 47000), (70000, 48000), (71000, 48500), (72000, 49000),
    ]
    borrowers.append(Borrower(
        id="BRW-004",
        dossier=FinancialDossier(
            company_name="UrbanGrow Collective",
            sector="Urban Agriculture",
            years_in_business=5,
            annual_revenue=773000,
            annual_expenses=541500,
            net_income=231500,
            employee_count=15,
            bank_statements=_generate_statements(
                monthly, 48000,
                ["FreshMart Grocers", "Farm-to-Table Distributors", "City School District"],
                ["Seed & Soil Supply", "HydroTech Systems"],
                seed=1004,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "UrbanGrow Collective operates vertical hydroponic farms in converted warehouse "
                "spaces, supplying fresh produce to local grocery chains and a school district "
                "contract. The business model is capital-light with predictable recurring revenue. "
                "Margins are modest but consistent at 30%. Seeking capital to open a second "
                "growing facility to serve a newly won restaurant group contract."
            ),
            loan_request_amount=150000,
            loan_purpose="Second hydroponic facility buildout for new contract",
        ),
        true_outcome="good",
    ))

    # GOOD 5: PixelForge Studios - project-based but healthy digital media
    monthly = [
        (120000, 88000), (95000, 80000), (180000, 115000), (140000, 95000),
        (85000, 75000), (200000, 125000), (165000, 108000), (110000, 85000),
        (190000, 120000), (145000, 98000), (130000, 90000), (175000, 112000),
    ]
    borrowers.append(Borrower(
        id="BRW-005",
        dossier=FinancialDossier(
            company_name="PixelForge Studios",
            sector="Digital Media",
            years_in_business=7,
            annual_revenue=1735000,
            annual_expenses=1191000,
            net_income=544000,
            employee_count=28,
            bank_statements=_generate_statements(
                monthly, 130000,
                ["AdVenture Agency", "StreamVault Media", "GameOn Interactive", "BrandCraft Corp"],
                ["Adobe Licensing", "CloudHost Services", "Freelancer Payments"],
                seed=1005,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "PixelForge Studios is a full-service digital media production company specializing "
                "in commercial video, animation, and interactive content. Revenue is project-based "
                "and naturally lumpy, but annual totals have grown consistently. The company "
                "maintains a 6-month cash runway at all times. Seeking capital for a motion "
                "capture studio that will unlock higher-margin work."
            ),
            loan_request_amount=400000,
            loan_purpose="Motion capture studio construction and equipment",
        ),
        true_outcome="good",
    ))

    # GOOD 6: IronClad Manufacturing - established precision machining
    monthly = [
        (210000, 158000), (215000, 160000), (220000, 163000), (218000, 162000),
        (225000, 165000), (228000, 168000), (230000, 170000), (235000, 172000),
        (232000, 171000), (238000, 174000), (240000, 176000), (245000, 178000),
    ]
    borrowers.append(Borrower(
        id="BRW-013",
        dossier=FinancialDossier(
            company_name="IronClad Manufacturing",
            sector="Advanced Manufacturing",
            years_in_business=12,
            annual_revenue=2736000,
            annual_expenses=2017000,
            net_income=719000,
            employee_count=65,
            bank_statements=_generate_statements(
                monthly, 195000,
                ["Pratt Aerospace", "Caterpillar Parts Div", "General Dynamics Sub-Assembly", "Northrop Tooling"],
                ["MetalStock Supply", "CNC Tooling Inc", "Industrial Power Co"],
                seed=1013,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "IronClad Manufacturing is a precision machining shop producing specialty "
                "components for aerospace and heavy equipment OEMs. Founded in 2013, the "
                "company holds AS9100 and ISO 9001 certifications. Revenue has grown steadily "
                "through long-term supply agreements. Seeking capital to add a 5-axis CNC "
                "machine to reduce outsourcing costs on complex geometries."
            ),
            loan_request_amount=400000,
            loan_purpose="5-axis CNC machine acquisition",
        ),
        true_outcome="good",
    ))

    # GOOD 7: BlueLine Plumbing - recession-resistant services
    monthly = [
        (88000, 62000), (92000, 64000), (95000, 66000), (105000, 72000),
        (110000, 75000), (115000, 78000), (120000, 82000), (118000, 80000),
        (108000, 74000), (100000, 70000), (95000, 66000), (90000, 63000),
    ]
    borrowers.append(Borrower(
        id="BRW-014",
        dossier=FinancialDossier(
            company_name="BlueLine Plumbing Services",
            sector="Construction Services",
            years_in_business=15,
            annual_revenue=1236000,
            annual_expenses=852000,
            net_income=384000,
            employee_count=32,
            bank_statements=_generate_statements(
                monthly, 75000,
                ["CityBuild Contractors", "HomeServ Warranty", "Lakeside Property Mgmt", "State DOT"],
                ["Ferguson Supply", "Fleet Fuel Card", "Workers Comp Insurance"],
                seed=1014,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "BlueLine Plumbing Services provides commercial and residential plumbing "
                "installation and repair. The company has operated continuously for 15 years "
                "through multiple economic cycles. Revenue is seasonal (peaks in summer) but "
                "annual margins are consistently above 30%. Seeking capital to purchase two "
                "service vans and inventory for a new municipal maintenance contract."
            ),
            loan_request_amount=200000,
            loan_purpose="Fleet expansion and inventory for municipal contract",
        ),
        true_outcome="good",
    ))

    # GOOD 8: Apex Data Solutions - B2B SaaS with recurring revenue
    monthly = [
        (165000, 110000), (170000, 112000), (178000, 116000), (182000, 118000),
        (190000, 122000), (195000, 125000), (200000, 128000), (208000, 132000),
        (215000, 136000), (220000, 138000), (228000, 142000), (235000, 146000),
    ]
    borrowers.append(Borrower(
        id="BRW-015",
        dossier=FinancialDossier(
            company_name="Apex Data Solutions",
            sector="Enterprise SaaS",
            years_in_business=6,
            annual_revenue=2386000,
            annual_expenses=1525000,
            net_income=861000,
            employee_count=40,
            bank_statements=_generate_statements(
                monthly, 140000,
                ["Regions Financial Group", "Sysco Food Services", "Hilton Hotels Corp", "AutoNation Inc", "Waste Management"],
                ["AWS Hosting", "Salesforce License", "Engineering Payroll"],
                seed=1015,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "Apex Data Solutions provides cloud-based analytics and reporting tools to "
                "mid-market enterprises. 90% of revenue is recurring SaaS subscriptions with "
                "annual contracts. Net revenue retention is 115% driven by seat expansion. "
                "The company has been cash-flow positive for 3 years. Seeking capital to hire "
                "a sales team targeting the hospitality vertical."
            ),
            loan_request_amount=350000,
            loan_purpose="Sales team expansion for hospitality vertical",
        ),
        true_outcome="good",
    ))

    # GOOD 9: Coastal Seafood Distributors - seasonal but profitable
    monthly = [
        (95000, 72000), (88000, 68000), (82000, 65000), (110000, 82000),
        (145000, 105000), (180000, 128000), (195000, 138000), (190000, 135000),
        (160000, 115000), (125000, 92000), (105000, 78000), (98000, 74000),
    ]
    borrowers.append(Borrower(
        id="BRW-016",
        dossier=FinancialDossier(
            company_name="Coastal Seafood Distributors",
            sector="Food Distribution",
            years_in_business=11,
            annual_revenue=1573000,
            annual_expenses=1152000,
            net_income=421000,
            employee_count=25,
            bank_statements=_generate_statements(
                monthly, 85000,
                ["Whole Foods Regional", "Chesapeake Restaurants", "FreshCatch Markets", "Harbor Hotels Group"],
                ["Fleet Fisheries", "Cold Storage Logistics", "DOT Compliance"],
                seed=1016,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "Coastal Seafood Distributors supplies fresh and frozen seafood to restaurants, "
                "hotels, and specialty grocers along the eastern seaboard. Revenue is seasonal "
                "with a strong May-September peak. The company has maintained positive annual "
                "cash flow every year since founding. Seeking capital to add a refrigerated "
                "truck and expand cold storage capacity for the upcoming peak season."
            ),
            loan_request_amount=275000,
            loan_purpose="Refrigerated truck and cold storage expansion",
        ),
        true_outcome="good",
    ))

    # GOOD 10: Keystone Legal Tech - legal SaaS with court system contracts
    monthly = [
        (130000, 92000), (132000, 93000), (135000, 95000), (138000, 96000),
        (140000, 98000), (142000, 99000), (145000, 100000), (148000, 102000),
        (150000, 104000), (152000, 105000), (155000, 106000), (158000, 108000),
    ]
    borrowers.append(Borrower(
        id="BRW-017",
        dossier=FinancialDossier(
            company_name="Keystone Legal Tech",
            sector="Legal Technology",
            years_in_business=7,
            annual_revenue=1725000,
            annual_expenses=1198000,
            net_income=527000,
            employee_count=30,
            bank_statements=_generate_statements(
                monthly, 120000,
                ["State Court Admin Office", "Baker McKenzie LLP", "LegalShield Corp", "County Clerk Consortium"],
                ["Azure Cloud Services", "Developer Payroll", "Compliance Audit Co"],
                seed=1017,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "Keystone Legal Tech provides case management and e-filing software to state "
                "court systems and law firms. Revenue is 80% recurring through multi-year "
                "government contracts with automatic renewal clauses. The company has never "
                "lost a contract renewal. Seeking capital to build an AI-assisted document "
                "review module that three existing clients have pre-committed to purchase."
            ),
            loan_request_amount=300000,
            loan_purpose="AI document review module development",
        ),
        true_outcome="good",
    ))

    # ===== BAD BUSINESSES (4) =====

    # BAD 1: AeroTrack Dynamics - dangerous single-customer concentration
    # 85% of revenue from "Titan Defense Corp"
    monthly = [
        (195000, 152000), (192000, 150000), (198000, 155000), (190000, 148000),
        (196000, 153000), (193000, 151000), (197000, 154000), (191000, 149000),
        (194000, 152000), (196000, 153000), (192000, 150000), (195000, 152000),
    ]
    borrowers.append(Borrower(
        id="BRW-006",
        dossier=FinancialDossier(
            company_name="AeroTrack Dynamics",
            sector="Aero-Logistics",
            years_in_business=5,
            annual_revenue=2329000,
            annual_expenses=1819000,
            net_income=510000,
            employee_count=38,
            bank_statements=_generate_statements(
                monthly, 160000,
                # Note: Titan Defense is ~85% of deposits
                ["Titan Defense Corp", "Titan Defense Corp - Div. B", "Skyline Freight Co"],
                ["Fleet Maintenance Ltd", "Aviation Fuel Direct", "Hangar Lease Co"],
                seed=1006,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "AeroTrack Dynamics provides specialized cargo handling and logistics for defense "
                "contractors. The company has a strong track record with its anchor client and "
                "recently expanded services to include hazardous materials handling. Revenue is "
                "stable and the company reports healthy margins. Seeking capital to acquire "
                "specialized loading equipment for a potential contract expansion."
            ),
            loan_request_amount=600000,
            loan_purpose="Specialized cargo loading equipment acquisition",
        ),
        true_outcome="bad",
        months_before_default=8,  # Titan Defense cuts contract at month 8
    ))

    # BAD 2: SynthaCure Pharma - margins compressing, expenses growing faster than revenue
    monthly = [
        (152000, 102000), (153000, 106000), (151000, 110000), (154000, 114000),
        (152000, 118000), (155000, 122000), (153000, 126000), (154000, 130000),
        (152000, 134000), (153000, 138000), (151000, 141000), (154000, 145000),
    ]
    borrowers.append(Borrower(
        id="BRW-007",
        dossier=FinancialDossier(
            company_name="SynthaCure Pharma",
            sector="Bio-Synthetics",
            years_in_business=6,
            annual_revenue=1834000,
            annual_expenses=1486000,
            net_income=348000,
            employee_count=42,
            bank_statements=_generate_statements(
                monthly, 110000,
                ["Regional Health Network", "PharmaBridge Distributors", "WellCare Clinics"],
                ["Raw Chemical Suppliers Inc", "Regulatory Compliance Co", "Lab Staff Agency"],
                seed=1007,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "SynthaCure Pharma manufactures generic pharmaceutical compounds for regional "
                "health networks. Revenue has been stable around $150K/month with a loyal customer "
                "base. The company recently invested in regulatory compliance upgrades and "
                "quality control systems. Management projects continued stable revenue and is "
                "seeking capital to scale production capacity."
            ),
            loan_request_amount=500000,
            loan_purpose="Production capacity expansion and equipment upgrades",
        ),
        true_outcome="bad",
        months_before_default=14,  # Margins turn negative, can't service debt
    ))

    # BAD 3: QuantumLeap Systems - grant-dependent, no real commercial revenue
    monthly = [
        (125000, 112000), (118000, 110000), (122000, 113000), (130000, 115000),
        (115000, 108000), (128000, 114000), (120000, 111000), (125000, 113000),
        (118000, 110000), (122000, 112000), (126000, 114000), (121000, 111000),
    ]
    borrowers.append(Borrower(
        id="BRW-008",
        dossier=FinancialDossier(
            company_name="QuantumLeap Systems",
            sector="Quantum Computing",
            years_in_business=3,
            annual_revenue=1470000,
            annual_expenses=1343000,
            net_income=127000,
            employee_count=19,
            bank_statements=_generate_statements(
                monthly, 72000,
                # Note: Most revenue is grants, not commercial
                ["Federal Research Grant - DARPA", "NSF Innovation Award", "Quantum Horizons Fund"],
                ["Cryogenics Equipment Co", "University Lab Lease", "Research Staff Payroll"],
                seed=1008,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "QuantumLeap Systems is a cutting-edge quantum computing research company "
                "developing novel qubit architectures. The company has received recognition from "
                "leading research institutions and maintains active research partnerships. "
                "Revenue streams are diversified across multiple sources. Seeking capital to "
                "build a dedicated testing facility for their latest quantum processor design."
            ),
            loan_request_amount=400000,
            loan_purpose="Quantum processor testing facility construction",
        ),
        true_outcome="bad",
        months_before_default=10,  # Grants not renewed
    ))

    # BAD 4: TerraVolt Energy - steadily declining revenue
    monthly = [
        (182000, 142000), (175000, 140000), (168000, 138000), (160000, 136000),
        (152000, 134000), (145000, 133000), (138000, 132000), (130000, 131000),
        (122000, 130000), (115000, 128000), (108000, 127000), (100000, 126000),
    ]
    borrowers.append(Borrower(
        id="BRW-009",
        dossier=FinancialDossier(
            company_name="TerraVolt Energy",
            sector="Green Energy",
            years_in_business=9,
            annual_revenue=1695000,
            annual_expenses=1597000,
            net_income=98000,
            employee_count=35,
            bank_statements=_generate_statements(
                monthly, 135000,
                ["PowerGrid Solutions", "EcoVolt Residential", "Green Municipal Alliance"],
                ["Battery Supply Co", "Electrician Contractors", "Warehouse Rent"],
                seed=1009,
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "TerraVolt Energy installs and services battery storage systems for residential "
                "and commercial clients. The company has been in business for 9 years and has an "
                "established reputation. Recent market conditions have shifted due to new "
                "competitors entering the space, but management is implementing a strategic pivot "
                "to commercial-scale installations. Seeking capital to fund the transition and "
                "retool operations."
            ),
            loan_request_amount=350000,
            loan_purpose="Strategic pivot funding - retooling for commercial installations",
        ),
        true_outcome="bad",
        months_before_default=6,  # Revenue decline accelerates
    ))

    # ===== FRAUDULENT BUSINESSES (3) =====

    # FRAUD 1: CloudNet Logistics - round number deposits, inflated revenue claims
    monthly = [
        (275000, 180000), (250000, 175000), (300000, 190000), (275000, 185000),
        (325000, 195000), (250000, 178000), (300000, 188000), (275000, 182000),
        (350000, 200000), (250000, 176000), (300000, 192000), (275000, 184000),
    ]
    borrowers.append(Borrower(
        id="BRW-010",
        dossier=FinancialDossier(
            company_name="CloudNet Logistics",
            sector="Aero-Logistics",
            years_in_business=4,
            annual_revenue=3525000,
            annual_expenses=2225000,
            net_income=1300000,
            employee_count=52,
            bank_statements=_generate_statements(
                monthly, 200000,
                ["GlobalFreight Partners", "AirCargo International", "TransOcean Shipping", "Swift Delivery Network"],
                ["Fuel Depot Services", "Aircraft Lease Corp", "Ground Crew Staffing"],
                seed=1010,
                fraud_type="round_numbers",
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "CloudNet Logistics is a rapidly growing air freight and last-mile delivery "
                "company. The company reports exceptional margins of 37% and has experienced "
                "explosive growth since inception. CloudNet services Fortune 500 clients across "
                "multiple continents and is seeking capital to open a new regional hub. Management "
                "is highly confident in continued growth trajectory."
            ),
            loan_request_amount=800000,
            loan_purpose="New regional distribution hub buildout",
        ),
        true_outcome="fraud",
    ))

    # FRAUD 2: BioGenesis Research - circular transfers between related entities
    monthly = [
        (220000, 195000), (235000, 210000), (225000, 200000), (240000, 215000),
        (228000, 203000), (232000, 208000), (238000, 213000), (222000, 197000),
        (230000, 205000), (236000, 211000), (226000, 201000), (234000, 209000),
    ]
    borrowers.append(Borrower(
        id="BRW-011",
        dossier=FinancialDossier(
            company_name="BioGenesis Research",
            sector="Bio-Synthetics",
            years_in_business=3,
            annual_revenue=2766000,
            annual_expenses=2467000,
            net_income=299000,
            employee_count=30,
            bank_statements=_generate_statements(
                monthly, 150000,
                ["Genova Pharmaceuticals", "LifeScience Direct"],
                ["BioGenesis Holdings LLC", "BGH Capital Partners", "Lab Lease Corp"],
                seed=1011,
                fraud_type="circular",
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "BioGenesis Research is a biotechnology company specializing in novel compound "
                "synthesis for pharmaceutical applications. The company operates under the "
                "BioGenesis Holdings umbrella and leverages shared resources across affiliated "
                "entities. Revenue has been strong and consistent. Seeking capital to fund Phase 2 "
                "clinical trials for their lead compound."
            ),
            loan_request_amount=450000,
            loan_purpose="Phase 2 clinical trial funding",
        ),
        true_outcome="fraud",
    ))

    # FRAUD 3: QubitTech Solutions - fabricated statements (unnaturally consistent)
    monthly = [
        (145200, 108100), (145400, 108300), (145100, 107900), (145300, 108200),
        (145500, 108400), (145200, 108100), (145300, 108000), (145400, 108300),
        (145100, 108200), (145300, 108100), (145200, 108300), (145400, 108200),
    ]
    borrowers.append(Borrower(
        id="BRW-012",
        dossier=FinancialDossier(
            company_name="QubitTech Solutions",
            sector="Quantum Computing",
            years_in_business=5,
            annual_revenue=1743400,
            annual_expenses=1298100,
            net_income=445300,
            employee_count=24,
            bank_statements=_generate_statements(
                monthly, 165000,
                ["TechForward Inc", "DataCore Systems", "QuantumSafe Security", "NeuralNet Partners"],
                ["Server Farm Lease", "Component Supply Chain", "R&D Consulting"],
                seed=1012,
                fraud_type="fabricated",
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "QubitTech Solutions provides quantum-resistant encryption services to enterprise "
                "clients. The company reports remarkably stable revenue and consistent margins "
                "across all periods. Operations are highly optimized with minimal variance in "
                "monthly performance. Seeking capital to expand data center capacity."
            ),
            loan_request_amount=350000,
            loan_purpose="Data center capacity expansion",
        ),
        true_outcome="fraud",
    ))

    # FRAUD 4: Orion Fleet Services - structured deposits (smurfing)
    # All deposits are broken into many sub-$10K transactions
    monthly = [
        (128000, 94000), (132000, 96000), (130000, 95000), (135000, 98000),
        (131000, 95000), (133000, 97000), (129000, 94000), (134000, 97000),
        (136000, 99000), (132000, 96000), (130000, 95000), (135000, 98000),
    ]
    borrowers.append(Borrower(
        id="BRW-018",
        dossier=FinancialDossier(
            company_name="Orion Fleet Services",
            sector="Construction Services",
            years_in_business=6,
            annual_revenue=1585000,
            annual_expenses=1154000,
            net_income=431000,
            employee_count=28,
            bank_statements=_generate_statements(
                monthly, 110000,
                ["Metro Builders", "Apex Construction", "Summit Paving Co",
                 "Ridgeline Contractors", "Ironwork Specialists"],
                ["Diesel Depot", "Heavy Equipment Leasing", "Fleet Insurance Corp"],
                seed=1018,
                fraud_type="structured",
            ),
            quarterly_income=_build_quarterly_income(monthly),
            narrative=(
                "Orion Fleet Services provides heavy equipment rental and fleet management "
                "to mid-size construction firms. The company has grown steadily over 6 years "
                "and reports healthy 27% margins. Revenue comes from a diversified base of "
                "regional contractors. Seeking capital to acquire three additional excavators "
                "and a crane to meet growing demand from infrastructure projects."
            ),
            loan_request_amount=350000,
            loan_purpose="Heavy equipment acquisition (3 excavators + crane)",
        ),
        true_outcome="fraud",
    ))

    return borrowers


# ---------------------------------------------------------------------------
# Lender configurations
# ---------------------------------------------------------------------------

def _build_lenders() -> list[LenderConfig]:
    return [
        LenderConfig(
            id="LND-001",
            name="Velocity Capital",
            persona=(
                "You are Velocity Capital, an aggressive growth-oriented fintech lender. "
                "You pursue high-yield opportunities and are willing to take on more risk for "
                "better returns. You favor fast-growing companies and innovative sectors. "
                "However, you still must avoid fraud and require basic creditworthiness. "
                "You prefer shorter loan terms (12-24 months) with higher interest rates."
            ),
            model="deepseek/deepseek-chat-v3-0324",
            target_yield_pct=14.0,
            max_single_loan=800000,
            total_capital=3000000,
            sector_limits={
                "Aero-Logistics": 0.30,
                "Bio-Synthetics": 0.30,
                "Quantum Computing": 0.25,
                "Green Energy": 0.30,
                "Urban Agriculture": 0.25,
                "Digital Media": 0.30,
                "Advanced Manufacturing": 0.25,
                "Construction Services": 0.25,
                "Enterprise SaaS": 0.30,
                "Food Distribution": 0.25,
                "Legal Technology": 0.25,
            },
            existing_portfolio=[
                ExistingLoan("QuantumCore Inc", "Quantum Computing", 400000, 320000, 12.0, 18),
                ExistingLoan("QuantumSafe Networks", "Quantum Computing", 350000, 280000, 13.0, 15),
                ExistingLoan("QuBit Dynamics", "Quantum Computing", 300000, 250000, 11.5, 20),
                ExistingLoan("AeroSwift Cargo", "Aero-Logistics", 250000, 180000, 10.0, 12),
                ExistingLoan("BioNova Therapeutics", "Bio-Synthetics", 200000, 150000, 12.5, 16),
            ],
        ),
        LenderConfig(
            id="LND-002",
            name="Heritage Trust Bank",
            persona=(
                "You are Heritage Trust Bank, a conservative regional bank with strict "
                "underwriting standards. Capital preservation is your top priority. You prefer "
                "established businesses with long track records and predictable cash flows. "
                "You avoid speculative sectors and require strong debt service coverage ratios. "
                "You offer lower interest rates but demand higher creditworthiness. "
                "You prefer longer terms (24-36 months) with moderate rates."
            ),
            model="qwen/qwen3-235b-a22b",
            target_yield_pct=8.0,
            max_single_loan=600000,
            total_capital=4000000,
            sector_limits={
                "Aero-Logistics": 0.25,
                "Bio-Synthetics": 0.25,
                "Quantum Computing": 0.15,
                "Green Energy": 0.25,
                "Urban Agriculture": 0.30,
                "Digital Media": 0.20,
                "Advanced Manufacturing": 0.25,
                "Construction Services": 0.30,
                "Enterprise SaaS": 0.20,
                "Food Distribution": 0.25,
                "Legal Technology": 0.25,
            },
            existing_portfolio=[
                ExistingLoan("SkyBridge Freight", "Aero-Logistics", 500000, 420000, 7.5, 24),
                ExistingLoan("AeroCourier Services", "Aero-Logistics", 450000, 380000, 7.0, 22),
                ExistingLoan("JetStream Cargo", "Aero-Logistics", 350000, 300000, 8.0, 20),
                ExistingLoan("FreshFields Farms", "Urban Agriculture", 200000, 160000, 6.5, 28),
                ExistingLoan("SolarPeak Energy", "Green Energy", 300000, 240000, 7.5, 26),
            ],
        ),
        LenderConfig(
            id="LND-003",
            name="Meridian Partners",
            persona=(
                "You are Meridian Partners, a balanced middle-market lender. You seek a "
                "mix of risk and return, carefully weighing each opportunity. You value "
                "strong cash flow fundamentals but are open to growth-stage companies if "
                "the unit economics make sense. You are particularly vigilant about fraud "
                "and inconsistencies in financial statements. "
                "You offer competitive terms (18-30 months) with fair interest rates."
            ),
            model="meta-llama/llama-3.3-70b-instruct",
            target_yield_pct=11.0,
            max_single_loan=700000,
            total_capital=3500000,
            sector_limits={
                "Aero-Logistics": 0.25,
                "Bio-Synthetics": 0.30,
                "Quantum Computing": 0.20,
                "Green Energy": 0.25,
                "Urban Agriculture": 0.25,
                "Digital Media": 0.25,
                "Advanced Manufacturing": 0.25,
                "Construction Services": 0.25,
                "Enterprise SaaS": 0.25,
                "Food Distribution": 0.25,
                "Legal Technology": 0.25,
            },
            existing_portfolio=[
                ExistingLoan("SynthWave Labs", "Bio-Synthetics", 400000, 340000, 10.0, 22),
                ExistingLoan("BioFusion Corp", "Bio-Synthetics", 350000, 290000, 9.5, 20),
                ExistingLoan("GreenVolt Systems", "Green Energy", 300000, 250000, 10.5, 18),
                ExistingLoan("DigitalPulse Media", "Digital Media", 250000, 200000, 9.0, 24),
                ExistingLoan("AgroTech Vertical", "Urban Agriculture", 150000, 120000, 8.5, 16),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Mix presets — control the good/bad/fraud ratio of the borrower pool
# ---------------------------------------------------------------------------

# Maps mix name -> (good_ids, bad_ids, fraud_ids)
# IDs are cherry-picked so each preset tells a different story.
MIX_PRESETS: dict[str, dict[str, list[str]]] = {
    # ~75% good, ~17% bad, ~8% fraud — realistic commercial pipeline
    "easy": {
        "good":  ["BRW-001", "BRW-002", "BRW-003", "BRW-004", "BRW-005",
                   "BRW-013", "BRW-014", "BRW-015", "BRW-016"],
        "bad":   ["BRW-007", "BRW-009"],
        "fraud": ["BRW-011"],
    },
    # ~50% good, ~29% bad, ~21% fraud — stressed market
    "balanced": {
        "good":  ["BRW-001", "BRW-002", "BRW-004", "BRW-005", "BRW-013",
                   "BRW-015", "BRW-017"],
        "bad":   ["BRW-006", "BRW-007", "BRW-008", "BRW-009"],
        "fraud": ["BRW-010", "BRW-011", "BRW-012", "BRW-018"],
    },
    # ~42% good, ~33% bad, ~25% fraud — adversarial stress test (original mix)
    "hard": {
        "good":  ["BRW-001", "BRW-002", "BRW-003", "BRW-004", "BRW-005"],
        "bad":   ["BRW-006", "BRW-007", "BRW-008", "BRW-009"],
        "fraud": ["BRW-010", "BRW-011", "BRW-012", "BRW-018"],
    },
    # Full pool — everything
    "all": {
        "good":  ["BRW-001", "BRW-002", "BRW-003", "BRW-004", "BRW-005",
                   "BRW-013", "BRW-014", "BRW-015", "BRW-016", "BRW-017"],
        "bad":   ["BRW-006", "BRW-007", "BRW-008", "BRW-009"],
        "fraud": ["BRW-010", "BRW-011", "BRW-012", "BRW-018"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_borrowers(mix: str = "easy") -> list[Borrower]:
    """Return borrowers filtered by the chosen mix preset."""
    all_borrowers = _build_borrowers()
    if mix not in MIX_PRESETS:
        raise ValueError(f"Unknown mix '{mix}'. Choose from: {list(MIX_PRESETS)}")
    preset = MIX_PRESETS[mix]
    allowed = set(preset["good"] + preset["bad"] + preset["fraud"])
    return [b for b in all_borrowers if b.id in allowed]


def get_lenders() -> list[LenderConfig]:
    return _build_lenders()
