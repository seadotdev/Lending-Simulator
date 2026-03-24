# Contributing to Lending Simulator

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo (with submodules)
git clone --recurse-submodules https://github.com/seadotdev/Lending-Simulator.git
cd Lending-Simulator

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Open LOS (submodule)

The simulation can run against the [Open LOS](https://github.com/seadotdev/open-los) loan origination system:

```bash
cd open-los
npm install
```

## Running Tests

```bash
# Unit / regression tests
python -m pytest tests/

# Determinism check (mock replay)
python scripts/check_mock_replay.py

# LOS integration smoke test (requires Open LOS running)
python scripts/test_los_smoke.py
```

## Making Changes

1. Fork the repo and create a feature branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Add or update tests if your change affects simulation logic or scoring.
4. Run the test suite and confirm everything passes.
5. Open a pull request with a clear description of what changed and why.

## Code Style

- Python code follows standard PEP 8 conventions.
- Keep functions focused and files reasonably sized.
- Prefer clarity over cleverness — this is a financial simulation, correctness matters.

## Borrower & Scenario Data

The 24 hand-crafted borrower dossiers in `loanville/data.py` are carefully designed to test specific underwriting skills. If you're modifying or adding borrowers, document the intended risk signal and expected outcome.

## Reporting Bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Relevant logs or error output

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
