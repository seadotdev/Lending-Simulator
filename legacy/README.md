# Legacy

Pre-LOS benchmark scripts, test harnesses, and result files from the
direct-OpenRouter evaluation era. Kept for reference only.

## Contents

- **benchmarks/** — Model benchmark and Elo tournament scripts
  (`benchmark_models.py`, `elo_benchmark.py`, `run_lite_eval.py`)
- **results/** — JSON/PNG output from old benchmark runs
- **tests/** — Ad-hoc test scripts (bank statements, jq, lite scenarios,
  underwriting modes, scoring verification)
- **runs/** — Mock replay hashes

## Current workflow

All simulation is now driven through:

```bash
# Mock (no API key):
python -m loanville --mock --economics balanced

# Via Open LOS:
./run_sim.sh --economics aggressive
```

See the root README for full usage.
