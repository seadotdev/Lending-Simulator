"""
Cost tracking and estimation for season mode.

Estimates cost based on typical tokens per evaluation.
"""

from .models import SeasonConfig, SeasonLenderState

# Approximate $/1M-token pricing from OpenRouter.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input $/M tokens, output $/M tokens)
    "anthropic/claude-3.5-haiku":       (0.80, 4.00),
    "google/gemini-3-flash-preview":    (0.50, 3.00),
    "google/gemini-2.5-flash":          (0.30, 2.50),
    "deepseek/deepseek-r1":             (0.70, 2.50),
    "moonshotai/kimi-k2":               (0.50, 2.40),
    "qwen/qwen3.5-plus-02-15":         (0.40, 2.40),
    "minimax/minimax-m1":               (0.40, 2.20),
    "mistralai/mistral-medium-3.1":     (0.40, 2.00),
    "mistralai/devstral-medium":        (0.40, 2.00),
    "bytedance-seed/seed-1.6":          (0.25, 2.00),
    "openai/gpt-5-mini":               (0.25, 2.00),
    "qwen/qwen3-235b-a22b":            (0.46, 1.82),
    "deepseek/deepseek-r1-0528":        (0.40, 1.75),
    "openai/gpt-4.1-mini":             (0.40, 1.60),
    "mistralai/mistral-large-2512":     (0.50, 1.50),
    "qwen/qwen3-coder-flash":          (0.30, 1.50),
    "anthropic/claude-3-haiku":         (0.25, 1.25),
    "qwen/qwen-plus":                   (0.40, 1.20),
    "deepseek/deepseek-chat":           (0.32, 0.89),
    "deepseek/deepseek-chat-v3-0324":   (0.19, 0.87),
    "deepseek/deepseek-chat-v3.1":      (0.15, 0.75),
    "openai/gpt-4o-mini":              (0.15, 0.60),
    "meta-llama/llama-4-maverick":      (0.15, 0.60),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.10, 0.40),
    "google/gemini-2.5-flash-lite":     (0.10, 0.40),
    "google/gemini-2.0-flash-001":      (0.10, 0.40),
    "qwen/qwq-32b":                     (0.15, 0.40),
    "meta-llama/llama-3.1-70b-instruct": (0.40, 0.40),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "meta-llama/llama-4-scout":         (0.08, 0.30),
    "qwen/qwen3-30b-a3b":              (0.08, 0.28),
    "qwen/qwen-2.5-7b-instruct":       (0.04, 0.10),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
}

# Typical tokens per evaluation (prompt + completion).
# Measured from real runs: full dossier ~2500 prompt, ~500 completion.
EST_PROMPT_TOKENS = 2500
EST_COMPLETION_TOKENS = 500


def _estimate_cost_per_eval(model: str) -> float:
    """Estimate cost of one (lender, borrower) evaluation."""
    in_price, out_price = MODEL_PRICING.get(model, (0.50, 1.50))
    return (EST_PROMPT_TOKENS / 1_000_000 * in_price +
            EST_COMPLETION_TOKENS / 1_000_000 * out_price)


def print_season_cost_summary(
    lender_states: dict[str, SeasonLenderState],
    config: SeasonConfig,
    mock: bool = False,
) -> None:
    """Print estimated cost summary for a season run."""
    print(f"\n{'─' * 60}")
    print(f"  ESTIMATED API COST (if run live)")
    print(f"{'─' * 60}")

    total_evals = sum(s.total_evaluations for s in lender_states.values())
    if total_evals == 0:
        total_evals = config.weeks * config.cohort_size * len(lender_states)

    # Group evaluations by model
    model_evals: dict[str, int] = {}
    for state in lender_states.values():
        n = state.total_evaluations or (config.weeks * config.cohort_size)
        model_evals[state.model] = model_evals.get(state.model, 0) + n

    total_est = 0.0
    for model in sorted(model_evals.keys()):
        n = model_evals[model]
        per_eval = _estimate_cost_per_eval(model)
        est = per_eval * n
        total_est += est
        in_price, out_price = MODEL_PRICING.get(model, (0.50, 1.50))
        print(f"  {model}")
        print(f"    {n} evaluations × ~{EST_PROMPT_TOKENS}p+{EST_COMPLETION_TOKENS}c tokens")
        print(f"    Rate: ${in_price:.2f}/${out_price:.2f} per M tokens (in/out)")
        print(f"    Est. cost: ${est:.4f}")

    print(f"  {'─' * 56}")
    print(f"  TOTAL EVALUATIONS: {total_evals}")
    print(f"  ESTIMATED TOTAL COST: ${total_est:.4f}")

    if total_est < 0.01:
        print(f"  (less than one cent)")
    elif total_est < 1.00:
        print(f"  (under a dollar)")
