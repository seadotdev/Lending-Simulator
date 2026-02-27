#!/usr/bin/env bash
# ============================================================================
# run_sim.sh — Start Open LOS + run Loanville lending simulation
# ============================================================================
#
# WHAT THIS DOES
# ==============
# This script boots the Open LOS (Loan Origination System) REST API server,
# then runs the Loanville lending simulator against it. The simulator sends
# borrower dossiers to the LOS, which uses an LLM to make underwriting
# decisions (approve/reject, set terms, detect fraud).
#
# All non-mock simulations run through the Open LOS pipeline. The LOS
# orchestrates the LLM calls with proper audit trails, financial spreading,
# and stage management.
#
# ARCHITECTURE
# ============
#
#   Loanville (simulator)
#       │
#       │  POST /v1/underwrite  (dossier + policy)
#       ▼
#   Open LOS (localhost:3000)
#       │
#       │  LLM call via OpenRouter
#       ▼
#   Nemotron 49B / Llama 70B / etc.
#       │
#       │  Structured underwriting decision
#       ▼
#   Loanville (scoring, Elo, RAROC)
#
# The simulator runs 3 AI lenders (different personas & risk appetites)
# against a pool of 20 borrowers (90% good, 5% bad, 5% fraud). Each
# lender independently evaluates every borrower, then borrowers pick
# the best offer. Loans are resolved (repaid or defaulted) and lenders
# are scored on RAROC.
#
# DEFAULT MODELS (30B-80B range, cheap on OpenRouter)
# ===================================================
#   Velocity Capital:   deepseek/deepseek-chat-v3-0324
#   Heritage Trust:     google/gemini-2.5-flash
#   Meridian Partners:  meta-llama/llama-3.3-70b-instruct
#
# Override all lenders to a single model with --los-model:
#   ./run_sim.sh --los-model nvidia/llama-3.3-nemotron-super-49b-v1.5
#
# PREREQUISITES
# =============
#   1. OpenRouter API key in ~/.env:
#        OPENROUTER_API_KEY=sk-or-v1-...
#
#   2. Node.js (for Open LOS):
#        git submodule update --init && cd open-los && npm install
#
#   3. Python deps (for Loanville):
#        cd ../loanville2 && pip install -r requirements.txt
#
# USAGE
# =====
#   ./run_sim.sh                     # Default: realistic mix, full LOS pipeline
#   ./run_sim.sh --mix easy          # Easier borrower pool (12 borrowers)
#   ./run_sim.sh --mix hard          # Adversarial stress test
#   ./run_sim.sh --full-pipeline     # Full LOS pipeline (explicit; same as default)
#   ./run_sim.sh --underwrite-only   # Legacy direct /v1/underwrite path (non-formal)
#   ./run_sim.sh --los-model MODEL   # Override LLM model for all lenders
#   ./run_sim.sh --economics aggressive  # Aggressive economics preset
#   ./run_sim.sh --mock              # Mock mode (no API key, no LOS, deterministic)
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Prefer submodule (./open-los), fall back to sibling (../open-los)
if [ -d "$SCRIPT_DIR/open-los" ]; then
    LOS_DIR="$(cd "$SCRIPT_DIR/open-los" && pwd)"
else
    LOS_DIR="$(cd "$SCRIPT_DIR/../open-los" && pwd)"
fi
LOS_PORT=3000
LOS_PID=""

# ── Defaults ────────────────────────────────────────────────────────────────
MIX="realistic"
ECONOMICS="balanced"
LOS_PROVIDER="openrouter"
LOS_MODE="full"
UNDERWRITE_ONLY=""
MOCK=false
LEADERBOARD=false
EXTRA_ARGS=()

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mix)
            MIX="$2"; shift 2 ;;
        --economics)
            ECONOMICS="$2"; shift 2 ;;
        --los-model)
            EXTRA_ARGS+=("--los-model" "$2"); shift 2 ;;
        --full-pipeline)
            UNDERWRITE_ONLY=""; shift ;;
        --underwrite-only)
            UNDERWRITE_ONLY="--underwrite-only --allow-non-los-formal"; shift ;;
        --mock)
            MOCK=true; shift ;;
        --leaderboard)
            LEADERBOARD=true; shift ;;
        --los-provider)
            LOS_PROVIDER="$2"; shift 2 ;;
        --los-mode)
            LOS_MODE="$2"; shift 2 ;;
        --port)
            LOS_PORT="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Load API key from ~/.env ────────────────────────────────────────────────
if [[ -f "$HOME/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$HOME/.env"; set +a
fi
# Also load local .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# ── Mock mode: skip LOS entirely ────────────────────────────────────────────
if $MOCK; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  LOANVILLE — MOCK MODE (no LOS, no API key)"
    echo "═══════════════════════════════════════════════════════════════"
    cd "$SCRIPT_DIR"
    LB_FLAG=""
    if $LEADERBOARD; then LB_FLAG="--leaderboard"; fi
    python -m loanville --mock --allow-non-los-formal --mix "$MIX" --economics "$ECONOMICS" $LB_FLAG ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    exit 0
fi

# ── Verify API key ──────────────────────────────────────────────────────────
# At least one LLM provider key must be set
HAS_KEY=false
[[ -n "${OPENROUTER_API_KEY:-}" ]] && HAS_KEY=true
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && HAS_KEY=true
[[ -n "${OPENAI_API_KEY:-}" ]] && HAS_KEY=true
[[ -n "${AI_GATEWAY_API_KEY:-}" ]] && HAS_KEY=true
if ! $HAS_KEY; then
    echo "ERROR: No LLM provider API key found."
    echo "Set at least one in ~/.env:"
    echo "  OPENROUTER_API_KEY=sk-or-v1-..."
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  OPENAI_API_KEY=sk-..."
    echo "  AI_GATEWAY_API_KEY=..."
    echo ""
    echo "Or run in mock mode:  ./run_sim.sh --mock"
    exit 1
fi

# ── Kill anything already on the port ────────────────────────────────────────
EXISTING_PID=$(lsof -ti:"$LOS_PORT" 2>/dev/null || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Killing existing process on port $LOS_PORT (pid $EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
    if [[ -n "$LOS_PID" ]]; then
        echo ""
        echo "Shutting down Open LOS (pid $LOS_PID)..."
        kill "$LOS_PID" 2>/dev/null || true
        wait "$LOS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Start Open LOS ──────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Starting Open LOS on port $LOS_PORT..."
echo "═══════════════════════════════════════════════════════════════"

export PORT="$LOS_PORT"
# Forward all provider keys to the LOS process
[[ -n "${OPENROUTER_API_KEY:-}" ]] && export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
[[ -n "${OPENAI_API_KEY:-}" ]] && export OPENAI_API_KEY="${OPENAI_API_KEY}"
[[ -n "${AI_GATEWAY_API_KEY:-}" ]] && export AI_GATEWAY_API_KEY="${AI_GATEWAY_API_KEY}"
# Forward optional LLM routing config
[[ -n "${LOS_DEFAULT_PROVIDER:-}" ]] && export LOS_DEFAULT_PROVIDER="${LOS_DEFAULT_PROVIDER}"
[[ -n "${LOS_DEFAULT_MODEL:-}" ]] && export LOS_DEFAULT_MODEL="${LOS_DEFAULT_MODEL}"
[[ -n "${LOS_LLM_ROUTES:-}" ]] && export LOS_LLM_ROUTES="${LOS_LLM_ROUTES}"
[[ -n "${LOS_OPENROUTER_BASE_URL:-}" ]] && export LOS_OPENROUTER_BASE_URL="${LOS_OPENROUTER_BASE_URL}"
[[ -n "${LOS_VERCEL_BASE_URL:-}" ]] && export LOS_VERCEL_BASE_URL="${LOS_VERCEL_BASE_URL}"
[[ -n "${OPENAI_BASE_URL:-}" ]] && export OPENAI_BASE_URL="${OPENAI_BASE_URL}"

cd "$LOS_DIR"
npm run start --workspace=packages/api &
LOS_PID=$!

# Wait for LOS to be ready
echo "Waiting for LOS to start..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$LOS_PORT/v1/deals" >/dev/null 2>&1; then
        echo "Open LOS is ready."
        break
    fi
    if ! kill -0 "$LOS_PID" 2>/dev/null; then
        echo "ERROR: LOS process died. Check logs above."
        exit 1
    fi
    sleep 1
done

# Final check
if ! curl -sf "http://localhost:$LOS_PORT/v1/deals" >/dev/null 2>&1; then
    echo "ERROR: LOS did not start within 30 seconds."
    exit 1
fi

# ── Run Loanville simulation ────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running Loanville simulation"
echo "  Mix: $MIX | Economics: $ECONOMICS | Provider: $LOS_PROVIDER | Mode: $LOS_MODE"
if [[ -n "$UNDERWRITE_ONLY" ]]; then
    echo "  Path: underwrite-only (POST /v1/underwrite)"
else
    echo "  Path: full pipeline (entity → deal → docs → spread → evaluate)"
fi
echo "═══════════════════════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR"
LB_FLAG=""
if $LEADERBOARD; then LB_FLAG="--leaderboard"; fi
python -m loanville \
    --los-url "http://localhost:$LOS_PORT" \
    --los-provider "$LOS_PROVIDER" \
    --los-mode "$LOS_MODE" \
    -v \
    --mix "$MIX" \
    --economics "$ECONOMICS" \
    $UNDERWRITE_ONLY \
    $LB_FLAG \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "Done."
