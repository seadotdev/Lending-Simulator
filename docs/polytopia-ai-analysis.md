# Polytopia AI vs Loanville2: LLM-Driven Strategy Game Agents

## Context

Analysis of [Polytopia AI](https://github.com/ProjectAI00/polytopia-ai) — a framework for LLM-powered agents playing The Battle of Polytopia, a turn-based strategy game. The system injects into the closed-source game via BepInEx/Harmony modding, extracts game state, and sends it to Claude Sonnet 4 (via OpenRouter) for strategic reasoning.

---

## 1. What Polytopia AI Does

### Architecture

Two-tier system:
1. **Game layer** (C# mod) — hooks into the game client, serializes visible game state (map, units, cities, tech), executes validated commands
2. **AI backend** (TypeScript/Node.js) — receives state via HTTP, processes through Claude, returns action sequences

### Capability Roadmap

| Phase | Status | Capability |
|-------|--------|-----------|
| 1. Basic execution | Done | Turn-by-turn action generation |
| 2. Training data | In progress | Game log collection for future learning |
| 3. Memory | Planned | Cross-game learning, persistent strategy |
| 4. World model | Planned | Monte Carlo Tree Search, RL integration |

### Key Limitation

**The game is closed-source.** The project works by modding into a commercial game binary, which means:
- Can't modify game rules or create custom scenarios
- Dependent on game updates not breaking the mod
- Can't open-source the game environment itself
- Limited to the game's existing mechanics

---

## 2. Relevance to Loanville2

### LOW-MEDIUM — Architectural Patterns Only

The closed-source dependency significantly limits what we can learn. There's no benchmark we can run, no evaluation framework to compare against, and no game environment to fork. However, some architectural patterns are worth noting:

### a) Game State Serialization for LLM Consumption

Polytopia AI serializes complex game state (grid map, unit positions, resource counts, tech tree progress) into structured data that an LLM can reason about. Loanville2 does the analogous thing with financial dossiers — serializing borrower data (bank statements, quarterly income, narratives) into LLM-readable formats.

**Parallel**: Both projects face the same challenge of representing rich, structured state in a format that fits within an LLM context window while preserving decision-relevant information.

### b) Phased Capability Roadmap

Polytopia AI's 4-phase roadmap (basic execution → data collection → memory → world models) is a useful template for Loanville2's evolution:

| Polytopia Phase | Loanville2 Equivalent | Status |
|----------------|----------------------|--------|
| Basic execution | Single-run evaluation | Done |
| Training data | Season replay logs, run logger | Done |
| Memory | Persistent scratchpad (now implemented) | Done |
| World model | Models that predict borrower outcomes before deciding | Future |

The "world model" phase is interesting: could a lender model build an internal simulator that predicts loan outcomes before committing capital? This would be analogous to Monte Carlo planning in games.

### c) Action Validation

Polytopia AI validates LLM-generated actions before executing them in the game. Loanville2 does this via term sheet normalization in `run_to_decision()` — clamping amounts to policy bounds, normalizing APR ranges, ensuring valid term lengths.

---

## 3. Key Differences

| Aspect | Polytopia AI | Loanville2 |
|--------|-------------|------------|
| Game type | Turn-based strategy (spatial) | Financial simulation (analytical) |
| Game source | Closed-source commercial | Open-source custom |
| Agent model | Claude Sonnet 4 (single) | Any model via OpenRouter (competitive) |
| Evaluation | Win/loss vs AI opponents | RAROC scoring + Elo tournament |
| Multi-agent | Implicit (game AI opponents) | Explicit (LLM vs LLM) |
| Reproducibility | Low (game updates, mod fragility) | High (deterministic seeds, mock mode) |

---

## 4. Takeaway

Limited direct applicability due to the closed-source game dependency. The project is more of a proof-of-concept for "LLMs can play complex strategy games" than a usable benchmark or evaluation framework. The phased capability roadmap is a useful reference for thinking about Loanville2's evolution, but there are no concrete features to adopt.

The strongest lesson is a negative one: **building on an open, controllable simulation (like Loanville2) is much more valuable for benchmarking than modding into closed-source games**, because you control the rules, can create custom scenarios, and can guarantee reproducibility.
