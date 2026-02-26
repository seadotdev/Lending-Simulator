# Game Visualizer & LLM Trace Research

Research into game visualizers, stats dashboards, and LLM trace tools — with focus on what Loanville can learn from the ecosystem.

---

## 1. LLM Skirmish (llmskirmish.com)

**What it is**: A competitive benchmark where LLMs write JavaScript code to play 1v1 RTS (real-time strategy) games against each other. Built on the Screeps open-source API. Each match runs up to 2,000 game ticks; LLMs adapt strategies across 5-round tournaments.

**Tech stack**: TypeScript monorepo (pnpm + Turbo), packages for `engine`, `replay`, `maps`, `types`. CLI tool published on npm.

**Visualizer**:
- `skirmish view [id|file]` opens a browser-based replay viewer
- Match data exported as JSONL (JSON Lines) format
- `packages/replay/` handles replay log parsing
- Built on top of the Screeps rendering pipeline (likely WebGL/PixiJS)
- Replays show unit movement, spawn creation, combat actions tick-by-tick

**Stats/leaderboard**:
- ELO rating system (Claude Opus 4.5 achieved highest ELO of 1778)
- Per-round cost tracking ($4.12/round for top model)
- In-context learning evaluation (strategies evolve across 5 rounds)

**Can we use their visualizer directly?** No — the mechanics are fundamentally different. Skirmish visualizes grid-based RTS combat (units moving, attacking, spawning on a map), while Loanville simulates a lending market (borrowers arriving, lenders making decisions, loans resolving over weeks). The data models don't overlap.

**What we can learn**:
- **JSONL replay format** — clean separation between game engine and viewer. Skirmish exports tick-by-tick game state as JSONL, which the viewer consumes independently. Loanville already does this with season JSON, which is a good pattern.
- **CLI → browser bridge** — the `skirmish view` command that auto-opens the browser with replay data is a nice UX pattern we could replicate (`python -m loanville view <season.json>`).
- **In-context learning tracking** — they track how LLMs adapt across rounds. We could add similar "learning curve" visualization showing how models adjust pricing/risk assessment across weeks in season mode.

---

## 2. Screeps Open-Source Renderer

**Source**: https://blog.screeps.com/2018/08/renderer/

The Screeps game open-sourced their renderer — a WebGL/PixiJS-based graphics engine that can be embedded in any web app to display game objects. This is what Skirmish builds on.

**What we can learn**:
- Screeps uses **PixiJS** for GPU-accelerated 2D rendering in the browser. Our Three.js 3D town is more complex but also heavier. For a simpler "dashboard" view, a PixiJS-based 2D visualization might be lighter and more information-dense.
- Their renderer is a standalone library that takes game state objects and renders them — clean separation of concerns. Our `town-scene.js` could similarly be extracted into a reusable component.

---

## 3. GamingAgent (ICLR 2026)

**Source**: https://github.com/lmgame-org/GamingAgent

**What it is**: A research framework for evaluating LLM/VLM performance in interactive gaming environments (Pokemon Red, Super Mario Bros, 2048, Tetris, Candy Crush, Sokoban). Published at ICLR 2026.

**Visualizer**: Generates **replay videos** from textual game state representations + agent episode logs. Command-line tool with configurable frame rate and output quality.

**What we can learn**:
- **Video export from logs** — they convert text-based game states into replay videos. We could generate video exports from our season JSON data, making it easy to share results on social media / in papers without requiring the interactive viewer.
- **Standardized environment interface** (Gymnasium) — makes it easy to plug in new games. Our season engine could expose a similar interface for RL training.

---

## 4. LLM Game Bench (llmgamebench.com)

**What it is**: First open-source framework for evaluating AI visual understanding, reasoning, and decision-making through classic game environments. Launched with a Pokemon Red benchmark.

**Approach**: Uses mGBA emulator → Lua script captures screenshots → sends to LLM → receives button press commands → executes. The LLM plays the game by "seeing" screenshots.

**What we can learn**:
- Their leaderboard shows model comparisons with rich per-game breakdowns
- Focus on **visual** understanding benchmarks — a direction we haven't explored (could we render financial statements as images and test vision models?)

---

## 5. LLM Trace Visualization Tools

### Langfuse (Open Source, Self-Hosted)

**Source**: https://langfuse.com / https://github.com/langfuse/langfuse

The gold standard for LLM trace visualization. Key features:

- **Nested trace trees** — each trace shows the full execution tree: LLM calls, tool invocations, retrieval steps, custom logic, all nested hierarchically
- **Agent graphs** — visualize multi-step agent workflows as directed graphs
- **Session grouping** — group traces into sessions (multi-step conversations/workflows)
- **Cost & latency dashboards** — per-trace and aggregate cost/token/latency metrics
- **Evaluation scores** — attach human/automated eval scores to traces
- **Prompt management** — version, test, and deploy prompts collaboratively
- **OpenTelemetry integration** — not locked into their SDK

**Relevance to Loanville**: Our existing trace viewer (in `web/app.js`, tab 2) shows LLM prompts, tool calls, and JSON responses. Langfuse's approach suggests we could add:
- **Hierarchical trace view** — show the full decision tree per borrower (system prompt → tool calls → jq queries → final decision) as a collapsible tree
- **Cost tracking per decision** — token usage and $ cost for each lending decision
- **Aggregate dashboards** — average tokens/cost per model, per decision type (approve vs reject)
- **Session view** — group all decisions within a season week as a "session"

### Braintrust

**Key feature**: **Chain-of-thought visualization** for reasoning models. When an agent fails at step 8 of 19, timeline replay shows the exact sequence, and chain-of-thought visualization shows model reasoning at each decision point.

**Relevance**: For models using extended thinking (Claude with thinking mode, o3, etc.), we could visualize the thinking trace separately from the tool calls, showing the model's internal reasoning about borrower creditworthiness.

### Weights & Biases (W&B) Weave

**Key feature**: Organizes logs into a **trace tree** with latency and cost automatically aggregated at every level. Can inspect detailed agent trajectories at every step.

**Relevance**: The trace tree with aggregated metrics at each level is a pattern we could adopt for our trace viewer.

### Other Notable Tools
- **LangSmith** — hierarchical trace capture from LangChain, good for multi-tool agent debugging
- **Datadog LLM Observability** — enterprise-grade, end-to-end tracing across agents
- **OpenLLMetry** — open-source, OpenTelemetry-based, works with Anthropic/OpenAI/vector DBs
- **DeepEval** — `@observe` decorator pattern for tracing + evaluation

---

## 6. General Leaderboard / Stats Dashboards

### LLM Stats (llm-stats.com)
- Interactive comparison tables with 50+ benchmarks
- Advanced filters (by capability, model family, cost tier)
- Clean, dense information design

### SEAL Leaderboard (Scale AI)
- Expert-driven evaluations
- Category-specific rankings (coding, reasoning, etc.)
- Clean multi-tab interface

### Chatbot Arena (lmsys.org)
- ELO-based rankings from human preferences
- Head-to-head comparison tool
- Confidence intervals and statistical significance

---

## Recommendations for Loanville

### Quick Wins (leverage what we already have)

1. **CLI viewer command** — Add `python -m loanville view <season.json>` that auto-opens the web viewer with data pre-loaded (like Skirmish's `skirmish view`)

2. **Cost tracking in traces** — Add token usage and cost per LLM call to our trace data, display in the trace viewer tab

3. **Hierarchical trace tree** — Restructure our trace viewer to show a collapsible tree: Season → Week → Borrower → LLM Call → Tool Uses → Decision. Inspired by Langfuse's nested observation model.

4. **Video/GIF export** — Add a "record" button to the 3D viewer that captures frames and exports as video/GIF for sharing (inspired by GamingAgent's replay video generation)

### Medium-Term Improvements

5. **Aggregate stats dashboard** — A "Stats" tab alongside "Game" and "LLM Trace" showing:
   - Model comparison radar charts (RAROC, fraud detection rate, approval rate, avg cost)
   - Per-week trajectory charts (already in console output, could be visualized)
   - Confusion matrix heatmaps (interactive, click to see underlying decisions)

6. **Chain-of-thought viewer** — For models with thinking/reasoning traces, show the thinking separately (inspired by Braintrust). Display the model's internal reasoning about each borrower alongside the actual decision.

7. **Learning curve visualization** — Track how models adapt across season weeks (inspired by Skirmish's 5-round tournament format). Show pricing trends, risk calibration changes, and strategy evolution.

### Longer-Term / Exploratory

8. **Langfuse integration** — Instrument our LLM calls with Langfuse/OpenTelemetry for production-grade trace collection. This would give us the full Langfuse UI "for free" alongside our custom visualizations.

9. **2D dashboard view** — A PixiJS or D3-based 2D alternative to the 3D town, optimized for information density rather than spectacle. Show the lending market as a flow diagram: borrowers on one side, lenders on the other, loans as animated connections.

10. **Shareable replay links** — Host season replays as static JSON + the viewer on a CDN, generate shareable URLs (like `loanville.dev/replay/abc123`).
