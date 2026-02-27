# Loanville Announcement Strategy

**Applying Braley's "Engineer the Moment" Framework**

---

## Step 1: Who Are We Talking To?

Based on your priorities, the ranked audiences are:

| Rank | Audience | Why | What They Care About |
|------|----------|-----|---------------------|
| **1** | **ARIA grant reviewers / research funders** | Direct funding opportunity; "Scaling Trust Arena" alignment is strong | Multi-agent coordination, open-source infrastructure, measurable trust signals, novel research contribution |
| **2** | **ML/RL researchers at labs & frontier companies** | Customer pipeline for RL env; acquisition intros | Training data quality, environment realism, Gymnasium compatibility, dense reward signals, benchmark credibility |
| **3** | **AI-native fintech builders & VCs** | Raise visibility; signal sophistication; attract intros | Market insight, defensible IP, team calibre, "why lending + AI now" narrative |

**Primary audience for announcement sequence: ML/RL researchers who are thinking about financial services.**

Rationale: This audience is the intersection that serves all your goals. ARIA reviewers are ML researchers. Labs looking for RL environments employ ML researchers. VCs follow what ML researchers are excited about. Acquirers want teams that ML researchers respect.

The CSO/lender customer audience is a different motion entirely and shouldn't dilute this launch.

### The Person We're Writing For

A senior research engineer or applied scientist at a frontier lab (Anthropic, OpenAI, DeepMind, Cohere, Mistral) or RL-focused startup (Scale AI, Imbue, Nous Research) who:
- Has tried building RL environments and knows most are toy problems
- Is frustrated that financial services benchmarks are either synthetic tabular data or regulatory compliance checklists
- Follows AI Twitter/X and reads arXiv, but doesn't read fintech press
- Would forward something interesting to their team Slack if it felt novel and credible

---

## Step 2: Message & Why Anyone Should Care

### The "Inside Baseball" Trap to Avoid

The natural instinct is to lead with: "We built a 36-borrower simulation with RAROC scoring and triple-Elo ratings." That's the feature list. It's what you're proudest of. But it fails the Braley test: *people can't care about your solution until they understand the change you're making.*

### One-Line Headline (Provocative, Clear, Implies Stakes)

**Option A (recommended):**
> We built a town full of AI loan officers and made them compete. Here's who went bankrupt.

**Option B (more technical):**
> The first open-source RL environment where AI models underwrite real commercial loans against each other

**Option C (shorter, punchier):**
> Which AI model is the best loan officer? We ran the tournament.

### One-Paragraph Narrative (Why You, Why This, Why Now)

> Every frontier lab is racing to make models better at reasoning. But how do you test whether a model can actually *think* about risk — not in a textbook way, but the way a real underwriter does: reading 12 months of bank statements, catching circular transfers buried in transaction data, pricing a loan competitively while managing a portfolio? We couldn't find a benchmark that tested this, so we built one. Loanville is a fully simulated commercial lending market where AI models compete head-to-head as autonomous underwriters — analyzing real-world-fidelity financial dossiers, bidding on deals, managing portfolios, and getting scored on risk-adjusted returns. It's open-source, Gymnasium-compatible, and the leaderboard already has 40+ models ranked across profit, credit quality, and market share. Turns out, the best underwriter isn't who you'd expect.

### Three Proof Points (Evidence That Makes This Real)

| # | Proof Point | Evidence | Why It Works |
|---|------------|---------|-------------|
| **1** | **40+ models benchmarked, leaderboard is live** | Link to leaderboard with Elo ratings, confusion matrices, cost-per-decision analysis across DeepSeek, Llama, GPT-4.1, Claude, Qwen, Gemini, etc. | Concrete, not vaporware. Researchers can immediately see where their model sits. |
| **2** | **Catches real analytical gaps in frontier models** | Share 2-3 specific findings: e.g., "Model X approved all 4 fraud cases," "The cheapest model outperformed the most expensive on credit accuracy," "No model caught the over-leverage trap where DSCR < 1.0" | This is the "maybe this is real" signal. Specific findings > general claims. |
| **3** | **Designed as an RL training environment, not just a benchmark** | Gymnasium-compatible interface, dense RAROC rewards, competitive self-play, multi-week seasons with carry-forward state. Cite the Scale AI vending machine comparison — Loanville is structurally more sophisticated. | Signals that this isn't a one-off blog post but research infrastructure. Gives the audience a reason to engage (they can use it). |

---

## Step 3: Distribution — Getting It In Front of Them

### The Coordinated Set (Tight Window: Same Day)

| Asset | Channel | Purpose | Timing |
|-------|---------|---------|--------|
| **Owned: Technical blog post / GitHub README** | GitHub + personal site/blog | Depth. Full methodology, architecture, findings, how to use it. The thing people link to. | Anchor — publish first |
| **Earned: One targeted amplifier** | Pick ONE: (a) a relevant Substack/newsletter (e.g., The Gradient, Import AI, Interconnects), or (b) a podcast guest slot, or (c) ARIA application itself as the "earned" channel | Credibility + reach to exact audience | Same day or within 48h |
| **Native social: X/Twitter thread** | X | Virality. The teaser video + the hook. This is where the "moment" happens. | Same morning as blog post |

### X/Twitter: The Anchor Post

This is your highest-leverage single asset. Here's the structure:

---

## The Posts

### Post 1: X/Twitter Thread (Primary Distribution)

**Tweet 1 (Hook + Video):**
> We built a simulated town where AI models compete as commercial loan officers.
>
> They read real bank statements. They price loans. They manage portfolios. They try to catch fraud.
>
> Some of them are really bad at it.
>
> Here's how 40+ models performed ↓
>
> [Attach: teaser video of the town visualization / season running]

**Tweet 2 (The Unexpected Finding):**
> The most expensive model wasn't the best underwriter.
>
> [Model X] at $Y/decision outperformed [Model Z] at $W/decision on risk-adjusted returns.
>
> But credit accuracy told a different story — [finding about which models catch fraud vs which don't].
>
> Full leaderboard: [link]

**Tweet 3 (What Makes This Different):**
> Most AI benchmarks test knowledge or coding. We wanted to test *judgment under uncertainty* — the same skill that separates a good underwriter from a bad one:
>
> → Can you spot circular transfers in 12 months of bank data?
> → Will you reject a deal that looks great but has DSCR < 1.0?
> → How aggressively do you price when you're competing for every deal?

**Tweet 4 (The RL Angle — For the Researchers):**
> We also built this as a Gymnasium-compatible RL environment.
>
> Dense RAROC rewards. Multi-week seasons. Self-play. Competitive deal adjudication that prevents reward hacking.
>
> If you're training models on financial reasoning, this is the environment we wish existed. Now it does.
>
> Repo: [link]

**Tweet 5 (Open Source + CTA):**
> Everything is open source:
>
> → 36 hand-crafted business dossiers with realistic financials
> → Full loan origination system (Open LOS)
> → Elo tournament infrastructure
> → Season mode with carry-forward portfolios
>
> Run it yourself, benchmark your model, or use it for RL training.
>
> [GitHub link]
>
> We're also looking for collaborators — especially people working on RL for financial reasoning. DMs open.

---

### Post 2: LinkedIn (Secondary — More Narrative, Less Technical)

> **Which AI model is the best loan officer?**
>
> We spent the last few months building something unusual: a simulated commercial lending market where AI models compete head-to-head as autonomous underwriters.
>
> Each model gets the same borrower applications — 12 months of bank statements, quarterly income statements, business narratives. Some borrowers are legitimate. Some are overleveraged. Some are outright frauds with fabricated transactions.
>
> The models have to figure out which is which, price their loans competitively, manage portfolio concentration, and avoid going bankrupt.
>
> **Three things surprised us:**
>
> 1. **Cost ≠ quality.** The cheapest models weren't the worst underwriters, and the most expensive weren't the best. The relationship between model price and lending performance is far more jagged than you'd expect.
>
> 2. **Fraud detection is the hardest test.** Models that scored well on credit analysis often approved every fraud case. The skills are different — analytical rigor vs. pattern recognition in transaction data.
>
> 3. **Competition changes behavior.** When models compete for the same deals (borrowers pick the lowest rate), their strategies diverge in fascinating ways. Some become aggressive on pricing and win volume. Others become conservative and win on returns. It mirrors real lending markets.
>
> We open-sourced everything: the simulation, the borrower data, the scoring system, and a full loan origination system we built as the system of record.
>
> If you're working on AI in financial services, or you're training models and want a benchmark that tests real-world judgment (not just pattern matching), we'd love to hear from you.
>
> [Link to repo / blog post]

---

### Post 3: UW Benchmark Teaser (Can Publish First as a Standalone)

**X/Twitter:**
> We benchmarked 40+ LLMs on commercial loan underwriting — not multiple choice, but full dossier analysis with real bank statements.
>
> Quick findings thread:
>
> → Best overall: [model]
> → Best on fraud detection: [model]
> → Best cost/performance ratio: [model]
> → Worst trap: over-leverage (almost nobody catches it)
>
> Full methodology + results: [link]

---

### Post 4: Open LOS Teaser (Publish After Sim/Benchmark Posts)

**X/Twitter:**
> Side effect of building our lending simulator: we needed a loan origination system for the AI agents to interact with.
>
> So we built one. And open-sourced it.
>
> Open LOS: a modern, API-first loan origination system designed for AI-native lending workflows.
>
> It's what we wish every lender we work with was using. If you're an SME lender still running on HubSpot + spreadsheets, this is for you.
>
> [GitHub link]

---

## Step 4: Execution Timeline

| Week | Action | Notes |
|------|--------|-------|
| **1** | Finalize leaderboard data, pick the 3 most interesting findings for proof points | Need enough matches for statistical credibility. Fill in the [model] placeholders above with real data. |
| **1** | Align on which video clip to use as teaser | The town visualization video you mentioned. 15-30 seconds, no narration needed, just the visual + a subtitle. |
| **2** | Write the technical blog post (owned content) | This is the canonical reference. Methodology, findings, how to use it, RL environment spec. Pressure-test with 2-3 people. |
| **2** | Polish the GitHub README for public consumption | It's already good. Main gap: add a "Quick Start" that gets someone to a result in <5 minutes. |
| **3** | Post UW Benchmark teaser on X | Standalone value. Tests the water. See what resonates. |
| **3** | Revise ARIA application with links to public assets | |
| **4** | Publish main announcement: blog + X thread + LinkedIn simultaneously | Coordinate amplification with investors/network in the first 2 hours. |
| **4** | Open LOS post (same day, 2-3 hours after main post) | Position as "by the way, here's the LOS we built as a by-product" |

---

## Sequencing Rationale

The Braley framework says "pick 1 owned + 1 earned + 1 social." But you have three distinct assets (Sim, LOS, UW Bench) that each serve a different purpose. The recommended sequence:

1. **UW Benchmark first** (teaser) — lowest lift, most immediately useful to the ML audience, establishes credibility
2. **Simulator + full announcement** (main event) — this is the "moment." Video, thread, blog post, coordinated amplification
3. **Open LOS** (follow-up) — positioned as a by-product, which is more compelling than leading with it. "We needed a LOS so we built one" > "Here's our LOS"

This sequence builds on itself: the benchmark teaser creates awareness, the simulator announcement is the main event, and the LOS is the "wait, they also built this?" kicker.

---

## Success Metrics (Per Braley: Define Before You Launch)

| Audience | Metric | Target |
|----------|--------|--------|
| **ARIA / Grants** | Application submitted with public links; reviewer engagement | Application accepted to next stage |
| **ML Researchers** | GitHub stars, forks, issues opened; DMs/emails from lab researchers | 50+ stars first week; 3+ inbound conversations with lab people |
| **VCs** | Inbound intros or follow-up meetings referencing the announcement | 2+ warm intros from the content |
| **Acquirers** | Interest signals from fintech companies (Stripe/Plaid/GoCardless-type) | 1+ exploratory conversation |

---

## What to Veto / Risks

| Risk | Recommendation |
|------|---------------|
| **"AI Slop" trap** — the town visualization video could feel gimmicky if over-produced | Keep it raw. Screen recording > polished production. Let the complexity of what's actually happening be the hook. |
| **"Inside Baseball" trap** — leading with RAROC methodology and triple-Elo | Save for the blog post. Social posts lead with findings and competitive narrative. |
| **Spreading too thin** — posting Sim + LOS + Benchmark all at once | Sequence them. Each gets its own moment. |
| **Premature LOS positioning** — presenting it as a product when it's not production-ready | Frame as "open-source research infrastructure" not "SaaS alternative." The by-product positioning is stronger and more honest. |
| **Over-claiming on RL** — saying "Gymnasium-compatible" before the wrapper is built | Either build the thin wrapper first (it's a weekend of work based on the rl-environment-analysis.md spec), or say "designed for RL training" without claiming Gymnasium compatibility. |
| **Leaderboard credibility** — 51 matches may not be enough for stable Elo | Either run more matches before announcing, or frame as "early results" with the tournament ongoing. Transparency > false precision. |
