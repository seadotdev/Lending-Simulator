# Modal Infrastructure Evaluation for Loanville

## Context

Loanville is a CLI-based lending simulation benchmark where LLMs act as autonomous loan underwriters. The simulation evaluates models by broadcasting borrower applications to competing LLM lenders, adjudicating offers, and scoring results via RAROC and Elo tournaments.

Key operational characteristics:
- **Single sim**: 27-72 LLM evaluations (borrowers × lenders), ~$0.05-0.30 at current DeepSeek V3 rates
- **Season mode**: 10+ weeks where capital, portfolio exposures, and loan outcomes carry forward
- **Elo tournament**: 50+ matches generating 9M-50M tokens for model comparison
- **Current state**: No mid-run checkpointing. Only Elo has JSON-based resumption. Season state is in-memory only.
- **Provider**: OpenRouter (50+ models, $0.04-$75/1M tokens depending on model)

---

## Part 1: Running Simulations on Modal — The Snapshot/Branch Argument

### The Core Inefficiency Today

In season mode, every experimental variation (different model, different capital config, different sector limits) requires re-running the *entire* season from week 1. A 10-week season takes 5-50 minutes depending on model and data mode. If you want to compare how 10 different models handle the late-season stress of a maturing portfolio, you burn 10× the full-season cost and time — even though weeks 1-5 might be identical setup.

The Elo tournament has the same problem at scale: 50 matches × full sim runs, with no ability to branch from shared initial conditions.

### What Modal Snapshots Enable

Modal offers three snapshot types that map directly to Loanville's needs:

#### 1. Memory Snapshots (the biggest unlock)

Memory snapshots capture a sandbox's *entire state* — running processes, in-memory objects, filesystem — and can restore an exact clone later. This means:

- **Run a season to week 5 → snapshot → fork into N branches**, each continuing with a different model, capital limit, or risk policy
- Every branch starts from the *exact same portfolio state* — same booked loans, same exposure levels, same capital utilization — making comparisons apples-to-apples
- Instead of 10 full seasons (10 × 10 weeks = 100 week-equivalents), you run 5 shared weeks + 10 × 5 forked weeks = **55 week-equivalents** — a 45% reduction
- With deeper branching (snapshot at week 3, 5, and 8), the savings compound further

#### 2. Filesystem Snapshots (persistence + long-running)

Modal sandboxes have a 24-hour limit. For extended tournament runs:

- Snapshot filesystem state at any point, spin up a new sandbox from that image
- Enables multi-day Elo tournaments without losing progress
- Only stores the diff from base image, so storage is cheap

#### 3. Directory Snapshots (hot-swap configurations)

- Mount different model configs or borrower pools into an already-running sandbox
- Swap the data layer without restarting the simulation engine
- Useful for A/B testing borrower mixes against the same lender state

### Concrete Use Cases

| Scenario | Without Modal | With Modal Snapshots |
|----------|--------------|---------------------|
| Compare 10 models over a full season | 10 × full season runs | 1 shared run to midpoint → 10 forked continuations |
| Test 5 capital configs late-season | 5 × full season runs | 1 run to week 8 → 5 forks for final 2 weeks |
| Elo tournament with 50 matches | 50 × independent full sims | Snapshot common borrower state → fan out matches |
| Find where a model fails | Re-run entire season per tweak | Snapshot before failure week → iterate fast |
| Stress test a single week's cohort | Re-run entire season to reach it | Snapshot week N-1 → replay week N with 20 variations |

### The Math: Efficiency at Scale

**Example: 10-model comparison over a 10-week season**

*Today*: 10 models × 10 weeks × 5 borrowers/week × 3 lenders = **1,500 LLM evaluations**

*With Modal branching at week 5*:
- Shared phase (1 model, weeks 1-5): 5 × 5 × 3 = 75 evaluations
- Branched phase (10 models, weeks 6-10): 10 × 5 × 5 × 3 = 750 evaluations
- **Total: 825 evaluations (45% fewer)**
- Plus: the first 5 weeks produce *identical* portfolio conditions, so the comparison is scientifically cleaner

**Example: Parameter sweep (5 capital configs × 5 models × 10-week season)**

*Today*: 25 full runs = 25 × 150 = **3,750 evaluations**

*With Modal (snapshot at week 5, branch both dimensions)*:
- 1 base run to week 5: 75 evaluations
- 25 forked runs for weeks 6-10: 25 × 75 = 1,875 evaluations
- **Total: 1,950 evaluations (48% fewer)**

### Cost Estimate

Modal CPU sandbox pricing: ~$0.000463/sec for 1 vCPU. A Loanville sim is CPU-bound (waiting on API calls), so lightweight compute suffices.

- A 10-minute season run costs ~$0.28 on Modal compute + LLM API costs
- Snapshot/restore overhead: sub-second (memory snapshots), negligible cost
- The real savings come from **fewer LLM API calls** (the dominant cost), not compute

At scale (100-run parameter sweep), branching from snapshots could save **$15-75 in LLM costs** per sweep, depending on model pricing. The Modal compute overhead is single-digit dollars.

### What Would Need to Change in the Codebase

1. **Serialize season state**: `SeasonEngine` and `SeasonLenderState` currently live in-memory. With memory snapshots this isn't a problem — the snapshot captures the Python process as-is. But for filesystem snapshots, we'd want `season.py` to support pickle/JSON serialization of the engine state.

2. **Wrap execution in Modal sandbox**: The CLI entry point (`__main__.py`) would need a Modal wrapper — essentially a `modal.Sandbox` that runs the Python process, with hooks to snapshot at configurable week boundaries.

3. **Fan-out orchestration**: A new script to manage the snapshot → branch → collect results pattern. Modal's Python SDK makes this straightforward:
   ```python
   # Pseudocode
   sb = modal.Sandbox.create(image=base_image, ...)
   sb.exec("python", "-m", "loanville", "--season", "--weeks", "5")
   snapshot = sb.snapshot_filesystem()

   # Fork 10 branches
   for model in models:
       branch = modal.Sandbox.create(image=snapshot, ...)
       branch.exec("python", "-m", "loanville", "--season", "--resume-week", "6", "--model", model)
   ```

4. **Result collection**: Branches need to emit structured results (JSON) that a parent orchestrator aggregates into comparison tables.

---

## Part 2: Self-Hosted LLMs on Modal GPUs vs. OpenRouter

### The Case For Modal GPUs

**1. Cost advantage at high volume**

Self-hosted LLM inference breaks even vs. API providers at roughly **2M+ tokens/day**. A full Elo tournament (50 matches) generates 9M-50M tokens — potentially a single day's burst that crosses this threshold.

| Model class | OpenRouter cost/1M tokens | Estimated Modal self-hosted cost/1M tokens | Savings |
|------------|--------------------------|-------------------------------------------|---------|
| Small (7-8B) | $0.04-0.10 (input) | ~$0.02-0.05 (A10G, $1.10/hr) | 50-75% |
| Medium (70B) | $0.10-0.80 (input) | ~$0.15-0.40 (A100, $1.80-3.50/hr) | 0-50% |
| Frontier (Claude, GPT-4o) | $5-15 (input) | N/A (closed weights) | N/A |

The sweet spot is **medium open-weight models (30B-70B)** at sustained high throughput. Smaller models are already so cheap on OpenRouter that self-hosting saves pennies. Frontier models can't be self-hosted.

**2. Reproducibility**

- Fixed model weights = bit-identical inference across runs (same quantization, same seed)
- No provider-side model updates silently changing behavior mid-tournament
- Critical for benchmark integrity: if DeepSeek V3 updates on OpenRouter, historical comparisons break

**3. Fine-tuned and unreleased models**

- Test LoRA-adapted models not available on OpenRouter
- Run models the day weights drop on HuggingFace, before any provider picks them up
- Test quantization variants (INT4 vs FP16) as a benchmark dimension

**4. Scale-to-zero economics**

- Modal charges per-second — spin up 8× A100s for a 2-hour tournament, pay $28, done
- No idle costs between benchmark runs (unlike reserved instances)
- GPU memory snapshots mean cold-start is ~2s, not 20s

**5. Batching and throughput control**

- With vLLM on Modal, you control the batch size, concurrency, and scheduling
- Can saturate a GPU with all 72 evaluations from a sim, maximizing throughput/cost
- OpenRouter's rate limits and shared infrastructure mean you're subject to queueing

### The Case Against Modal GPUs

**1. Speed to test new models is the killer drawback**

This is the big one for a benchmark tool:

- OpenRouter: new model appears → change a string in config → run benchmark. **Minutes.**
- Modal self-hosted: download weights (20-100GB) → configure vLLM → test → deploy. **Hours to days.**
- The entire *point* of Loanville is rapid model comparison. Self-hosting creates friction at the core of the workflow.

**2. Loanville's per-run token volume is low**

- A single sim: ~150K tokens → **$0.05-0.30 on OpenRouter**
- A 10-week season: ~375K tokens → **$0.25 on OpenRouter**
- Even a 50-match Elo tournament: 9M-50M tokens → **$30-150 on OpenRouter**
- Self-hosting breakeven requires 2M+ tokens/day *sustained*. Loanville runs are bursty, not continuous.

**3. Operational complexity**

- Managing model weights, quantization configs, GPU memory — all orthogonal to the benchmark itself
- vLLM version updates, CUDA compatibility, OOM errors on 70B models
- When a run fails, is it the model's underwriting quality or an infra issue? Debugging ambiguity.

**4. Model coverage gap**

- OpenRouter: 300+ models including Claude, GPT-4o, Gemini, Mistral, Cohere, etc.
- Self-hosted: Open-weight only (Llama, DeepSeek, Qwen, Mistral). No Claude. No GPT-4o.
- Loanville's value as a benchmark comes partly from comparing *across* model families — including closed models.

**5. Cold start and latency**

- Even with GPU memory snapshots (2s cold start), first-request latency is higher than OpenRouter
- For interactive/demo use, this is noticeable
- For batch benchmark runs, it amortizes — but adds complexity

**6. The models that benefit most are already cheap**

- Open-weight models where self-hosting helps (Llama 70B, DeepSeek, Qwen) are the cheapest on OpenRouter too
- The expensive models (Claude Opus at $15/$75 per 1M) are closed-weight — can't self-host anyway
- Marginal cost savings on cheap models may not justify the operational burden

### Recommendation: Hybrid Approach

The sweet spot is probably **OpenRouter as the default, with Modal GPU as an optional backend for specific scenarios**:

| Scenario | Best option | Why |
|----------|------------|-----|
| Day-to-day model comparison | OpenRouter | Speed, breadth, simplicity |
| Elo tournament (50+ matches) | Modal GPU for open-weight models | Cost savings at scale |
| Fine-tuned model evaluation | Modal GPU (only option) | Weights not on any provider |
| New model day-of-release | Modal GPU (if open-weight) | Available before providers |
| Closed-model benchmarks | OpenRouter (only option) | Claude, GPT-4o, Gemini |
| Reproducibility-critical runs | Modal GPU | Fixed weights, no silent updates |
| Quick demo / single run | OpenRouter | No cold start, no setup |

### What a Modal LLM Backend Would Look Like

```python
# In llm.py, add a Modal vLLM endpoint option
MODAL_ENDPOINT = "https://your-app--vllm-serve.modal.run/v1"

# Switch between providers based on config
if provider == "modal":
    client = openai.AsyncOpenAI(base_url=MODAL_ENDPOINT, api_key=modal_token)
elif provider == "openrouter":
    client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
```

Since both Modal/vLLM and OpenRouter expose OpenAI-compatible APIs, switching between them is a base_url change. The tool-use protocol and response parsing remain identical.

---

## Summary

| Dimension | Simulation Snapshots on Modal | Self-hosted LLMs on Modal GPUs |
|-----------|------------------------------|-------------------------------|
| **Primary value** | Branch-and-compare from shared state | Cost savings at high volume |
| **Effort to implement** | Medium (sandbox wrapper + orchestration) | Low (base_url swap) |
| **Risk** | Low (Modal is additive, not replacing anything) | Medium (operational burden) |
| **When it pays off** | Parameter sweeps, multi-model season comparisons | 50+ match tournaments, fine-tuned models |
| **When it doesn't** | Single ad-hoc runs | Day-to-day model comparison, closed models |
| **Recommendation** | **Strong yes** — biggest ROI for benchmark workflows | **Selective** — use for tournaments and custom models only |

The snapshotting capability is the more compelling case. It addresses a real structural inefficiency in how seasons and tournaments work today, and the savings scale multiplicatively with the number of experimental variations. The LLM hosting story is more nuanced — OpenRouter's breadth and zero-ops experience is hard to beat for a benchmark tool, but Modal GPUs carve out a clear niche for high-volume tournament runs and models that aren't available through API providers.

---

## Sources

- [Modal Sandbox Snapshots Documentation](https://modal.com/docs/guide/sandbox-snapshots)
- [Modal Directory Snapshots Blog Post](https://modal.com/blog/directory-snapshots-resumable-project-state-for-sandboxes)
- [Modal Memory Snapshots Blog Post](https://modal.com/blog/mem-snapshots)
- [Modal GPU Memory Snapshots](https://modal.com/blog/gpu-mem-snapshots)
- [Modal Pricing](https://modal.com/pricing)
- [Modal vLLM Deployment Guide](https://modal.com/blog/how-to-deploy-vllm)
- [Modal LLM Engineer's Almanac](https://modal.com/llm-almanac/advisor)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [OpenRouter Models](https://openrouter.ai/models)
- [Modal NVIDIA A10G Pricing](https://modal.com/blog/nvidia-a10g-price-article)
- [Serverless vs Dedicated LLM Deployments](https://www.bentoml.com/blog/serverless-vs-dedicated-llm-deployments)
