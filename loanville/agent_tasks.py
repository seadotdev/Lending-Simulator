"""
Task definitions for the agentic LOS simulation.

Progressive difficulty:
1. simple_underwrite — process one application through the LOS
2. incomplete_application — identify gaps, request docs
3. portfolio_batch — manage 5 deals, sector concentration
4. deal_update — find existing deal, add new docs, re-evaluate
5. complex_structure — multi-entity corporate graph
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentTask:
    """A task for the agent to complete."""
    name: str
    description: str
    task_prompt: str
    min_api_calls: int  # minimum reasonable API calls for efficiency scoring
    expected_artifacts: list[str] = field(default_factory=list)  # what should exist in LOS after
    difficulty: str = "easy"  # easy, medium, hard


TASKS: dict[str, AgentTask] = {
    "simple_underwrite": AgentTask(
        name="simple_underwrite",
        description="Process a single loan application through the full LOS pipeline",
        task_prompt=(
            "Process this borrower's loan application through the LOS. "
            "Create the entity, deal, upload financial documents, create a spread, "
            "advance through stages, and make your underwriting decision. "
            "You should approve, decline, or counter the application based on the financials."
        ),
        min_api_calls=8,  # entity + deal + 2 docs + spread + outcome + 2 stage advances
        expected_artifacts=["entity", "deal", "document", "spread"],
        difficulty="easy",
    ),
    "incomplete_application": AgentTask(
        name="incomplete_application",
        description="Handle an application with missing or insufficient documentation",
        task_prompt=(
            "Process this borrower's loan application. Note: the financial data may be "
            "incomplete or inconsistent. Identify any gaps in the application, document "
            "what additional information would be needed, and make a decision. "
            "If the data is insufficient, you should decline or refer the application "
            "with clear reasoning about what's missing."
        ),
        min_api_calls=6,
        expected_artifacts=["entity", "deal"],
        difficulty="medium",
    ),
    "portfolio_batch": AgentTask(
        name="portfolio_batch",
        description="Manage multiple deals with portfolio concentration awareness",
        task_prompt=(
            "You have 5 loan applications to process. For each one, create the entity "
            "and deal in the LOS. Be mindful of sector concentration — your portfolio "
            "should not be overexposed to any single sector. Process all 5 and make "
            "individual decisions, considering how each new loan affects your overall "
            "portfolio risk."
        ),
        min_api_calls=30,  # 5 * (entity + deal + docs + spread + stages + decision)
        expected_artifacts=["entity", "deal", "document", "spread"],
        difficulty="hard",
    ),
    "deal_update": AgentTask(
        name="deal_update",
        description="Find an existing deal and update it with new information",
        task_prompt=(
            "There is an existing deal in the LOS for this borrower. Find it using "
            "the deal list, add the new financial documents provided, re-run the "
            "spread analysis, and re-evaluate the deal with the updated information."
        ),
        min_api_calls=6,  # list + get + doc upload + spread + evaluate
        expected_artifacts=["document", "spread"],
        difficulty="medium",
    ),
    "complex_structure": AgentTask(
        name="complex_structure",
        description="Handle a multi-entity corporate structure with guarantees",
        task_prompt=(
            "This is a complex corporate borrower with multiple related entities. "
            "Create the full entity graph in the LOS: parent company, subsidiary "
            "(the borrower), and guarantor. Establish ownership and guarantee "
            "relationships. Then process the loan application for the subsidiary, "
            "noting the parent guarantee in your analysis."
        ),
        min_api_calls=12,  # 3 entities + 2 relationships + deal + docs + spread + stages
        expected_artifacts=["entity", "deal", "document", "spread", "relationship"],
        difficulty="hard",
    ),
    # --- Budget experiment tasks ---
    "multi_app_allocation": AgentTask(
        name="multi_app_allocation",
        description="Allocate a fixed LOS call budget across multiple borrower applications",
        task_prompt=(
            "You have a SHARED LOS call budget across multiple applications in this run. "
            "Each LOS tool call costs 1 credit (los_deal_evaluate costs 2; los_quick_assess is FREE). "
            "Your score is the AVERAGE accuracy across all applications — not just this one. "
            "\n\n"
            "Strategy: use los_quick_assess (free) to triage first. If the screen is clearly "
            "pass/fail, spend fewer paid credits. Save credits for borderline cases. "
            "A model that burns all credits on the first application and guesses for the rest "
            "will score much lower than one that spreads ~4 credits per application."
            "\n\n"
            "Process this borrower's application. Create the entity, deal, upload financials, "
            "create a spread, and make your decision — but budget your LOS calls carefully."
        ),
        min_api_calls=3,  # minimum viable: entity + deal + quick assess
        expected_artifacts=["entity", "deal"],
        difficulty="medium",
    ),
    "explicit_allocation": AgentTask(
        name="explicit_allocation",
        description="Explicit cost/benefit allocation — choose actions from a priced menu",
        task_prompt=(
            "You have a FIXED call budget for this application. Choose your actions wisely.\n\n"
            "## Action Cost Menu\n"
            "- los_quick_assess ............. FREE  (noisy, ±30% error rate)\n"
            "- los_entity_create ............  1 credit\n"
            "- los_deal_create ..............  1 credit\n"
            "- los_doc_upload ...............  1 credit each\n"
            "- los_spread_create ............  1 credit\n"
            "- los_deal_advance .............  1 credit\n"
            "- los_deal_evaluate ............  2 credits (reliable, ±5% error)\n"
            "- agent_done ...................  FREE\n\n"
            "## Optimal pattern (4-credit budget)\n"
            "1. los_quick_assess (free) — triage: is this obvious pass/fail?\n"
            "2. los_entity_create + los_deal_create (2 credits) — establish the case\n"
            "3. los_spread_create (1 credit) — compute ratios\n"
            "4. agent_done — decide based on spread (skip evaluate if spread is clear)\n\n"
            "Process this borrower and make your underwriting decision."
        ),
        min_api_calls=3,
        expected_artifacts=["entity", "deal"],
        difficulty="medium",
    ),
}


def get_task(name: str) -> AgentTask:
    """Get a task by name. Defaults to simple_underwrite."""
    return TASKS.get(name, TASKS["simple_underwrite"])


def list_tasks() -> list[str]:
    return list(TASKS.keys())
