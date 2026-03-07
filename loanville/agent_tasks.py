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
}


def get_task(name: str) -> AgentTask:
    """Get a task by name. Defaults to simple_underwrite."""
    return TASKS.get(name, TASKS["simple_underwrite"])


def list_tasks() -> list[str]:
    return list(TASKS.keys())
