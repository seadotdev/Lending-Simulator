"""
Custom tool creation system for season mode.

Lenders can create reusable tools between evaluation weeks. Tools persist
across the season and can be used during borrower evaluation.
"""

from dataclasses import dataclass, field


@dataclass
class CustomTool:
    name: str
    description: str
    implementation: str  # e.g. bash script or logic description
    created_week: int
    times_used: int = 0


class LenderToolkit:
    """Per-lender tool registry persisting across season weeks."""

    MAX_TOOLS = 5
    CREATION_COST = 2   # tool-call equivalents charged to efficiency
    UPDATE_COST = 1

    def __init__(self, lender_id: str):
        self.lender_id = lender_id
        self.tools: list[CustomTool] = []

    def create_tool(
        self, name: str, description: str, implementation: str, week: int,
    ) -> CustomTool | str:
        """Create a tool. Returns the tool or an error string if at limit."""
        if len(self.tools) >= self.MAX_TOOLS:
            return f"Tool limit reached ({self.MAX_TOOLS})"
        # Prevent duplicate names
        if any(t.name == name for t in self.tools):
            return f"Tool '{name}' already exists. Use update_tool to modify."
        tool = CustomTool(
            name=name,
            description=description,
            implementation=implementation,
            created_week=week,
        )
        self.tools.append(tool)
        return tool

    def update_tool(
        self, name: str, description: str | None = None,
        implementation: str | None = None,
    ) -> CustomTool | str:
        """Update an existing tool. Returns the tool or error string."""
        for tool in self.tools:
            if tool.name == name:
                if description is not None:
                    tool.description = description
                if implementation is not None:
                    tool.implementation = implementation
                return tool
        return f"Tool '{name}' not found."

    def get_tool(self, name: str) -> CustomTool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_tool_definitions(self) -> list[dict]:
        """Export tools in function-calling format for LLM consumption."""
        definitions = []
        for tool in self.tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            })
        return definitions

    def record_usage(self, tool_name: str) -> bool:
        """Increment usage counter for a tool. Returns True if found."""
        for tool in self.tools:
            if tool.name == tool_name:
                tool.times_used += 1
                return True
        return False

    def total_usage(self) -> int:
        """Total times any custom tool was used."""
        return sum(t.times_used for t in self.tools)

    def tools_used_after_creation(self) -> int:
        """Count of tools that have been used at least once after creation."""
        return sum(1 for t in self.tools if t.times_used > 0)
