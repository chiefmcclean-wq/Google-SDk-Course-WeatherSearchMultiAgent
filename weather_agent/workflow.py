"""Multi-Agent Workflow Module for Google ADK.

Defines the multi-agent workflow architecture comprising:
1. Root Workflow Agent ('root_agent'): Coordinates routing and loop-backs.
2. Specialized Weather Agent ('weather_agent'): Handles weather forecasts and alerts.
3. Specialized Search Agent ('search_agent'): Queries Google Search for web information.

Workflow Routing & Loop-back Mechanics:
- If a user sends a weather request, the root agent delegates to 'weather_agent'.
- If a user sends a general knowledge/search request, the root agent delegates to 'search_agent'.
- If 'weather_agent' or 'search_agent' cannot answer a question (out-of-domain or insufficient data),
  the sub-agent loops back to 'root_agent' using ADK's Agent-to-Agent (A2A) transfer protocol.
- The root agent evaluates the loop-back reason and re-routes the task to the appropriate agent.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger("weather_agent.workflow")


def get_workflow_metadata() -> Dict[str, Any]:
    """Returns metadata describing the multi-agent workflow."""
    return {
        "architecture": "Hierarchical Multi-Agent with Bidirectional Loop-Back",
        "root_agent": "root_agent",
        "sub_agents": ["weather_agent", "search_agent"],
        "loop_back_enabled": True,
        "transfer_targets": {
            "root_agent": ["weather_agent", "search_agent"],
            "weather_agent": ["root_agent", "search_agent"],
            "search_agent": ["root_agent", "weather_agent"],
        },
        "description": (
            "Multi-agent workflow consisting of root coordinator, weather specialist, "
            "and Google search specialist. Sub-agents loop back to root if unable to answer."
        ),
    }


def append_to_state(tool_context, field: str, response: Any) -> Dict[str, str]:
    """Maintains workflow state history across iterative agent loops.

    Args:
        tool_context: ADK ToolContext instance.
        field (str): The state dictionary key to update.
        response (Any): Content to append to the state field.

    Returns:
        Dict[str, str]: Status dictionary.
    """
    existing_state = tool_context.state.get(field, [])
    if isinstance(existing_state, list):
        tool_context.state[field] = existing_state + [response]
    else:
        tool_context.state[field] = [existing_state, response]
    return {"status": "success"}
