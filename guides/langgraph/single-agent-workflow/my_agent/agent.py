from typing import TypedDict, Literal

from my_agent.utils.state import AgentState
from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node


# Define config
class GraphConfig(TypedDict):
    model_name: Literal["openai", "gemini"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Simplified routing
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# Add edge back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
graph = workflow.compile()
