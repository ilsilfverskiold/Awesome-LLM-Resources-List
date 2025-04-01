from langgraph.graph import MessagesState

# We pass it MessagesState that are already defined
class MultiAgentState(MessagesState):
    """State for the hierarchical agent system."""
    next: str = ""  # Next agent to run
