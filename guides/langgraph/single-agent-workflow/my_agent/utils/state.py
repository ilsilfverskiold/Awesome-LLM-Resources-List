from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

# State schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
