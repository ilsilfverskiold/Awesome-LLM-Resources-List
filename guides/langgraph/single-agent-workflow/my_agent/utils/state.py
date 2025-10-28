from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from typing_extensions import TypedDict

# State schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
