#read in the .env file amongst other things
from dotenv import load_dotenv
#load_dotenv(dotenv_path="/Users/allenshaw/Documents/Development/AI/Agentic-LLM-Ideas/guides/langgraph/tech_news_bot_multiagent/.env")
load_dotenv()

from langgraph.graph import StateGraph, START
from my_agent.utils.nodes import (
    supervisor_node, 
    research_supervisor_node, trending_keywords_node, top_keywords_node, search_keywords_node,
    editing_supervisor_node, fact_checker_node, summarizer_node
)
from my_agent.utils.state import MultiAgentState

# Create the main graph
workflow = StateGraph(MultiAgentState)

# Add all nodes 

workflow.add_node("supervisor", supervisor_node)

# Research team nodes
workflow.add_node("research_supervisor", research_supervisor_node)
workflow.add_node("trending_keywords_agent", trending_keywords_node)
print("12. trending keywords node call")
workflow.add_node("top_keywords_agent", top_keywords_node)  
workflow.add_node("keyword_search_agent", search_keywords_node)

# Editing team nodes
workflow.add_node("editing_supervisor", editing_supervisor_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("summarizer", summarizer_node)

# Only need the starting edge
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from my_agent.utils.nodes import supervisor_node
from my_agent.utils.state import MultiAgentState

def run_workflow():
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.set_entry_point("supervisor")
    initial_state = {
    "messages": [HumanMessage(content="Track what's trending in post-growth economics this week.")],
    "next": "supervisor"
    }
    # Compile the graph
    graph = workflow.compile()
    final_state = graph.invoke(initial_state)
    print("FINAL MESSAGE:")
    print(final_state["messages"][-1].content)

run_workflow()

