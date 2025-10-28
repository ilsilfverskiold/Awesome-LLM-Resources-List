from langgraph.graph import StateGraph, START, END
from my_agent.utils.nodes import (
    supervisor_node, 
    research_supervisor_node, trending_keywords_node, top_keywords_node, search_keywords_node, github_keywords_node,
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
workflow.add_node("top_keywords_agent", top_keywords_node)  
workflow.add_node("keyword_search_agent", search_keywords_node)
workflow.add_node("trending_github_repos_agent", github_keywords_node)

# Editing team nodes
workflow.add_node("editing_supervisor", editing_supervisor_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("summarizer", summarizer_node)

# Only need the starting edge
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()
