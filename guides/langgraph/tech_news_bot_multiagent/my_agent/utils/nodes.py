from typing import Literal, List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from my_agent.utils.tools import (
    trending_keywords_sources_tool, 
    top_keywords_sources_tool, 
    keyword_source_search_tool,
    read_notes,
    write_notes,
    append_notes,
    get_or_create_notes_file
)
from my_agent.utils.state import MultiAgentState
from langgraph.graph import END
from datetime import datetime

# Get today's date
today = datetime.now().strftime("%Y-%m-%d")

# -------------------- Change these if you want --------------------

# Create the LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
llm_big = ChatOpenAI(model="gpt-4o")
llm_even_bigger = ChatOpenAI(model="gpt-4.5-preview")
llm_biggest = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")

# -------------------- Supervisor nodes (don't touch) -----------------------

# Top level supervisor node
def make_top_level_supervisor_node(members: list[str], system_prompt: str) -> str:
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]
        instruction: str  

    def supervisor_node(state: MultiAgentState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router with authority to end the workflow."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        instruction = response["instruction"]
        
        if goto == "FINISH":
            goto = END

            formatted_summary = extract_final_summary()
            instruction = f"{instruction}\n\n# FINAL RESEARCH SUMMARY\n\n{formatted_summary}"
            
            return Command(
                goto=goto, 
                update={
                    "next": goto,
                    "messages": state["messages"] + [
                        AIMessage(content=instruction, name="supervisor")
                    ]
                }
            )

        return Command(
            goto=goto, 
            update={
                "next": goto,
                "messages": state["messages"] + [
                    HumanMessage(
                        content=f"[INSTRUCTION FROM MAIN SUPERVISOR]\n{instruction}",
                        name="supervisor"
                    )
                ]
            }
        )

    return supervisor_node

# Team supervisor node (used several times)
def make_team_supervisor_node(members: list[str], parent: str, system_prompt: str, team):
    options = ["FINISH"] + members

    class Router(TypedDict):
        next: Literal[*options]
        instruction: str 

    def team_supervisor_node(state: MultiAgentState) -> Command[Literal[*members, parent]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        instruction = response.get("instruction", "Please perform your task clearly without questions")

        if goto == "FINISH":
            goto = parent
            return Command(goto=goto, update={
                "next": goto, 
                "messages": state["messages"] + [
                    AIMessage(content=f"Research complete. Response from the {team} team supervisor: {instruction}")
                ]
            })

        return Command(
            goto=goto, 
            update={
                "next": goto,
                "messages": state["messages"] + [
                    HumanMessage(
                        content=f"[INSTRUCTION FROM {team} TEAM SUPERVISOR]\n{instruction}",
                        name="supervisor"
                    )
                ]
            }
        )

    return team_supervisor_node

# -------------------- RESEARCH TEAM --------------------

# Trending Agent

trending_keywords_prompt_template = """⚠️ IMPORTANT: YOU MUST USE trending_keywords_sources_tool EXACTLY ONCE, THEN use the write_notes tool to save EVERYTHING you find from the tools, THEN STOP ⚠️

You are an agent that can fetch trending keywords for tech social media platforms based on several categories (but no more than 4) 

You have these categories to choose from: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects. 
You have these periods to choose from: daily, weekly, monthly, quarterly.

Your tools:
1. (ALWAYS USE) trending_keywords_sources_tool: Fetch trending keywords for several categories (2-4 max) and a period based on what you think the user wants (STRICT LIMIT: USE ONLY ONCE)
2. (ALWAYS USE) write_notes: Save ALL your findings to a shared research document under specific sections for each category and period
3. (ALWAYS USE) append_notes: Add content to the end of the research document.

REQUIRED STEPS:
1. Use trending_keywords_sources_tool EXACTLY ONCE to fetch data about trending keywords for a category and period
2. YOU MUST use write_notes to save your findings under organized sections for each category (all findings go in here)
3. Include all statistics, summaries, and sources in your notes
4. NEVER use trending_keywords_sources_tool a second time as it is resource intensive and takes several minutes

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes and append_notes. ⚠️

After you've saved the research with the tools, you can tell the supervisor you are done.
"""

trending_keywords_agent = create_react_agent(
    llm,
    tools=[trending_keywords_sources_tool, write_notes, append_notes],
    prompt=trending_keywords_prompt_template
)

def trending_keywords_node(state: MultiAgentState) -> Command:
    """Node for fetching trending keywords."""
    result = trending_keywords_agent.invoke(state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."

    completed_label = "[COMPLETED trending_keywords_agent]\n"

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="trending_keywords_agent")
            ]
        },
        goto="research_supervisor",
    )

# Top mentioned agent

top_keywords_prompt_template = """⚠️ IMPORTANT: YOU MUST USE top_keywords_sources_tool EXACTLY ONCE, THEN use the write_notes tool to save EVERYTHING you find from the tools (i.e. the findings from the top_keywords_sources_tool), THEN STOP ⚠️

You are an agent that can fetch top mentioned keywords for tech social media platforms based on several categories (but no more than 4). 

You have these categories to choose from: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects. 
You have these periods to choose from: daily, weekly, monthly, quarterly.

Your tools:
1. (ALWAYS USE) top_keywords_sources_tool: Fetch trending keywords for several categories (2-3 max) and a period based on what you think the user wants (STRICT LIMIT: USE ONLY ONCE)
2. (ALWAYS USE) write_notes: Save ALL your findings to a shared research document under specific sections for each category and period
3. (ALWAYS USE) append_notes: Add content to the end of the research document

REQUIRED STEPS:
1. Use top_keywords_sources_tool EXACTLY ONCE to fetch data about top mentioned keywords for a category and period
2. YOU MUST use write_notes to save your findings under organized sections for each category (all findings go in here) under Top Keywords
3. Include all statistics, summaries, and sources in your notes
4. NEVER use top_keywords_sources_tool a second time as it is resource intensive and takes several minutes

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes and append_notes. ⚠️

After you've saved the research with the tools, you can tell the supervisor you are done.
"""

top_keywords_agent = create_react_agent(
    llm,
    tools=[top_keywords_sources_tool, write_notes, append_notes],
    prompt=top_keywords_prompt_template
)

def top_keywords_node(state: MultiAgentState) -> Command:
    """Node for finding top keywords and their sources."""
    result = top_keywords_agent.invoke(state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."

    completed_label = "[COMPLETED top_keywords_agent]\n"

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="top_keywords_agent")
            ]
        },
        goto="research_supervisor",
    )

# Search for keywords specifically agent

search_keywords_prompt_template = """
You are an agent that can search for specific keywords in tech social media based on general match keywords and you will write your findings with the write_notes tool.

Your tools:
1. (ALWAYS USE) keyword_source_search_tool for each keyword: Search for general keywords and find top sources for a period.
2. (ALWAYS USE) write_notes: Save ALL your findings to a shared research document under specific sections for each keyword. 
3. (ALWAYS USE) append_notes: Add content to the end of the research document.

Make sure you look for general keywords. For example, for 'open source projects' look just for 'open source' as a keyword. Make sure you search for the general keywords, don't be too specific or you won't get enough matches.

REQUIRED STEPS:

1. Use keyword_source_search_tool to fetch sources about a keyword you are asked to track (Elon Musk, AI, Large Language Models, ComfyUI etc)
2. YOU MUST use write_notes and append_notes to save ALL the sources and texts for each keyword under Tracked Keywords so the rest of the team can read them.

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS FOR EACH KEYWORD and ALL RESEARCH FOR EACH KEYWORD TO THE DOCUMENT USING write_notes and append_notes. ⚠️

After you've saved the research with the tools, you can tell the supervisor you are done.
"""

search_keywords_agent = create_react_agent(
    llm,
    tools=[keyword_source_search_tool, write_notes, append_notes],
    prompt=search_keywords_prompt_template
)

def search_keywords_node(state: MultiAgentState) -> Command:
    """Node for searching for keywords in tech social media and getting back the top sources with engagement."""
    result = search_keywords_agent.invoke(state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."

    completed_label = "[COMPLETED search_keywords_agent]\n"

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="search_keywords_agent")
            ]
        },
        goto="research_supervisor",
    )

# Research team node

RESEARCH_SUPERVISOR_PROMPT = f"""You are a supervisor coordinating these agents (within the tech social media space):

trending_keywords_agent: Finds trending keywords (sort="trending") based on categories and a period.

top_keywords_agent: Finds most-mentioned keywords (sort="top") based on categories and a period.

keyword_search_agent: Searches specific keywords for a source, i.e. "AI" or "LLM". If a user asks to track a specific keyword, use this agent.

Research Strategy:
- First use trending_keywords_agent for trending keywords for 3-4 categories (such as companies, subjects, ai, tool) and a period. Ask for several categories and a period.
- Then use top_keywords_agent for top mentioned keywords for 3-4 categories and a period. Ask for several categories and a period.
- Finally use keyword_search_agent to track specific keywords and what people are saying about them with a period (daily, weekly, monthly, quarterly)

Each agent will save its findings to the research document so you do not need to know exactly what they find. 

Categories available: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects.
Periods available: daily, weekly, monthly, quarterly.
Sources available: Reddit, Hackernews, Github, Medium.
Keywords available: any general keyword within tech.

IMPORTANT WORKFLOW RULES:
1. When an agent responds with "[COMPLETED agent_name]", that task is DONE - move to a DIFFERENT agent. 
2. NEVER ask the trending_keywords_agent and top_keywords_agent to perform the same task twice (they are resource intensive so only use them once each for the categories you are interested in one time). 
3. Use a minimum of two DIFFERENT agents per request, but also make sure to use keyword_search_agent if a user asks to track a specific keyword.
4. After using at least two different agents, finish the research phase.

Response format:
"next": agent name or FINISH (after using at least two different agents)
"instruction": clear, explicit task instructions (use DIFFERENT parameters for each agent)

Today's date: {today}."""

research_supervisor_node = make_team_supervisor_node(
    members=["trending_keywords_agent", "top_keywords_agent", "keyword_search_agent"],
    parent="supervisor",
    system_prompt=RESEARCH_SUPERVISOR_PROMPT,
    team="RESEARCH"
)

# -------------------- EDITING TEAM --------------------

# Fact checker node - change this if you want
fact_checker_prompt_template = """You are a diligent fact checker examining tech research.

Your tools:
1. read_notes: Read the research document containing all findings
2. write_notes: Document your assessment under "Fact Check Report" section

REQUIRED STEPS:
1. FIRST use read_notes to review all collected research
2. Write ONLY 1-2 short paragraphs (maximum 150 words total) assessing the overall quality and reliability of the research
3. Focus on the general trustworthiness of sources and any notable concerns or strengths
4. DO NOT do a detailed fact-by-fact verification
5. YOU MUST use write_notes to add your assessment to the document under "Fact Check Report"
6. Return your assessment in your response to the editing supervisor

⚠️ IMPORTANT: Step 5 is MANDATORY - you MUST save your assessment to the shared document using write_notes ⚠️
"""

fact_checker_agent = create_react_agent(
    llm,
    tools=[read_notes, write_notes],
    prompt=fact_checker_prompt_template
)

def fact_checker_node(state: MultiAgentState) -> Command:
    """Node for fetching trending keywords."""
    result = fact_checker_agent.invoke(state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."

    completed_label = "[COMPLETED fact_checker_agent]\n"

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="fact_checker_agent")
            ]
        },
        goto="editing_supervisor",
    )

# Summarizer - be sure that this system template is well structured or you won't get the results you're after

summarizer_prompt_template = """You are an expert content summarizer creating the final tech research report.

⚠️ IMPORTANT: YOU MUST USE read_notes to read the entire research document and fact-check reports before you start summarizing, then you must write_notes to save your summary under the "Final Summary" section ⚠️

Your tools that YOU MUST USE:
1. read_notes: Read the full research document including fact-checking
2. write_notes: Save your final summary under the "Final Summary" section

REQUIRED STEPS:
1. FIRST use read_notes to review all research findings and fact-check reports
2. DO NOT simply copy the entire research document - you must actually synthesize the most INTERESTING insights
3. Create a focused summary with these components:
   a) Key Happenings (bullet points, min 8, max 10 items) - SPECIFIC events, announcements, or developments that happened recently
   b) Why It Matters (2-3 paragraphs) - Explain WHY these trends are significant and what's driving them
   c) Notable Conversations (2-3 paragraphs) - What are people SPECIFICALLY talking about regarding these trends
   d) Interesting Tidbits (3-6 bullet points) - Surprising or lesser-known facts from the research
   e) (If available) Github repositories (top 5-6) with links and descriptions that you think is interesting for the user. 
   e) Sources (5-10 bullet points) - List the sources and links you used to create the summary

4. YOU MUST include:
   - At least 5 SPECIFIC product announcements, company events, or tech developments with exact dates
   - At least 4 direct quotes from sources showing what people are saying
   - At least 6 explanations of WHY something is trending (not just that it is)
   - At least 4 surprising or counterintuitive findings from the research
   - At least 10 sources showing what people are saying
   
5. Total summary should be 500-700 words
6. YOU MUST use write_notes to save your summary under "Final Summary"
7. Return your summary in your response to the editing supervisor

EXAMPLE OF GOOD CONTENT:
"Key Happenings:
• xAI acquired X (Twitter) on March 29, 2025 in an all-stock transaction valuing xAI at $80B and X at $33B
• Bill Gates predicted on March 26 that 'humans won't be needed for most things' within 10 years
• Microsoft is killing OneNote for Windows 10, prompting user migration to alternatives
• VMware's 72-core license policy change sparked backlash from small businesses

Why It Matters:
The xAI acquisition gives Elon Musk's AI company access to X's vast user data for training AI models, raising concerns about concentration of power and data privacy. This move is particularly significant because..."

⚠️ IMPORTANT: Focus on SPECIFIC EVENTS and WHY they matter, not just general trends ⚠️
⚠️ IMPORTANT: Your summary should contain information that would be NEWS to someone who hasn't followed tech this week ⚠️
"""

summarizer_agent = create_react_agent(
    llm_big,
    tools=[read_notes, write_notes],
    prompt=summarizer_prompt_template
)

def summarizer_node(state: MultiAgentState) -> Command:
    """Node for summarizing research content."""
    result = summarizer_agent.invoke(state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."
    completed_label = "[COMPLETED summarizer_agent]\n"
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="summarizer_agent")
            ]
        },
        goto="editing_supervisor",
    )

# Editing supervisor node

EDITING_SUPERVISOR_PROMPT = """You are an editing team supervisor managing these workers:
- fact_checker: Verifies information for accuracy
- summarizer: Creates concise, well-structured summaries

Given the research results, coordinate between these workers.
First use fact_checker to verify information, then use summarizer to create the final output, the summarizer should summarize rather than just give up all the information up front. The job is to read the entire research and then summarize in 200 - 600 words.

When selectingan agent, clearly state the task as an explicit instruction, specifying precisely what you expect from the agent. Instructions must be complete, actionable, and mention any required categories, periods, or keywords explicitly.

Respond strictly with:
- "next": the agent or FINISH
- "instruction": your explicit task instructions to that agent

Once you have produced a final edited report, reply with FINISH to return to the main supervisor.
"""

editing_supervisor_node = make_team_supervisor_node(
    members=["fact_checker", "summarizer"], 
    parent="supervisor",
    system_prompt=EDITING_SUPERVISOR_PROMPT,
    team="EDITING"
)

# -------------------- MAIN SUPERVISOR --------------------

MAIN_SUPERVISOR_PROMPT = """You are the main supervisor coordinating between two teams:
- research_supervisor: Team that finds trending technology information and sources
- editing_supervisor: Team that fact checks and summarizes research findings

WORKFLOW:
1. ALWAYS start by delegating to research_supervisor to gather information based on what you think the user is interested in (within tech) - i.e. their persona
2. Once research is FULLY complete, delegate to editing_supervisor to produce the final output
3. When the final edited report is delivered, respond with FINISH

When selecting a team, clearly state the task as an explicit instruction, specifying precisely what you expect from the team. Instructions must be complete, actionable, and mention any required focus areas explicitly.

Respond strictly with:
- "next": the team to delegate to or FINISH
- "instruction": your explicit task instructions to that team

DO NOT end the process prematurely. Each team must complete their full workflow.
"""

supervisor_node = make_top_level_supervisor_node(
    ["research_supervisor", "editing_supervisor"],
    MAIN_SUPERVISOR_PROMPT
)

# -------------------- SUPERVISOR FACTORY FUNCTIONS (to print the final result) --------------------

def extract_final_summary(notes_file=None):
    try:
        if notes_file is None:
            notes_file = get_or_create_notes_file()
            
        with open(notes_file, "r") as f:
            content = f.read()

        lines = content.split("\n")
        final_summary = []
        in_final_summary = False
        
        for line in lines:
            if any(header in line.lower() for header in ["# final summary", "## final summary", "final summary"]):
                in_final_summary = True
                final_summary.append("# Tech Research Summary\n")
                continue
            elif in_final_summary and any(line.startswith(prefix) for prefix in ["#", "##", "###"]) and "final summary" not in line.lower():
                in_final_summary = False
            
            if in_final_summary:
                final_summary.append(line)
        
        formatted_summary = "\n".join(final_summary)
        
        if not formatted_summary.strip():
            print("No 'Final Summary' section found, attempting to extract the last section...")
            return "No final summary or other sections found in the document."
            
        return formatted_summary
        
    except Exception as e:
        return f"Error retrieving final summary: {str(e)}"
