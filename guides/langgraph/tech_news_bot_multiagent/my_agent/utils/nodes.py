import json
import logging
from typing import Literal, List, TypedDict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages.utils import convert_to_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from my_agent.utils.tools import (
    trending_keywords_sources_tool, 
    top_keywords_sources_tool, 
    keyword_source_search_tool,
    read_notes,
    write_notes,
    get_or_create_notes_file
)
from my_agent.utils.state import MultiAgentState
from langgraph.graph import END
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
logger = logging.getLogger(__name__)

# -------------------- LLMs --------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
llm_big = ChatOpenAI(model="gpt-4o")
llm_even_bigger = ChatOpenAI(
    model="gpt-5",
    reasoning_effort="medium",
    streaming=False,
    disable_streaming=True,
)
llm_biggest = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")

# -------------------- Supervisor nodes --------------------

def make_top_level_supervisor_node(members: list[str], system_prompt: str) -> str:
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]
        instruction: str  

    def supervisor_node(state: MultiAgentState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router with authority to end the workflow."""
        messages = prepare_supervisor_messages(system_prompt, state["messages"])
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

def make_team_supervisor_node(members: list[str], parent: str, system_prompt: str, team):
    options = ["FINISH"] + members

    class Router(TypedDict):
        next: Literal[*options]
        instruction: str 

    def team_supervisor_node(state: MultiAgentState) -> Command[Literal[*members, parent]]:
        messages = prepare_supervisor_messages(system_prompt, state["messages"])
        
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

trending_keywords_prompt_template = """⚠️ IMPORTANT: YOU MUST USE trending_keywords_sources_tool EXACTLY ONCE, THEN use the write_notes and append_notes tools to save EVERYTHING you find from the tools, THEN STOP ⚠️

You are an agent that can fetch trending keywords for tech social media platforms based on several categories (but no more than 4) 

You have these categories to choose from: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects. 
You have these periods to choose from: daily, weekly, monthly, quarterly.

REQUIRED STEPS:
1. Use trending_keywords_sources_tool EXACTLY ONCE to fetch data about trending keywords for several categories and a period (STRICT LIMIT: USE ONLY ONCE)
2. YOU MUST use write_notes to save your findings under a section called="Trending Keywords Analysis" (all research goes in here)

⚠️ IMPORTANT: NEVER use trending_keywords_sources_tool a second time as it is resource intensive and takes several minutes ⚠️

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes otherwise the other teams will not be able to see your findings.. ⚠️

YOU DO NOT RETURN THE RESEARCH TO THE SUPERVISOR, YOU SAVE IT TO THE DOCUMENT (using write_notes)

After you've saved the research with the tools, you can tell the supervisor you are done are done with a simple "I'm done, I have saved all the research in the document". 
"""

trending_keywords_agent = create_react_agent(
    llm_big,
    tools=[trending_keywords_sources_tool, write_notes],
    prompt=trending_keywords_prompt_template
)

def trending_keywords_node(state: MultiAgentState) -> Command:
    """Node for fetching trending keywords."""
    filtered_state = optimize_agent_state(state)
    
    result = trending_keywords_agent.invoke(filtered_state)
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


top_keywords_prompt_template = """⚠️ IMPORTANT: YOU MUST USE top_keywords_sources_tool EXACTLY ONCE, THEN use the write_notes tool to save EVERYTHING you find from the tools (i.e. the findings from the top_keywords_sources_tool), THEN STOP ⚠️

You are an agent that can fetch top mentioned keywords for tech social media platforms based on several categories (but no more than 4). 

You have these categories to choose from: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects. 
You have these periods to choose from: daily, weekly, monthly, quarterly.

REQUIRED STEPS:
1. Use top_keywords_sources_tool EXACTLY ONCE to fetch data about top mentioned keywords for several categories and period
2. YOU MUST use write_notes to save your findings under a section called="Top Keywords Analysis"

⚠️ IMPORTANT: NEVER use top_keywords_sources_tool a second time as it is resource intensive and takes several minutes ⚠️

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes otherwise the other teams will not be able to see your findings. ⚠️

YOU DO NOT RETURN THE RESULTS TO THE SUPERVISOR, YOU SAVE IT TO THE DOCUMENT (using write_notes). After you've saved the research with the tools, you can tell the supervisor you are done are done with a simple "I'm done, I have saved all the research in the document". 
"""

top_keywords_agent = create_react_agent(
    llm_big,
    tools=[top_keywords_sources_tool, write_notes],
    prompt=top_keywords_prompt_template
)

def top_keywords_node(state: MultiAgentState) -> Command:
    """Node for finding top keywords and their sources."""
    filtered_state = optimize_agent_state(state)
    
    result = top_keywords_agent.invoke(filtered_state)
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


search_keywords_prompt_template = """⚠️ IMPORTANT: YOU MUST USE keyword_source_search_tool EXACTLY ONCE, THEN use the write_notes tool to save EVERYTHING you find from the tools (i.e. the findings from the keyword_source_search_tool), THEN STOP ⚠️

You are an agent for tech social media keyword searches. You will receive a list of keywords.

You have these periods to choose from: daily, weekly, monthly, quarterly.
You have these sources to choose from: reddit, hacker (hackernews), github, medium.

REQUIRED STEPS:
1. Use keyword_source_search_tool EXACTLY ONCE to fetch data about with all the keywords for period (optional: specify a source)
2. YOU MUST use write_notes to save your findings under a section called="Specific Keyword Search Results"

⚠️ IMPORTANT: NEVER use keyword_source_search_tool a second time as it is resource intensive and takes several minutes ⚠️

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes otherwise the other teams will not be able to see your findings. ⚠️

YOU DO NOT RETURN THE RESULTS TO THE SUPERVISOR, YOU SAVE IT TO THE DOCUMENT (using write_notes). After you've saved the research with the tools, you can tell the supervisor you are done are done with a simple "I'm done, I have saved all the research in the document". 
"""

search_keywords_agent = create_react_agent(
    llm_big,
    tools=[keyword_source_search_tool, write_notes],
    prompt=search_keywords_prompt_template
)

def search_keywords_node(state: MultiAgentState) -> Command:
    """Node for searching for keywords in tech social media."""
    filtered_state = optimize_agent_state(state)
    
    result = search_keywords_agent.invoke(filtered_state)
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

github_trending_repos_prompt_template = """⚠️ IMPORTANT: YOU MUST USE keyword_source_search_tool EXACTLY ONCE with source=github, THEN use the write_notes tool to save EVERYTHING you find from the tools (i.e. the findings from the keyword_source_search_tool), THEN STOP ⚠️

You are an agent that searches trending github repositories for a keyword and a period. You will receive a list of keywords.

You have these periods to choose from: weekly, monthly, quarterly (do not use daily as it won't give back many results)

REQUIRED STEPS:
1. Use keyword_source_search_tool EXACTLY ONCE to fetch data about with all the keywords for period with source="github" (IMPORTANT: SPECIFY THE SOURCE AS "github")
2. YOU MUST use write_notes to save your findings under a section called="Trending Github repositories for keywords"

⚠️ IMPORTANT: NEVER use keyword_source_search_tool a second time as it is resource intensive and takes several minutes ⚠️

⚠️ IMPORTANT: YOU MUST SAVE YOUR FINDINGS and ALL RESEARCH TO THE DOCUMENT USING write_notes otherwise the other teams will not be able to see your findings. ⚠️

YOU DO NOT RETURN THE RESULTS TO THE SUPERVISOR, YOU SAVE IT TO THE DOCUMENT (using write_notes). After you've saved the research with the tools, you can tell the supervisor you are done are done with a simple "I'm done, I have saved all the research in the document". 
"""

trending_github_repos_agent = create_react_agent(
    llm_big,
    tools=[keyword_source_search_tool, write_notes],
    prompt=github_trending_repos_prompt_template
)

def github_keywords_node(state: MultiAgentState) -> Command:
    """Node for searching for keywords in tech social media."""
    filtered_state = optimize_agent_state(state)
    
    result = trending_github_repos_agent.invoke(filtered_state)
    agent_messages = [msg for msg in result["messages"] if msg.content.strip()]
    agent_content = agent_messages[-1].content if agent_messages else "No valid results."

    completed_label = "[COMPLETED trending_github_repos_agent]\n"

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=completed_label + agent_content, name="search_keywords_agent")
            ]
        },
        goto="research_supervisor",
    )

RESEARCH_SUPERVISOR_PROMPT = f"""You are a supervisor coordinating these agents (within the tech social media space):

trending_keywords_agent: Finds trending keywords (sort="trending") based on categories and a period.

top_keywords_agent: Finds most-mentioned keywords (sort="top") based on categories and a period.

keyword_search_agent: Searches specific keywords for a source, i.e. "AI" or "LLM". If a user asks to track a specific keyword, use this agent.

Research Strategy:
- First use trending_keywords_agent for trending keywords for 3-4 categories (such as companies, subjects, ai, tool) and a period. Ask for several categories and a period.
- Then use top_keywords_agent for top mentioned keywords for 3-4 categories and a period. Ask for several categories and a period.
- Finally use keyword_search_agent to track specific keywords and what people are saying about them with a period (daily, weekly, monthly, quarterly)
- (optional) use the trending_github_repos_agent to find trending github repositories based on a keyword and a period (weekly, monthly, quarterly)

Each agent will save its findings to the research document so you do not need to know exactly what they find. 

Categories available: companies, ai, tools, platforms, hardware, people, frameworks, languages, concepts, websites, subjects.
Periods available: daily, weekly, monthly, quarterly.
Sources available: Reddit, Hackernews, Github, Medium.
Keywords available: any general keyword within tech.

IMPORTANT WORKFLOW RULES:
1. When an agent responds with "[COMPLETED agent_name] I'm done", that task is DONE - move to a DIFFERENT agent. 
2. NEVER ask the trending_keywords_agent and top_keywords_agent to perform the same task twice (they are resource intensive so only use them once each for the categories you are interested in one time). 
3. Use a minimum of two DIFFERENT agents per request, but also make sure to use keyword_search_agent if a user asks to track a specific keyword.
4. After using at least two different agents, finish the research phase.

Response format:
"next": agent name or FINISH (after using at least two different agents)
"instruction": clear, explicit task instructions (use DIFFERENT parameters for each agent)

Today's date: {today}."""

research_supervisor_node = make_team_supervisor_node(
    members=["trending_keywords_agent", "top_keywords_agent", "keyword_search_agent", "trending_github_repos_agent"],
    parent="supervisor",
    system_prompt=RESEARCH_SUPERVISOR_PROMPT,
    team="RESEARCH"
)

# -------------------- EDITING TEAM --------------------

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
    """Node for checking facts in research."""
    # Filter state to include original user message and latest supervisor message
    filtered_state = optimize_agent_state(state)
    
    result = fact_checker_agent.invoke(filtered_state)
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


summarizer_prompt_template = """You are an expert content summarizer creating the final tech research report.

⚠️ IMPORTANT: YOU MUST USE read_notes to read the entire research document, then you must write_notes to save your summary under the "Final Summary" section ⚠️

Your tools that YOU MUST USE:
1. read_notes: Read the full research document including fact-checking
2. write_notes: Save your final summary under the "Final Summary" section

REQUIRED STEPS:
1. FIRST use read_notes to review all research findings and fact-check reports
2. Look through through all the messages in state.
2. DO NOT simply copy the entire research document - you must actually synthesize the most INTERESTING insights
3. Create a focused summary with these components:
   a) Key Happenings (bullet points, min 10, max 15 items) - SPECIFIC events, announcements, or developments that happened with each keyword. 
   b) Why It Matters (3-4 paragraphs) - Explain WHY these trends are significant and what's driving them
   c) Notable Conversations (1-2 paragraphs) - What are people SPECIFICALLY talking about regarding these trends (look at discussions, comments, etc.)
   d) Interesting Tidbits (5-8 bullet points) - Surprising or lesser-known facts from the research (note the same as  Key Happenings - find interesting tidbits)
   e) (If available) Github repositories (top 5-6) with links and descriptions that you think is interesting for the user. 
   e) Sources (10-15 bullet points) - List the sources and links you used to create the summary

4. YOU MUST include:
   - At least 6 SPECIFIC product announcements, company events, or tech developments with exact dates
   - At least 4 direct quotes from sources showing what people are saying
   - At least 5 explanations of WHY something is trending (not just that it is)
   - At least 5 surprising or counterintuitive findings from the research
   - At least 10 sources showing what people are saying
   
5. Total summary should be 500-800 words
6. YOU MUST use write_notes to save your summary under "Final Summary"
7. Return your summary in your response to the editing supervisor

EXAMPLE OF GOOD CONTENT:
"Key Happenings:
• Bill Gates predicted on March 26 that 'humans won't be needed for most things' within 10 years
• Microsoft is killing OneNote for Windows 10, prompting user migration to alternatives
• VMware's 72-core license policy change sparked backlash from small businesses

Why It Matters:
Bill Gates' comments on AI's potential to replace human roles underscore growing anxieties about employment and the role of humans in an AI-driven future...."

⚠️ IMPORTANT: Focus on SPECIFIC EVENTS and WHY they matter, not just general trends ⚠️
⚠️ IMPORTANT: Your summary should contain information that would be NEWS to someone who hasn't followed tech for the period they are asking. ⚠️
"""

summarizer_agent = create_react_agent(
    llm_even_bigger,
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

EDITING_SUPERVISOR_PROMPT = """You are an editing team supervisor managing these workers:
- fact_checker: Verifies information for accuracy
- summarizer: Creates a concise, well-structured summary of the research

Given the research results, coordinate between these workers.
First use fact_checker to verify information, then use summarizer to create the final output, the summarizer should summarize rather than just give up all the information up front. The job is to read the entire research and then summarize in 600 - 1000 words.

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

# -------------------- HELPERS --------------------

def prepare_supervisor_messages(system_prompt: str, state_messages: List[Any]) -> List[BaseMessage]:
    """Normalize state messages so Gemini always receives plain-text content."""
    history = convert_to_messages(state_messages or [])
    normalized_history: List[BaseMessage] = []

    for message in history:
        content_text = _coerce_message_content_to_text(message.content)
        if not content_text:
            continue
        normalized_history.append(message.model_copy(update={"content": content_text}))

    if not normalized_history:
        logger.warning(
            "Supervisor invoked without usable messages; injecting placeholder instruction."
        )
        normalized_history.append(
            HumanMessage(
                content=(
                    "No user context was available. Provide a concise plan for how the "
                    "teams should gather and summarize the latest technology signals."
                ),
                name="system-fallback",
            )
        )

    return [SystemMessage(content=system_prompt), *normalized_history]


def _coerce_message_content_to_text(content: Any) -> str:
    """Convert LangGraph/Assistants content payloads into plain text for Gemini."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                text_value = None
                for key in ("text", "input_text", "output_text", "content"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        text_value = value
                        break
                if text_value:
                    chunks.append(text_value)
                elif part.get("type") == "image_url" and part.get("image_url"):
                    chunks.append(f"[image:{part['image_url']}]")
                else:
                    chunks.append(json.dumps(part))
                continue
            chunks.append(str(part))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    if content is None:
        return ""
    return str(content).strip()

def extract_final_summary(notes_file=None):
    try:
        if notes_file is None:
            notes_file = get_or_create_notes_file()
            
        with open(notes_file, "r") as f:
            content = f.read()

        lines = content.split("\n")
        final_summary = []
        in_final_summary = False
        final_summary_level = 0 
        
        for line in lines:
            if not in_final_summary and any(header in line.lower() for header in ["# final summary", "## final summary", "final summary"]):
                in_final_summary = True
                final_summary_level = line.count('#') if '#' in line else 2  
                final_summary.append("# Tech Research Summary\n")
                continue
                
            elif in_final_summary and line.strip().startswith('#'):
                current_level = line.count('#')
                if current_level <= final_summary_level and "final summary" not in line.lower():
                    in_final_summary = False
                    continue
            
            if in_final_summary:
                final_summary.append(line)
        
        formatted_summary = "\n".join(final_summary)
        
        if not formatted_summary.strip():
            print("No 'Final Summary' section found in the exact format expected.")
            return "No final summary found. Please check if the summarizer agent correctly created a 'Final Summary' section."
            
        return formatted_summary
        
    except Exception as e:
        return f"Error retrieving final summary: {str(e)}"

def optimize_agent_state(state: MultiAgentState):
    """Create an optimized state that includes only the original user message and the latest supervisor message."""
    original_message = state["messages"][0] if state["messages"] else None
    supervisor_messages = [
        msg for msg in state["messages"] 
        if msg.type == "human" and "SUPERVISOR" in str(msg.content)
    ]
    latest_supervisor = supervisor_messages[-1] if supervisor_messages else None
    filtered_messages = []
    if original_message:
        filtered_messages.append(original_message)
    if latest_supervisor:
        filtered_messages.append(latest_supervisor)
    
    return {"messages": filtered_messages}
