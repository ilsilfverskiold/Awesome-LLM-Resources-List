import os
from functools import lru_cache
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from my_agent.utils.tools import all_tools

MODEL_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

_MODEL_PRIORITY = ("openai", "anthropic", "gemini")


def _detect_default_model() -> str | None:
    """Pick the first model with credentials configured."""
    for model_name in _MODEL_PRIORITY:
        env_var = MODEL_ENV_VARS[model_name]
        if os.getenv(env_var):
            return model_name
    return None


DEFAULT_MODEL_NAME = _detect_default_model()


def _ensure_credentials(model_name: str) -> str:
    env_var = MODEL_ENV_VARS.get(model_name)
    if not env_var:
        raise ValueError(f"Unsupported model type: {model_name}")
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(
            f"{model_name!r} selected but {env_var} is not set. "
            "Add the key to your environment/.env or choose a different model."
        )
    return api_key

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        _ensure_credentials(model_name)
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        _ensure_credentials(model_name)
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    elif model_name == "gemini":
        api_key = _ensure_credentials(model_name)
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model = model.bind_tools(all_tools)
    
    return model

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

today = datetime.now().strftime("%Y-%m-%d")

system_prompt = f"""You are a thorough tech industry research assistant with access to these tools:

1. tech_trends_tool: Get trending technology keywords
2. tech_sources_tool: Find sources discussing specific keywords (use limit=10)

Today's date is {today}.

You should always do the following:

REQUIRED RESEARCH PROCESS - FOLLOW EVERY STEP:
1. FIRST: Use tech_trends_tool to identify relevant keywords
   - Choose an appropriate period (daily, weekly, monthly, quarterly)
   - Choose a category that is relevant to the user (investor may be interested in companies, ai, people, websites, subjects while a developer may be interested in frameworks, languages, tools, platforms, etc.)
   
2. SECOND: Use tech_sources_tool for EACH interesting keyword
   - ALWAYS use limit=10 to get diverse sources
   - CRITICAL: Use the SAME PERIOD parameter that you used in tech_trends_tool
   - Example: If you used period="weekly" in tech_trends_tool, also use period="weekly" in tech_sources_tool
   
3. FINALLY: Synthesize all findings into a comprehensive answer with source citations

DON'T SKIP STEPS - all research questions require using MULTIPLE tools in sequence.
DON'T provide final answers until you've completed the full research workflow.
ALWAYS match the period parameter between tech_trends_tool and tech_sources_tool.
"""

def call_model(state, config):
    """Call the model without forcing tool calls."""
    messages = state["messages"]
    
    # Add system message
    system_message = SystemMessage(content=system_prompt)
    full_messages = [system_message] + messages
    
    configurable = (config or {}).get("configurable", {})
    model_name = configurable.get("model_name") or DEFAULT_MODEL_NAME
    if not model_name:
        raise RuntimeError(
            "No LLM credentials found. Set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or GOOGLE_API_KEY (for Gemini) or pass `configurable.model_name` when "
            "invoking the graph."
        )
    model = _get_model(model_name)
    
    try:
        # Just invoke the model normally and return its response
        response = model.invoke(full_messages)
        print(f"Model returned response of type: {type(response)}")
        
        # Check if we got a valid response
        if response:
            return {"messages": [response]}
        else:
            print("Trying again...")
            response = model.invoke(full_messages)
    except Exception as e:
        return {"messages": [{
            "role": "assistant",
            "content": f"I encountered an error while processing your request. Let me try a different approach. Error: {str(e)[:100]}"
        }]}

tool_node = ToolNode(all_tools)
