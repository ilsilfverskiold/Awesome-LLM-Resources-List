import requests
from langchain_core.tools import tool
from typing import List, Optional
import os
import uuid
from datetime import datetime

# -------------------- CORE API FUNCTIONS --------------------
# -------------------- Keywords Data -------------------

def fetch_keywords_data(period="daily", category=None, limit=3, sort="trending"):
    """Core function to fetch keyword data from the API."""
    
    valid_periods = ['daily', 'weekly', 'monthly', 'quarterly']
    valid_categories = ['companies', 'ai', 'tools', 'platforms', 'hardware', 'people', 
                      'frameworks', 'languages', 'concepts', 'websites', 'subjects']
    valid_sorts = ['trending', 'top']
    validation_dict = {
        "period": (period, valid_periods, True),
        "category": (category, valid_categories if category else None, False),
        "sort": (sort, valid_sorts, True)
    }
    is_valid, error_message = validate_parameters(validation_dict)
    if not is_valid:
        return False, error_message
    
    params = {
        "period": period,
        "slim": "true",
        "limit": str(limit),
        "sort": sort
    }
    
    if category:
        params["category"] = category.lower()
    
    try:
        response = requests.get("https://public.api.safron.io/v2/keywords", params=params)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error fetching keywords: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Exception during API call: {str(e)}"
    
# -------------------- Sources Data -------------------

def fetch_sources_data(keyword, source=None, period="daily", limit=5, type=None):
    """Core function to fetch source data from the API."""

    valid_periods = ['daily', 'weekly', 'monthly', 'quarterly']
    validation_dict = {
        "period": (period, valid_periods, True),
        "keyword": (keyword, None, True) 
    }
    is_valid, error_message = validate_parameters(validation_dict)
    if not is_valid:
        return False, error_message
        
    payload = {"search": keyword}
    params = {
        "limit": limit,
        "slim": "true",
        "sort": "engagement",
        "period": period
    }
    if source:
        params["source"] = source
    if type:
        params["type"] = type
    
    try:
        response = requests.post(
            "https://public.api.safron.io/v2/sources", 
            headers={"Content-Type": "application/json"}, 
            json=payload, 
            params=params
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error fetching sources: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Exception during API call: {str(e)}"
    
# -------------------- AI Summaries (Keywords) -------------------

def fetch_keyword_summary(keyword, period="daily"):
    """Fetch an AI-generated summary for a keyword."""
    base_url = "https://public.api.safron.io/v2/ai-summary"
    
    valid_periods = ['daily', 'weekly', 'monthly', 'quarterly']
    validation_dict = {
        "period": (period, valid_periods, True),
        "keyword": (keyword, None, True)
    }
    is_valid, error_message = validate_parameters(validation_dict)
    if not is_valid:
        return False, error_message
        
    payload = {
        "keywords": keyword,
        "period": period
    }
    try:
        response = requests.post(
            base_url,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error fetching summary: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Exception during API call: {str(e)}"

# -------------------- Helper functions --------------------

# -------------------- Formatting --------------------

def format_source_items(sources, standalone=False):
    """Format a list of source items consistently."""
    formatted_text = ""
    
    if not sources:
        return "No sources available."
    
    for idx, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            formatted_text += f"{idx}. {source}\n\n"
            continue
            
        title = source.get('text', source.get('title', 'No title'))
        
        published = source.get("published", "")
        if published:
            try:
                from datetime import datetime
                date_obj = datetime.fromisoformat(published.replace("Z", "+00:00"))
                published = date_obj.strftime("%b %d, %Y")
            except:
                pass
        
        formatted_text += f"{idx}. **{title}**\n"
        if published:
            formatted_text += f"   - Published: {published}\n"
        formatted_text += f"   - Engagement: {source.get('engagement', 'Unknown')}\n"
        formatted_text += f"   - Source: {source.get('source', 'Unknown')}\n"
        formatted_text += f"   - Type: {source.get('type', 'Unknown type')}\n"
        
        link = source.get('link', source.get('url', '#'))
        source_name = source.get('source', 'Link')
        formatted_text += f"   - Link: [{source_name}]({link})\n\n"
    
    return formatted_text

def format_enhanced_report(all_results, report_title, category_suffix):
    """Format report with enhanced data including statistics and summaries."""
    formatted_response = f"# {report_title}\n\n"
    
    for category, keyword_data in all_results.items():
        formatted_response += f"## {category.upper()} - {category_suffix}\n\n"
        
        if not isinstance(keyword_data, dict):
            formatted_response += f"{keyword_data}\n\n"
            continue
        
        for keyword, data in keyword_data.items():
            stats = data.get("stats", {})
            summary = data.get("summary", "No summary available")
            sources = data.get("sources", [])
            formatted_response += f"### {keyword}\n\n"
            
            if stats:
                formatted_response += "**Statistics:**\n"
                formatted_response += f"- Mentions: {stats.get('count', 'N/A')}\n"
                if 'change_in_count' in stats:
                    change = stats.get('change_in_count')
                    direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                    formatted_response += f"- Trend: {direction} {abs(change)}%\n"
                formatted_response += f"- Engagement: {stats.get('engagement', 'N/A')}\n"
                formatted_response += f"- Sentiment: {stats.get('sentiment', 'N/A')}\n\n"
            
            if summary and summary != "No summary available":
                formatted_response += "**Summary:**\n"
                formatted_response += f"{summary}\n\n"
            
            formatted_response += "**Top Sources:**\n\n"
            if isinstance(sources, list):
                formatted_response += format_source_items(sources)
            else:
                formatted_response += f"Sources: {sources}\n\n"
    
    return formatted_response

# -------------------- Validation --------------------

def validate_parameters(params_dict):
    """Validates API parameters against allowed values."""
    for param_name, (param_value, allowed_values, required) in params_dict.items():
        if param_value is None and not required:
            continue
            
        if required and param_value is None:
            return False, f"Error: {param_name} is required but not provided"
            
        if allowed_values and param_value not in allowed_values:
            return False, f"Error: Invalid {param_name} '{param_value}'. Please use one of: {', '.join(allowed_values)}"
    
    return True, None

# -------------------- Shared Implementation --------------------

def get_keywords_sources_data(sort, categories=None, period="daily", limit=None):
    if not categories:
        categories = ['companies', 'subjects', 'people', 'websites'] if sort == "trending" else ['companies', 'subjects']
    
    if limit is None:
        limit = 3 if sort == "trending" else 2
    
    all_results = {}
    
    for category in categories:
        success, keywords_data = fetch_keywords_data(
            period=period, 
            category=category, 
            limit=limit, 
            sort=sort
        )
        
        if not success:
            all_results[category] = keywords_data  # Error message
            continue
            
        keyword_results = {}
        
        try:
            for item in keywords_data.get("keywords", []):
                keyword = item.get("keyword")
                
                if not keyword:
                    continue
                
                stats = {
                    "count": item.get("count"),
                    "change_in_count": item.get("change_in_count"),
                    "engagement": item.get("engagement"),
                    "sentiment": item.get("sentiment")
                }
                
                summary_success, summary_data = fetch_keyword_summary(keyword=keyword, period=period)
                summary = summary_data.get("summary") if summary_success else "Summary not available"

                sources_success, sources_data = fetch_sources_data(
                    keyword=keyword, 
                    period=period, 
                    limit=3
                )
                
                sources = []
                if sources_success:
                    try:
                        for source in sources_data.get("articles", []):
                            sources.append({
                                "text": source.get("text"),
                                "engagement": source.get("engagement"),
                                "url": source.get("link"),
                                "source": source.get("source"),
                                "type": source.get("type")
                            })
                    except Exception as e:
                        print(f"Error parsing sources: {e}")
                        sources = [f"Error parsing sources: {str(e)}"]
                else:
                    sources = [sources_data]
                
                keyword_results[keyword] = {
                    "stats": stats,
                    "summary": summary,
                    "sources": sources
                }
        except Exception as e:
            keyword_results = {"Error": f"Failed to process keywords: {str(e)}"}
        
        all_results[category] = keyword_results
    
    report_title = "Trending Keywords Analysis" if sort == "trending" else "Top Keywords Analysis"
    category_suffix = "TRENDS" if sort == "trending" else "MOST MENTIONED"
    
    return all_results, report_title, category_suffix

# -------------------- TOOLS --------------------

@tool
def trending_keywords_sources_tool(categories: Optional[List[str]] = None, period: str = "daily", limit: int = 3) -> str:
    """Find trending keywords AND their sources for various categories in tech from various social media platforms.
    
    This tool automatically fetches trending keywords, statistics, summaries and sources for various categories.
    
    How to use it:
    Args:
        categories: List of categories to search [companies, ai, tools, platforms, hardware, people, 
                  frameworks, languages, concepts, websites, subjects]. Default is companies, subjects, people, websites.
        period: Time period for analysis - 'daily', 'weekly', 'monthly', or 'quarterly'. Default is 'daily'.
               Note: With 'monthly' and 'quarterly' you need to provide a category.
        limit: Number of keywords per category. Default is 3.
        
    Returns:
        Complete report with trending keywords, statistics, summaries and sources.
    """

    results, title, suffix = get_keywords_sources_data(
        sort="trending",
        categories=categories,
        period=period,
        limit=limit
    )
    
    return format_enhanced_report(results, title, suffix)

@tool
def top_keywords_sources_tool(categories: Optional[List[str]] = None, period: str = "daily", limit: int = 2) -> str:
    """Find most mentioned keywords with enhanced statistics and summaries in tech social media platforms.
    
    This tool automatically fetches top keywords, statistics, summaries and sources for various categories.
    
    Args:
        categories: List of categories to search [companies, ai, tools, platforms, hardware, people,
                  frameworks, languages, concepts, websites, subjects]. Default is companies and subjects.
        period: Time period for analysis - 'daily', 'weekly', 'monthly', or 'quarterly'. Default is 'daily'.
               Note: With 'monthly' and 'quarterly' you need to provide a category.
        limit: Number of keywords per category. Default is 2.
        
    Returns:
        Complete report with top keywords, statistics, summaries and sources.
    """

    results, title, suffix = get_keywords_sources_data(
        sort="top",
        categories=categories,
        period=period,
        limit=limit
    )
    
    return format_enhanced_report(results, title, suffix)


@tool
def keyword_source_search_tool(
    keywords: str, 
    source: str = None, 
    period: str = "daily", 
    limit: int = 10,
    content_type: str = None
) -> str:
    """Search for keywords and their sources in tech social media platforms.
    
    Args:
        keywords: Single keyword or comma-separated list (e.g., "AI" or "AI, Python, AWS")
        source: Optional platform to search (e.g., "reddit", "hackernews", "github", "medium")
        period: Time period - 'daily', 'weekly', 'monthly', or 'quarterly'. Default is 'daily'.
        limit: Maximum number of sources to return. Default is 10.
        content_type: Optional filter for content type
        
    Returns:
        Formatted report of sources discussing the keyword(s).
    """

    keyword_list = [k.strip() for k in keywords.split(',')] if ',' in keywords else [keywords.strip()]

    response = "# Keyword Search Results\n\n"
    
    for kw in keyword_list:
        sources_success, sources_data = fetch_sources_data(
            keyword=kw,
            source=source,
            period=period,
            limit=limit,
            type=content_type
        )
        
        response += f"## Sources for '{kw}'\n\n"
        
        if not sources_success:
            response += f"Error fetching sources: {sources_data}\n\n---\n\n"
            continue
            
        articles = sources_data.get("articles", [])
        if not articles:
            response += f"No sources found matching your criteria.\n\n---\n\n"
            continue
            
        response += format_source_items(articles, standalone=True)
        response += "\n---\n\n"
        
    return response

CURRENT_NOTES_FILE = None

def get_or_create_notes_file():
    """Get the current notes file path or create a new one."""
    global CURRENT_NOTES_FILE
    if CURRENT_NOTES_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"research_notes_{timestamp}_{unique_id}.md"
        
        os.makedirs("notes", exist_ok=True)
        CURRENT_NOTES_FILE = os.path.join("notes", filename)
        
        with open(CURRENT_NOTES_FILE, "w") as f:
            f.write(f"# Research Notes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"\n----- CREATED NEW NOTES FILE: {CURRENT_NOTES_FILE} -----\n")
    
    return CURRENT_NOTES_FILE

@tool
def read_notes() -> str:
    """Read the current research notes file.
    
    Returns:
        The current contents of the research notes file.
    """
    notes_file = get_or_create_notes_file()
    
    try:
        with open(notes_file, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading notes file: {str(e)}"

@tool
def write_notes(content: str, section: str = "General") -> str:
    """Write content to the research notes file under a specific section.
    
    Args:
        content: The content to write to the notes
        section: The section heading to place the content under
        
    Returns:
        Confirmation message.
    """
    notes_file = get_or_create_notes_file()
    
    try:
        existing_content = ""
        try:
            with open(notes_file, "r") as f:
                existing_content = f.read()
        except FileNotFoundError:
            pass
        
        section_header = f"## {section}"
        if section_header in existing_content:
            lines = existing_content.split("\n")
            with open(notes_file, "w") as f:
                section_found = False
                for line in lines:
                    f.write(line + "\n")
                    if line == section_header:
                        section_found = True
                        f.write(f"\n{content}\n\n")
                
                if not section_found:
                    f.write(f"\n## {section}\n\n{content}\n\n")
        else:
            with open(notes_file, "a") as f:
                f.write(f"\n## {section}\n\n{content}\n\n")
        
        with open(notes_file, "r") as f:
            current_content = f.read()
        print(f"\n----- NOTES FILE CONTENTS AFTER WRITING TO SECTION '{section}' -----\n")
        print(current_content)
        print(f"\n----- END OF NOTES FILE CONTENTS -----\n")
        
        return f"Successfully added to research notes under section '{section}'."
    except Exception as e:
        return f"Error writing to notes file: {str(e)}"




