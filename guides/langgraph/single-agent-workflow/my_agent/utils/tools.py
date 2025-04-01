import os
import requests
import json
from langchain_core.tools import tool

# Get trending tech keywords from Safron API
@tool
def tech_trends_tool(period: str = "daily", category: str = None, limit: int = 10) -> str:
    """Find trending tech keywords and their metrics. ALWAYS USE THIS TOOL FIRST.
    
    This is the starting point for any research. The keywords returned by this tool
    should be used as inputs to tech_sources_tool in the next step.
    
    Output includes a "keywords" array with objects containing:
    - "keyword": The trending term (use this exact string with tech_sources_tool)
    - "category": The category it belongs to
    - "count": Number of mentions
    - "sentiment": Overall sentiment toward the keyword
    
    Args:
        period: Time period for trends - one of ['daily', 'weekly', 'monthly', 'quarterly']. Default is 'daily'.
        category: Optional category filter - one of ['companies', 'ai', 'tools', 'platforms', 
                 'hardware', 'people', 'frameworks', 'languages', 'concepts', 'websites', 'subjects'].
        limit: Number of results to return (max 100). Default is 10.
    
    Returns:
        JSON with trending keywords. Extract the "keyword" field values to use with tech_sources_tool.
    """
    base_url = "https://public.api.safron.io/v2/keywords"
    
    valid_periods = ['daily', 'weekly', 'monthly', 'quarterly']
    if period not in valid_periods:
        return f"Error: Invalid period '{period}'. Please use one of: {', '.join(valid_periods)}"
    
    valid_categories = ['companies', 'ai', 'tools', 'platforms', 'hardware', 'people', 
                        'frameworks', 'languages', 'concepts', 'websites', 'subjects']
    if category and category.lower() not in valid_categories:
        return f"Error: Invalid category '{category}'. Please use one of: {', '.join(valid_categories)}"
    
    params = {
        "period": period,
        "slim": "true",
        "limit": str(limit),
        "sort": "trending"
    }
    
    if category:
        params["category"] = category.lower()
        
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=2)
    else:
        return f"Error fetching tech trends: {response.status_code} - {response.text}"

# Get sources for specific keywords
@tool
def tech_sources_tool(keyword: str, limit: int = 10, period: str = "daily", source: str = None, sentiment: str = None) -> str:
    """Find sources (comments/posts) discussing specific keywords identified from tech_trends_tool.
    
    This tool should be used AFTER identifying relevant keywords with tech_trends_tool.
    It helps find specific articles and discussions about those keywords.
    
    Args:
        keyword: Keyword from tech_trends_tool (e.g., "OpenAI", "Nvidia")
        limit: Number of sources to return. Default is 10 to get diverse perspectives.
        period: Time period for sources - one of ['daily', 'weekly', 'monthly', 'quarterly']. Default is 'daily'.
        source: Optional source filter (e.g., 'hackernews', 'reddit', 'devto', 'github')
        sentiment: Optional sentiment filter - 'neutral', 'positive', or 'negative'
    
    Returns:
        List of sources discussing the specified keyword.
    """
    base_url = "https://public.api.safron.io/v2/sources"
    
    payload = {
        "search": keyword
    }
    
    params = {
        "limit": limit,
        "slim": "true",
        "sort": "engagement" 
    }
    
    if period:
        valid_periods = ['daily', 'weekly', 'monthly', 'quarterly']
        if period not in valid_periods:
            return f"Error: Invalid period '{period}'. Please use one of: {', '.join(valid_periods)}"
        params["period"] = period
    
    if source:
        params["source"] = source
    
    if sentiment:
        valid_sentiments = ['neutral', 'positive', 'negative']
        if sentiment not in valid_sentiments:
            return f"Error: Invalid sentiment '{sentiment}'. Please use one of: {', '.join(valid_sentiments)}"
        params["sentiment"] = sentiment
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(base_url, headers=headers, json=payload, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=2)
    else:
        return f"Error fetching sources: {response.status_code} - {response.text}"

all_tools = [tech_trends_tool, tech_sources_tool]