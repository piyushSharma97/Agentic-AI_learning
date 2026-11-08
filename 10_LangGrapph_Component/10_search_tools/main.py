"""
=============================================================================
LANGGRAPH COMPONENT 10: SEARCH TOOLS ON THEIR OWN
=============================================================================
TOPIC OVERVIEW:
Before letting a chatbot choose between tools, you should test each tool by
itself. This lets you inspect the exact text the model will receive as its
`ToolMessage` observation.

TOOLS COVERED:
1. `search_arxiv`: Fetches academic research papers. No API key needed.
   (Uses modern `arxiv.Client` replacing broken legacy wrappers).
2. `Wikipedia`: Fetches encyclopedic summaries. No API key needed.
3. `Tavily`: Searches the live web for current events. Requires `TAVILY_API_KEY`.

EXECUTION FLOW:
    [Direct Tool Invocation] ──> [Inspect Raw Output] ──> [Format & Truncate]
=============================================================================
"""

import sys
from pathlib import Path
import os
import warnings

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import arxiv
from langchain_core.tools import tool
from config import safe_run

# Suppress community sunset warnings for cleaner educational output
warnings.filterwarnings("ignore", message=".*langchain-community.*")


# ---------------------------------------------------------------------------
# TOOL 1: ARXIV RESEARCH PAPERS
# ---------------------------------------------------------------------------
@tool
def search_arxiv(query: str) -> str:
    """Search arXiv by keywords or by a paper id such as 1706.03762."""
    # Using modern arxiv.Client (safe against breaking changes in arxiv 4.0+)
    client = arxiv.Client(page_size=2, delay_seconds=1.0, num_retries=2)
    cleaned = query.strip()
    
    # Check if query is a paper ID (e.g. '1706.03762' for 'Attention Is All You Need')
    if cleaned.replace("v", "").replace(".", "").isdigit() and "." in cleaned:
        search = arxiv.Search(id_list=[cleaned], max_results=1)
    else:
        search = arxiv.Search(query=cleaned, max_results=2)

    try:
        papers = list(client.results(search))
        if not papers:
            return f"No arXiv paper found for {query!r}."

        blocks = []
        for paper in papers:
            # Truncate summary to ~500 characters so we don't overflow the LLM context window!
            summary = " ".join(paper.summary.split())[:500]
            authors = ", ".join(author.name for author in paper.authors[:5])
            blocks.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.date()}\n"
                f"URL: {paper.entry_id}\n"
                f"Summary: {summary}..."
            )
        return "\n\n".join(blocks)
    except Exception as e:
        return f"arXiv search error: {e}"


def example_arxiv():
    """Fetch 'Attention Is All You Need' paper by ID."""
    print(f"Tool Schema Name: {search_arxiv.name}")
    print(f"Tool Description:\n{search_arxiv.description}\n")
    print("Executing search for paper '1706.03762':")
    output = search_arxiv.invoke("1706.03762")
    print(output)


# ---------------------------------------------------------------------------
# TOOL 2: WIKIPEDIA ENCYCLOPEDIC SUMMARIES
# ---------------------------------------------------------------------------
def example_wikipedia():
    """Fetch introductory summary from Wikipedia."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        # doc_content_chars_max=500 keeps the observation compact
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

        print(f"Tool Schema Name: {wiki.name}")
        print("Executing search for 'Machine learning':\n")
        result = wiki.invoke("Machine learning")
        print(result)
    except Exception as e:
        print(f"Wikipedia execution notice: {e}")


# ---------------------------------------------------------------------------
# TOOL 3: TAVILY REAL-TIME WEB SEARCH
# ---------------------------------------------------------------------------
def example_tavily():
    """Search live web. Gracefully warns if TAVILY_API_KEY is absent."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY is not set in your .env file.")
        print("To enable live web search, register at https://tavily.com and add:")
        print("    TAVILY_API_KEY=your_key_here")
        print("Skipping live web search test cleanly.\n")
        return

    try:
        from langchain_tavily import TavilySearch
        tavily = TavilySearch(max_results=3)
        print(f"Tool Schema Name: {tavily.name}")
        print("Executing search for 'Recent generative AI news':\n")
        result = tavily.invoke({"query": "Recent generative AI news"})
        print(result)
    except Exception as e:
        print(f"Tavily execution notice: {e}")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_run("Arxiv Research Tool", example_arxiv)
    safe_run("Wikipedia Encyclopedic Tool", example_wikipedia)
    safe_run("Tavily Live Web Search Tool", example_tavily)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 10: Search Tools
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates standalone testing of external search tools (arXiv, Wikipedia, Tavily)
# before integrating them into an autonomous chatbot.
#
# KEY CONCEPTS USED:
# • Tool Isolation: Testing tools independently separates API/scraping issues
#   from model reasoning bugs.
# • Context Budgeting: Truncating long tool returns (e.g. 500 chars) prevents
#   context window bloat in agent conversations.
# • Modern Client Usage: Using `arxiv.Client` directly to ensure compatibility
#   with recent package updates.
#
# HOW IT CONNECTS TO THE PROJECT:
# These exact tools are passed to the multi-tool chatbot in Topic 11, where the
# model chooses which tool to invoke based on user intent.
#
# THINGS TO REMEMBER:
# • Arxiv and Wikipedia are free and do not require API keys.
# • Tavily requires an API key and internet access.
#
# COMMON MISTAKES TO AVOID:
# • Returning thousands of characters of raw HTML or PDF text to an LLM,
#   wasting tokens and diluting relevance.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `search_arxiv`: notice how `client.results(search)` formats the papers
# into a clean, concise string for the model.
# ============================================================
