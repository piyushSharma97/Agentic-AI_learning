"""
=============================================================================
LANGGRAPH COMPONENT 10: SEARCH TOOLS ON THEIR OWN
=============================================================================
Before a chatbot chooses tools, call each tool yourself.
Then you know what the model will receive.

- search_arxiv  research papers. No API key. Uses the `arxiv` package directly.
- Wikipedia     short article text. No API key. Needs the `wikipedia` package.
- Tavily        web search. Needs TAVILY_API_KEY in the project .env.

The notebook used ArxivQueryRun from langchain_community. That wrapper still
calls Search.results(), which arxiv 4.0 removed (the method moved to
arxiv.Client). search_arxiv below is the same job, written so you can see
the request and the text the model would get.

Later, 11_multi_tool_chatbot.py gives these tools to the model and lets the
model pick.

HOW TO STUDY THIS FILE
main() runs one example. Comment it and uncomment the next when you are ready.
These examples call the internet.
=============================================================================
"""

import os
import warnings

import arxiv
from langchain_core.tools import tool

from config import safe_run

# The Wikipedia tool still comes from langchain_community, which prints a
# sunset warning on import. That warning is about the package, not this lesson.
warnings.filterwarnings("ignore", message=".*langchain-community.*")


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv by keywords or by a paper id such as 1706.03762."""
    client = arxiv.Client(page_size=2, delay_seconds=1.0, num_retries=2)
    cleaned = query.strip()
    # An id looks like 1706.03762. Anything else is treated as a text search.
    if cleaned.replace("v", "").replace(".", "").isdigit() and "." in cleaned:
        search = arxiv.Search(id_list=[cleaned], max_results=1)
    else:
        search = arxiv.Search(query=cleaned, max_results=2)

    papers = list(client.results(search))
    if not papers:
        return f"No arXiv paper found for {query!r}."

    blocks = []
    for paper in papers:
        summary = " ".join(paper.summary.split())[:500]
        authors = ", ".join(author.name for author in paper.authors[:5])
        blocks.append(
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Published: {paper.published.date()}\n"
            f"URL: {paper.entry_id}\n"
            f"Summary: {summary}"
        )
    return "\n\n".join(blocks)


def example_arxiv():
    """Fetch a paper. 1706.03762 is 'Attention Is All You Need'."""
    print(f"Tool name the model will see: {search_arxiv.name}")
    print(f"Tool description:\n{search_arxiv.description}\n")
    print(search_arxiv.invoke("1706.03762"))


def example_wikipedia():
    """Fetch the start of a Wikipedia page."""
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    print(f"Tool name the model will see: {wiki.name}\n")
    result = wiki.invoke("Machine learning")
    print(result)


def example_tavily():
    """Search the live web. Skips cleanly when the API key is missing."""
    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY is not in .env, so this example cannot search.")
        print("Create a key at https://tavily.com and add this line to the project .env:")
        print("    TAVILY_API_KEY=your_key_here")
        print("Arxiv and Wikipedia do not need that key.")
        return

    # langchain_tavily is the current package. The notebook's
    # TavilySearchResults import from langchain_community is the older one.
    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=3)
    print(f"Tool name the model will see: {tavily.name}\n")
    result = tavily.invoke({"query": "Recent artificial intelligence news"})
    print(result)


def main():
    # Uncomment ONE example at a time.
    safe_run("Arxiv tool", example_arxiv)
    # safe_run("Wikipedia tool", example_wikipedia)
    # safe_run("Tavily web search", example_tavily)


if __name__ == "__main__":
    main()
