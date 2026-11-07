"""
=============================================================================
LANGGRAPH COMPONENT 11: CHATBOT WITH SEVERAL TOOLS
=============================================================================
The model sees Arxiv, Wikipedia, and (when you have a key) Tavily.
bind_tools does not run them. It only lets the model request one.

Graph:

    START -> assistant -> tools_condition -> tools -> assistant -> ... -> END

tools_condition:
- latest AIMessage has tool_calls -> node "tools"
- latest AIMessage is a normal answer -> END

ToolNode runs the requested tool and appends a ToolMessage.
The edge tools -> assistant is required. The notebook went tools -> END,
so the program stopped on the raw search text and the model never summarized it.

Run 10_search_tools.py first if you want to see each tool alone.

HOW TO STUDY THIS FILE
main() runs one example. Comment it and uncomment the next when you are ready.
=============================================================================
"""

import os
import warnings
from typing import Annotated

import arxiv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config import get_llm, print_mermaid, safe_run

warnings.filterwarnings("ignore", message=".*langchain-community.*")


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv by keywords or by a paper id such as 1706.03762."""
    # arxiv 4.0 moved fetching onto Client. The older ArxivQueryRun wrapper
    # still calls Search.results(), which no longer exists.
    client = arxiv.Client(page_size=2, delay_seconds=1.0, num_retries=2)
    cleaned = query.strip()
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
            f"Title: {paper.title}\nAuthors: {authors}\nSummary: {summary}"
        )
    return "\n\n".join(blocks)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_tools():
    """Tools the assistant is allowed to request."""
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    )
    tools = [search_arxiv, wiki]

    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
    else:
        print("TAVILY_API_KEY is not set. This chatbot can use Arxiv and Wikipedia only.")

    print("Tools bound to the model:")
    for item in tools:
        print(f"  - {item.name}")
    print()
    return tools


def build_chatbot(tools):
    llm_with_tools = get_llm(temperature=0).bind_tools(tools)

    def assistant(state: State) -> dict:
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


def ask(question: str):
    tools = build_tools()
    graph = build_chatbot(tools)
    print_mermaid(graph, "Multi-tool chatbot")

    print(f"User: {question}\n")
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    for message in result["messages"]:
        message.pretty_print()


def example_arxiv_paper():
    """An Arxiv id should push the model toward the Arxiv tool."""
    ask("Look up arxiv paper 1706.03762. Tell me the title and one sentence on why it matters.")


def example_wikipedia():
    """A definition question should push the model toward Wikipedia."""
    ask("What is machine learning? Answer from Wikipedia in two sentences.")


def example_recent_news():
    """Live news needs Tavily. The other two tools cannot browse today's web."""
    if not os.getenv("TAVILY_API_KEY"):
        print("Add TAVILY_API_KEY to .env before this example.")
        print("Arxiv and Wikipedia are already covered by the other two examples.")
        return
    ask("What is a recent headline about artificial intelligence? Use web search.")


def main():
    # Uncomment ONE example at a time.
    safe_run("Multi-tool chatbot: Arxiv paper", example_arxiv_paper)
    # safe_run("Multi-tool chatbot: Wikipedia", example_wikipedia)
    # safe_run("Multi-tool chatbot: recent news", example_recent_news)


if __name__ == "__main__":
    main()
