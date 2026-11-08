"""
=============================================================================
LANGGRAPH COMPONENT 11: CHATBOT WITH SEVERAL TOOLS
=============================================================================
TOPIC OVERVIEW:
This module demonstrates an autonomous chatbot equipped with multiple
specialized tools (arXiv research, Wikipedia definitions, and Tavily live search).

WHY THIS APPROACH?
Writing manual string parsing (`if "arxiv" in query: call_arxiv()`) fails
as soon as the user says "Look up that 2017 attention paper".
By binding all tools at once with `llm.bind_tools(tools)`, the LLM reads each
tool's docstring and autonomously selects the best tool for the question.

EXECUTION FLOW:
    [START] 
       │
       ▼
  [assistant]  (Evaluates query & decides which tool to call)
       │
       ▼ (tools_condition)
     ┌─┴────────────────┐
     │                  │
(tool_calls emitted) (No tools needed)
     │                  │
     ▼                  ▼
  [tools]             [END]
     │ (ToolNode runs arXiv/Wiki/Tavily)
     │
     └──────> [assistant] (Synthesizes raw tool observation into final reply)
=============================================================================
"""

import sys
from pathlib import Path
import os
import warnings
from typing import Annotated

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import arxiv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from config import get_llm, print_mermaid, safe_run

warnings.filterwarnings("ignore", message=".*langchain-community.*")


# ---------------------------------------------------------------------------
# STEP 1: DEFINE SEARCH TOOLS
# ---------------------------------------------------------------------------
@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for scientific research papers by keywords or paper ID (e.g. 1706.03762)."""
    client = arxiv.Client(page_size=2, delay_seconds=1.0, num_retries=2)
    cleaned = query.strip()
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
            summary = " ".join(paper.summary.split())[:500]
            authors = ", ".join(author.name for author in paper.authors[:5])
            blocks.append(
                f"Title: {paper.title}\nAuthors: {authors}\nSummary: {summary}..."
            )
        return "\n\n".join(blocks)
    except Exception as e:
        return f"arXiv search error: {e}"


# ---------------------------------------------------------------------------
# STEP 2: DEFINE STATE SCHEMA
# ---------------------------------------------------------------------------
class State(TypedDict):
    """Message history state using add_messages to track queries and tool replies."""
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# STEP 3: ASSEMBLE TOOL SUITE DYNAMICALLY
# ---------------------------------------------------------------------------
def build_tools():
    """Builds the active tool list based on available packages and API keys."""
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    )
    tools = [search_arxiv, wiki]

    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch
            tools.append(TavilySearch(max_results=3))
            print("Loaded Tools: [search_arxiv, wikipedia, tavily]")
        except ImportError:
            print("Loaded Tools: [search_arxiv, wikipedia]")
    else:
        print("TAVILY_API_KEY is not set. Chatbot will use [search_arxiv, wikipedia].")

    return tools


# ---------------------------------------------------------------------------
# STEP 4: BUILD CHATBOT GRAPH
# ---------------------------------------------------------------------------
def build_chatbot(tools):
    """Wires assistant node, ToolNode, and the cyclic ReAct loop."""
    llm_with_tools = get_llm(temperature=0).bind_tools(tools)

    def assistant(state: State) -> dict:
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    # Loop back to assistant so the model synthesizes the raw tool observations
    builder.add_edge("tools", "assistant")

    return builder.compile()


def ask(question: str):
    """Executes a question through the multi-tool chatbot."""
    tools = build_tools()
    graph = build_chatbot(tools)
    print_mermaid(graph, "Multi-Tool Chatbot Topology")

    print(f"User: {question}\n")
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    for message in result["messages"]:
        message.pretty_print()


# ---------------------------------------------------------------------------
# DEMONSTRATION EXAMPLES
# ---------------------------------------------------------------------------
def example_arxiv_paper():
    """An arXiv ID pushes the model toward search_arxiv."""
    ask("Look up arxiv paper 1706.03762. Tell me the title and why it matters.")


def example_wikipedia():
    """A definition question pushes the model toward Wikipedia."""
    ask("What is machine learning? Answer from Wikipedia in two sentences.")


def main():
    safe_run("Multi-tool chatbot: arXiv query", example_arxiv_paper)
    safe_run("Multi-tool chatbot: Wikipedia query", example_wikipedia)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 11: Multi-Tool Chatbot
# ============================================================
#
# WHAT THIS FILE DOES:
# Connects multiple search tools (arXiv, Wikipedia, Tavily) to an LLM
# inside a cyclic LangGraph architecture. The model dynamically evaluates
# user questions and selects the most relevant tool without manual if-statements.
#
# KEY CONCEPTS USED:
# • Multi-Tool Binding: `llm.bind_tools([t1, t2, t3])`.
# • Semantic Routing: LLM reads function signatures and docstrings to choose tools.
# • ToolNode Integration: Executes whichever tool was selected.
# • Synthesis Loop: Returns `tools -> assistant` so the LLM summarizes the output.
#
# THINGS TO REMEMBER:
# • Docstrings are the only information the model has to distinguish between tools.
# • Write clear, distinct docstrings (e.g. "Search research papers" vs "Search encyclopedia").
#
# COMMON MISTAKES TO AVOID:
# • Overlapping tool descriptions that confuse the model.
# • Terminating the graph at `tools`, which exposes raw search snippets to the user.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `build_chatbot(tools)`:
# `builder.add_node("tools", ToolNode(tools))`
# `builder.add_conditional_edges("assistant", tools_condition)`
# `builder.add_edge("tools", "assistant")`
# ============================================================
