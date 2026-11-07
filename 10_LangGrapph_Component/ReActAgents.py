"""
=============================================================================
ReAct Agent Architecture & Memory in LangGraph
=============================================================================
ReAct = Reasoning + Acting
1. ACT: Model decides to call specific tools based on the question.
2. OBSERVE: ToolNode executes the tool and passes output back as a ToolMessage.
3. REASON: Model reasons over the observation to either call another tool or respond directly.

This file provides a unified, crash-free implementation of:
1. Custom math tools & optional knowledge tools
2. ReAct Agent Graph with `ToolNode` and `tools_condition`
3. State persistence and multi-turn memory using `MemorySaver`
=============================================================================
"""

import os
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import shared helper
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# 1. TOOL DEFINITIONS
# ===========================================================================
@tool
def add(a: float, b: float) -> float:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies a and b."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divides a and b."""
    if b == 0:
        return "Error: Division by zero."
    return str(round(a / b, 4))


@tool
def search_knowledge(query: str) -> str:
    """
    Search AI and science facts (Transformers, Machine Learning, etc.).
    Falls back gracefully if external search packages are not installed.
    """
    kb = {
        "attention": "'Attention Is All You Need' (2017) introduced the Transformer architecture.",
        "machine learning": "Machine learning is the study of computer algorithms that improve automatically through experience.",
    }
    q_lower = query.lower()
    for k, v in kb.items():
        if k in q_lower:
            return v
    return f"Knowledge lookup result for '{query}': Found general literature reference."


# Gather active tools
active_tools = [add, multiply, divide, search_knowledge]


# ===========================================================================
# 2. STATE SCHEMA
# ===========================================================================
class State(TypedDict):
    # add_messages handles appending, deduplication, and message history
    messages: Annotated[List[AnyMessage], add_messages]


# ===========================================================================
# 3. GRAPH BUILDER FACTORY
# ===========================================================================
def create_react_graph(checkpointer=None):
    """
    Builds the ReAct graph.
    If checkpointer is provided, the compiled graph possesses multi-turn memory!
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(active_tools)

    def tool_calling_llm(state: State) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(active_tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# 4. EXECUTION DEMO
# ===========================================================================
def demo_react_agents():
    # Part 1: Stateless ReAct Agent
    print("--- 1. Stateless ReAct Agent (Single Turn Multi-Step) ---")
    graph = create_react_graph()
    print_mermaid(graph, "Stateless ReAct Graph")

    query = "Add 5 plus 5 and then multiply that result by 10."
    print(f"User: {query}")
    res = graph.invoke({"messages": [HumanMessage(content=query)]})
    for m in res["messages"]:
        m.pretty_print()

    # Part 2: Stateful ReAct Agent with MemorySaver
    print("\n--- 2. Stateful ReAct Agent with MemorySaver (Multi-Turn) ---")
    memory = MemorySaver()
    graph_with_memory = create_react_graph(checkpointer=memory)

    config = {"configurable": {"thread_id": "thread_demo_1"}}

    # Turn 1
    t1 = "Add 12 and 13."
    print(f"\n[Turn 1] User: {t1}")
    res1 = graph_with_memory.invoke({"messages": [HumanMessage(content=t1)]}, config=config)
    print(f"Agent: {res1['messages'][-1].content}")

    # Turn 2 (Context-dependent query!)
    t2 = "Divide that by 5."
    print(f"\n[Turn 2] User: {t2}")
    res2 = graph_with_memory.invoke({"messages": [HumanMessage(content=t2)]}, config=config)
    print(f"Agent: {res2['messages'][-1].content}")


def main():
    safe_run("ReAct Agents and MemorySaver in LangGraph", demo_react_agents)


if __name__ == "__main__":
    main()
