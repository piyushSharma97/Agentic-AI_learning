"""
=============================================================================
LANGGRAPH COMPONENT 12: A CHATBOT THAT REMEMBERS THE THREAD
=============================================================================
TOPIC OVERVIEW:
This module demonstrates a clean, tool-free conversational chatbot in LangGraph.
It shows how a simple 1-node graph (`START -> assistant -> END`) achieves
conversational memory using `MemorySaver` and `thread_id`.

GRAPH TOPOLOGY:
    [START] ──> [assistant (LLM)] ──> [END]
                       │
                       ▼ (Snapshot saved to checkpointer)
                 [MemorySaver]

WHY THIS MATTERS:
This is the minimal, canonical baseline for conversational graphs.
In Topic 13, this exact graph is used to demonstrate real-time streaming!
=============================================================================
"""

import sys
from pathlib import Path
from typing import Annotated

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from config import get_llm, print_mermaid, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE STATE WITH ADD_MESSAGES REDUCER
# ---------------------------------------------------------------------------
class State(TypedDict):
    """
    State schema holding the conversation history.
    `add_messages` appends new turns so previous questions and answers are kept.
    """
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# STEP 2: BUILD THE ONE-NODE PERSISTENT CHATBOT
# ---------------------------------------------------------------------------
def build_chatbot():
    """
    Builds a 1-node conversational graph and compiles with MemorySaver checkpointer.
    """
    llm = get_llm(temperature=0)

    def assistant(state: State) -> dict:
        # LLM receives the full message thread loaded by the checkpointer
        reply = llm.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    # Attach MemorySaver checkpointer at compile time!
    return builder.compile(checkpointer=MemorySaver())


def show_last_reply(result: dict):
    """Helper to display the latest message formatted cleanly."""
    result["messages"][-1].pretty_print()


# ---------------------------------------------------------------------------
# DEMO 1: MULTI-TURN CONTINUITY ON A SINGLE THREAD
# ---------------------------------------------------------------------------
def example_remember_across_turns():
    """Turn 2 only makes sense if Turn 1 is preserved in state."""
    graph = build_chatbot()
    print_mermaid(graph, "Simple Memory Chatbot Topology")

    # Thread ID identifies Krish's conversation
    config = {"configurable": {"thread_id": "krish-session-1"}}

    print("--- Turn 1 ---")
    first = graph.invoke(
        {"messages": [HumanMessage(content="Hi, my name is Krish and I like cricket.")]},
        config,
    )
    show_last_reply(first)

    print("\n--- Turn 2 (Same thread_id: remembers name and favorite sport!) ---")
    second = graph.invoke(
        {"messages": [HumanMessage(content="What is my name, and which sport did I say I like?")]},
        config,
    )
    show_last_reply(second)


# ---------------------------------------------------------------------------
# DEMO 2: THREAD ISOLATION (NO DATA LEAKAGE)
# ---------------------------------------------------------------------------
def example_separate_threads():
    """Demonstrates that a different thread_id starts with a blank slate."""
    graph = build_chatbot()

    config_krish = {"configurable": {"thread_id": "krish"}}
    config_other = {"configurable": {"thread_id": "someone-else"}}

    print("--- Storing secret in thread 'krish' ---")
    graph.invoke(
        {"messages": [HumanMessage(content="Remember this: my favorite sport is cricket.")]},
        config_krish,
    )
    print("Stored in 'krish' thread.\n")

    print("--- Asking from a DIFFERENT thread 'someone-else' ---")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What favorite sport did I just tell you?")]},
        config_other,
    )
    show_last_reply(result)
    print("Notice: Thread 'someone-else' knows nothing about 'krish'!\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_run("Chatbot memory on one thread", example_remember_across_turns)
    safe_run("Two thread IDs stay strictly separate", example_separate_threads)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 12: Simple Chatbot Memory
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates a clean, 1-node conversational chatbot in LangGraph.
# Shows how compiling with `checkpointer=MemorySaver()` and providing
# a `thread_id` enables multi-turn memory without manual state passing.
#
# KEY CONCEPTS USED:
# • 1-Node Chatbot: `START -> assistant -> END`.
# • MemorySaver(): In-memory state persistence layer.
# • thread_id: Identifies each unique conversation thread.
# • Session Isolation: Different thread_ids maintain completely separate histories.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file serves as the minimal baseline for multi-turn chats, and is the
# exact graph used in Topic 13 to study streaming modes.
#
# THINGS TO REMEMBER:
# • Passing `thread_id` without compiling with a checkpointer will NOT retain memory.
# • Both `compile(checkpointer=...)` and `config={"configurable": {"thread_id": ...}}`
#   are required together!
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `add_messages` on the state schema.
# • Hardcoding a single thread_id for all users in a multi-user app.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `example_remember_across_turns`:
# Turn 1 and Turn 2 both pass `config = {"configurable": {"thread_id": "krish-session-1"}}`.
# ============================================================
