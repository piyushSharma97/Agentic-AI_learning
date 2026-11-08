"""
=============================================================================
LANGGRAPH COMPONENT 13: STREAMING A CHATBOT
=============================================================================
TOPIC OVERVIEW:
`graph.invoke()` waits until the entire graph finishes before returning.
In production UIs, users expect to see real-time updates while the agent works.

LangGraph provides 3 distinct streaming modalities:
1. `stream_mode="updates"`:
   Yields only the dictionary keys that the active node just wrote:
   `{ "node_name": { "messages": [new_reply] } }`
   Ideal for pipeline step indicators and audit logs.

2. `stream_mode="values"`:
   Yields the entire graph state after each node finishes:
   `{ "messages": [turn1, turn2, turn3] }`
   Ideal for UI re-renders that need the complete conversation history.

3. `astream_events(version="v2")` (Token-by-Token Streaming):
   Updates and values only fire when a NODE completes.
   `astream_events` fires while the LLM is actively generating tokens,
   enabling the classic ChatGPT typewriter effect!

FLOW:
   User Input ──► graph.stream()
                        │
                        ├─► stream_mode="updates" (Yields node delta)
                        ├─► stream_mode="values"  (Yields full state)
                        └─► astream_events()      (Yields live tokens)
=============================================================================
"""

import sys
from pathlib import Path
import asyncio
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
from config import get_llm, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE STATE & BUILD BASE CHATBOT
# ---------------------------------------------------------------------------
class State(TypedDict):
    """Conversation state using add_messages reducer."""
    messages: Annotated[list[AnyMessage], add_messages]


def build_chatbot():
    """Builds the 1-node memory chatbot from Topic 12."""
    llm = get_llm(temperature=0)

    def assistant(state: State) -> dict:
        reply = llm.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# MODE 1: STREAM NODE UPDATES (`stream_mode="updates"`)
# ---------------------------------------------------------------------------
def example_stream_updates():
    """
    Each yielded chunk has the format: `{node_name: partial_state_delta}`.
    Only shows what changed in that specific step.
    """
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-updates"}}

    print("stream_mode='updates' -> yields only what changed in each node:\n")
    for step, chunk in enumerate(
        graph.stream(
            {"messages": [HumanMessage(content="Hi, my name is Krish and I like cricket.")]},
            config,
            stream_mode="updates",
        ),
        start=1,
    ):
        print(f"--- Step {step} ---")
        for node_name, update in chunk.items():
            print(f"Node that executed: {node_name}")
            for message in update.get("messages", []):
                message.pretty_print()


# ---------------------------------------------------------------------------
# MODE 2: STREAM FULL STATE VALUES (`stream_mode="values"`)
# ---------------------------------------------------------------------------
def example_stream_values():
    """
    Each yielded chunk is the COMPLETE state snapshot after each node.
    Notice how Turn 2 includes Turn 1!
    """
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-values"}}

    print("stream_mode='values' -> yields complete state snapshot after each node:\n")
    turns = [
        "Hi, my name is Krish and I like cricket.",
        "I also like football.",
    ]
    for turn in turns:
        print(f"\n===== User: {turn} =====")
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=turn)]},
            config,
            stream_mode="values",
        ):
            print("--- Full State Snapshot ---")
            for message in chunk["messages"]:
                message.pretty_print()


# ---------------------------------------------------------------------------
# MODE 3: ASYNC TOKEN STREAMING (`astream_events`)
# ---------------------------------------------------------------------------
async def _stream_tokens():
    """
    Asynchronous event generator yielding individual tokens as the model writes them.
    Filters specifically for `on_chat_model_stream` events.
    """
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-tokens"}}

    print("Tokens streamed live as the model generates them (Typewriter effect):\n")
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="In two sentences, what is LangGraph?")]},
        config,
        version="v2",
    ):
        # We only care about chat model streaming chunks
        if event["event"] != "on_chat_model_stream":
            continue

        text_chunk = event["data"]["chunk"].content
        if text_chunk:
            print(text_chunk, end="", flush=True)
    print("\n")


def example_stream_tokens():
    """Sync wrapper to execute async event stream via asyncio.run()."""
    asyncio.run(_stream_tokens())


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    # Uncomment ONE example at a time:
    safe_run("Stream node updates (stream_mode='updates')", example_stream_updates)
    # safe_run("Stream full state values (stream_mode='values')", example_stream_values)
    # safe_run("Stream live tokens (astream_events)", example_stream_tokens)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 13: Streaming in LangGraph
# ============================================================
#
# WHAT THIS FILE DOES:
# Explores the three streaming strategies available in LangGraph:
# 1. `stream_mode="updates"`: Incremental updates per node.
# 2. `stream_mode="values"`: Full cumulative state snapshots.
# 3. `astream_events(version="v2")`: Real-time token streaming for frontend UIs.
#
# KEY CONCEPTS USED:
# • stream(): Synchronous generator yielding state chunks.
# • updates mode: Yields `{node_name: {updated_keys}}`.
# • values mode: Yields `{full_state}`.
# • astream_events: Async event stream capturing internal lifecycle events
#   (chain start, model stream, tool execution, chain end).
# • on_chat_model_stream: Event name corresponding to incremental text chunks.
#
# WHEN TO USE WHICH:
# • Use "updates" when logging agent actions or displaying progress spinners.
# • Use "values" when your frontend re-renders the entire chat history.
# • Use "astream_events" when building ChatGPT-like typewriter experiences.
#
# THINGS TO REMEMBER:
# • `astream_events` requires Python `async`/`await` and an event loop (`asyncio.run`).
# • Always specify `version="v2"` when calling `astream_events`.
#
# COMMON MISTAKES TO AVOID:
# • Expecting `stream(stream_mode="updates")` to stream words one by one
#   (it only yields when a NODE completes).
# • Forgetting `flush=True` when printing streamed tokens without newlines.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `example_stream_tokens()`:
# `async for event in graph.astream_events(..., version="v2"):`
# `  if event["event"] == "on_chat_model_stream": print(chunk, end="", flush=True)`
# ============================================================
