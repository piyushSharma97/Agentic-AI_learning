"""
=============================================================================
LANGGRAPH COMPONENT 13: STREAMING A CHATBOT
=============================================================================
USE
Show progress while a chatbot runs, instead of waiting for the full answer.
Three views of the same graph: the new write, the whole state, and the tokens.

WHY THIS APPROACH
invoke() returns once, at the end. A UI cannot type the reply out that way.
stream(stream_mode="updates") is used when you only need what just changed,
which is the new AIMessage. stream_mode="values" is used when you need the
full history after each node, because updates hides the older turns.
Token streaming is a separate call, astream_events, because updates and
values fire when the node finishes. By then the model has already written
the whole message. Events fire while it is still writing.
The chatbot is the one from file 12 so the only new idea here is the stream.

graph.invoke() waits until the graph finishes, then returns the final state.
graph.stream() yields a piece after each node so you can show progress.

stream_mode changes WHAT each piece contains:

- "updates"  only the keys the node just wrote.
             You see the new AIMessage, not the older turns.

- "values"   the full state after the node.
             You see the whole conversation every time. It grows each turn.

Token streaming is a third thing. "updates" and "values" fire when a NODE
finishes. The model may have taken several seconds to write that message.
astream_events(..., version="v2") fires while the model is still writing.

Useful event fields:
- event     what happened. "on_chat_model_stream" means "here is the next token"
- name      which runnable emitted it
- data      the payload. For tokens, data["chunk"] is an AIMessage chunk
- metadata  includes langgraph_node, the graph node that was running

stream() is synchronous. astream() is the same modes, but async.
This file uses astream_events for tokens because that is the notebook's goal.
Build the chatbot in 12_simple_chatbot.py first if this graph is new.

The notebook printed every event and every raw chunk. That floods the terminal.
These examples print the part you are learning. A commented print(chunk) is
left in each one if you want the raw object.

HOW TO STUDY THIS FILE
main() runs one example. Comment it and uncomment the next when you are ready.
=============================================================================
"""

import asyncio
from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import get_llm, safe_run


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_chatbot():
    """Same one-node memory chatbot as 12_simple_chatbot.py."""
    llm = get_llm(temperature=0)

    def assistant(state: State) -> dict:
        reply = llm.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=MemorySaver())


def example_stream_updates():
    """Each chunk is {node_name: partial_state}."""
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-updates"}}

    print("stream_mode='updates' -> only the new write from each node.\n")
    for step, chunk in enumerate(
        # Use: stream_mode="updates".
        # Why: each chunk is only the node's new keys, shaped as
        # {node_name: partial_state}. That is the right view for a log of
        # "what just happened", and the wrong view if you need older turns.
        graph.stream(
            {"messages": [HumanMessage(content="Hi, my name is Krish and I like cricket.")]},
            config,
            stream_mode="updates",
        ),
        start=1,
    ):
        # print(chunk)  # raw chunk, the way the notebook printed it
        print(f"--- update {step} ---")
        for node_name, update in chunk.items():
            print(f"Node that just ran: {node_name}")
            for message in update.get("messages", []):
                message.pretty_print()


def example_stream_values():
    """Each chunk is the entire state. The second turn shows the history."""
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-values"}}

    print("stream_mode='values' -> full state after each node.\n")
    turns = [
        "Hi, my name is Krish and I like cricket.",
        "I also like football.",
    ]
    for turn in turns:
        print(f"\n===== user: {turn} =====")
        # Use: stream_mode="values" on the same thread for two turns.
        # Why: each chunk is the entire state. Turn 2's snapshot still
        # contains turn 1, which is how you see memory inside a stream.
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=turn)]},
            config,
            stream_mode="values",
        ):
            # print(chunk)  # raw snapshot
            print("--- full state snapshot ---")
            for message in chunk["messages"]:
                message.pretty_print()


async def _stream_tokens():
    graph = build_chatbot()
    config = {"configurable": {"thread_id": "stream-tokens"}}

    print("Tokens as the model writes them:\n")
    # Use: astream_events, and keep only on_chat_model_stream.
    # Why: this is async, so a script starts it with asyncio.run.
    # version="v2" is the current event shape. Other events are chain
    # start and chain end. Filtering keeps the tokens you would print in a UI.
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="In two sentences, what is LangGraph?")]},
        config,
        version="v2",
    ):
        # print(event)  # every event: chain start, model start, each token, chain end
        if event["event"] != "on_chat_model_stream":
            continue
        text = event["data"]["chunk"].content
        if text:
            print(text, end="", flush=True)
    print("\n")


def example_stream_tokens():
    """astream_events is async. asyncio.run starts the event loop for the script."""
    asyncio.run(_stream_tokens())


def main():
    # Uncomment ONE example at a time.
    safe_run("Stream node updates", example_stream_updates)
    # safe_run("Stream full state values", example_stream_values)
    # safe_run("Stream tokens", example_stream_tokens)


if __name__ == "__main__":
    main()
