"""
=============================================================================
LANGGRAPH COMPONENT 12: A CHATBOT THAT REMEMBERS THE THREAD
=============================================================================
USE
A chatbot with no tools that still knows your name on the next question.

WHY THIS APPROACH
Calling the model directly forgets the previous invoke. The memory is
the graph state, saved by a checkpointer under a thread_id.
MemorySaver is used because it needs no database and is enough to see
the idea. It lives in RAM and is gone when the script exits.
The same compiled graph serves every thread. The thread_id, not a new
graph, is what keeps Krish's chat away from someone else's chat.
There is no tool loop here so you can see memory by itself. File 07 adds tools.

This is the graph that the streaming notebook builds before it streams.

No tools. One node calls the chat model and writes the reply into messages.

Memory is not inside the model. Two pieces do the work:

1. MemorySaver
   A checkpointer. After every node, LangGraph stores a snapshot of state.

2. thread_id
   The key for that snapshot.
   Same thread_id  -> the next invoke sees the old messages.
   New thread_id   -> a blank conversation.

compile(checkpointer=memory) is what turns the snapshots on.
invoke(..., config=config) is what picks the conversation.

File 07_agent_memory_checkpointer.py is the same idea with math tools.
This file is the plain chatbot, which 13_streaming.py then streams.

HOW TO STUDY THIS FILE
main() runs one example. Comment it and uncomment the next when you are ready.
=============================================================================
"""

from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import get_llm, print_mermaid, safe_run


class State(TypedDict):
    # Use: one growing list of messages for this thread.
    # Why: the checkpointer saves this list. add_messages makes each new
    # turn append, so turn 2 still contains "my name is Krish".
    messages: Annotated[list[AnyMessage], add_messages]


def build_chatbot():
    llm = get_llm(temperature=0)

    def assistant(state: State) -> dict:
        # The model receives the whole list, including turns from earlier invokes
        # on this thread_id.
        reply = llm.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    # Use: MemorySaver as the checkpointer.
    # Why: compile is the moment memory is turned on. A graph compiled
    # without it starts empty on every invoke, even with a thread_id.
    return builder.compile(checkpointer=MemorySaver())


def show_last_reply(result: dict):
    result["messages"][-1].pretty_print()


def example_remember_across_turns():
    """Turn 2 only makes sense if turn 1 is still in the state."""
    graph = build_chatbot()
    print_mermaid(graph, "Simple chatbot")

    # Use: thread_id names the saved conversation.
    # Why: the checkpointer stores many chats. This id selects Krish's.
    # Reusing the id on turn 2 is what loads turn 1. A new id would not.
    config = {"configurable": {"thread_id": "krish-1"}}

    print("Turn 1")
    first = graph.invoke(
        {"messages": [HumanMessage(content="Hi, my name is Krish and I like cricket.")]},
        config,
    )
    show_last_reply(first)

    print("\nTurn 2 (same thread_id, so the name and the sport are still in state)")
    second = graph.invoke(
        {"messages": [HumanMessage(content="What is my name, and which sport did I say I like?")]},
        config,
    )
    show_last_reply(second)


def example_separate_threads():
    """A second thread_id cannot see the first conversation."""
    graph = build_chatbot()

    krish = {"configurable": {"thread_id": "krish"}}
    other = {"configurable": {"thread_id": "someone-else"}}

    graph.invoke(
        {"messages": [HumanMessage(content="Remember this: my favorite sport is cricket.")]},
        krish,
    )

    print("Asking on a DIFFERENT thread_id:")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What favorite sport did I just tell you?")]},
        other,
    )
    show_last_reply(result)
    print("\nThis thread never heard 'cricket'. Switch the config to `krish` and ask again.")


def main():
    # Uncomment ONE example at a time.
    safe_run("Chatbot memory on one thread", example_remember_across_turns)
    # safe_run("Two thread ids stay separate", example_separate_threads)


if __name__ == "__main__":
    main()
