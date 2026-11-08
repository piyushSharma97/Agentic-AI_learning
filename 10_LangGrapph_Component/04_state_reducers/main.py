"""
=============================================================================
LANGGRAPH COMPONENT 4: STATE REDUCERS (OVERWRITING VS ACCUMULATING)
=============================================================================
TOPIC OVERVIEW:
This module tackles the #1 conceptual trap in LangGraph: State Reducers.

THE DEFAULT PROBLEM:
By default, LangGraph OVERWRITES state keys. If Node A sets `{"score": 10}`,
and Node B sets `{"score": 20}`, the final score is 20!
In a chat application, if Step 1 sets `[HumanMessage("Hello")]` and Step 2 sets
`[AIMessage("Hi")]`, the user's question would be permanently wiped out!

THE SOLUTION: `typing.Annotated` REDUCERS
Reducers tell LangGraph HOW to combine new node updates with the existing state:
1. `Annotated[List[str], operator.add]`: Appends new items to existing lists.
2. `Annotated[int, operator.add]`: Sums numbers together (e.g. running scores).
3. `Annotated[List[AnyMessage], add_messages]`: The standard agent reducer that
   appends messages, manages message IDs, and prevents duplicate turns.

FLOWS DEMONSTRATED:
1. Overwrite Demo:
   [START] ──> [step_one (emits 'Step 1')] ──> [step_two (emits 'Step 2')] ──> [END]
   Result: Only ['Step 2'] survives!

2. operator.add Demo:
   [START] ──> [round_one (+15)] ──> [round_two (+25)] ──> [END]
   Result: Both logs kept, total score = 40!

3. add_messages Demo:
   [START] ──> [user (HumanMessage)] ──> [ai (AIMessage)] ──> [END]
   Result: Full conversation thread preserved!
=============================================================================
"""

import sys
from pathlib import Path
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from config import print_mermaid, safe_run


# ===========================================================================
# DEMO 1: GRAPH WITHOUT REDUCER (DEMONSTRATING THE OVERWRITE TRAP)
# ===========================================================================
class OverwriteState(TypedDict):
    """
    Notice: `messages` is just `List[str]` without `Annotated`.
    Every node that returns `messages` will completely replace the previous value!
    """
    messages: List[str]


def node_step_one(state: OverwriteState) -> dict:
    print("[Node 1] Emitting: ['Step 1 complete']")
    return {"messages": ["Step 1 complete"]}


def node_step_two(state: OverwriteState) -> dict:
    print("[Node 2] Emitting: ['Step 2 complete']")
    return {"messages": ["Step 2 complete"]}


def demo_without_reducer():
    print("--- 1. Graph WITHOUT Reducer (Demonstrating Overwrite Trap) ---")
    builder = StateGraph(OverwriteState)
    builder.add_node("step_one", node_step_one)
    builder.add_node("step_two", node_step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    graph = builder.compile()

    initial_state = {"messages": ["Initial user request"]}
    print(f"Initial State: {initial_state['messages']}")

    result = graph.invoke(initial_state)
    print(f"\nFinal State Result: {result['messages']}")
    print(">> NOTICE: 'Initial user request' and 'Step 1 complete' were completely WIPED OUT!")
    print(">> Only 'Step 2 complete' remains because the default behavior is OVERWRITE.\n")


# ===========================================================================
# DEMO 2: GRAPH WITH operator.add REDUCER (LIST & INTEGER ACCUMULATION)
# ===========================================================================
class AppendingState(TypedDict):
    """
    Annotated[T, reducer_function] tells LangGraph how to merge new state updates.
    - log: uses `operator.add` to concatenate new lists onto the existing list.
    - total_score: uses `operator.add` to add new numbers onto the existing running total.
    """
    log: Annotated[List[str], operator.add]
    total_score: Annotated[int, operator.add]


def score_round_one(state: AppendingState) -> dict:
    print("[Score Node 1] Logging action and adding 15 points...")
    return {
        "log": ["Round 1: Hit target"],
        "total_score": 15
    }


def score_round_two(state: AppendingState) -> dict:
    print("[Score Node 2] Logging action and adding 25 points...")
    return {
        "log": ["Round 2: Bonus multiplier"],
        "total_score": 25
    }


def demo_with_list_and_int_reducer():
    print("--- 2. Graph WITH operator.add Reducer (List & Integer Accumulation) ---")
    builder = StateGraph(AppendingState)
    builder.add_node("round_one", score_round_one)
    builder.add_node("round_two", score_round_two)
    builder.add_edge(START, "round_one")
    builder.add_edge("round_one", "round_two")
    builder.add_edge("round_two", END)
    graph = builder.compile()

    initial = {"log": ["Game Started"], "total_score": 0}
    print(f"Initial State: log={initial['log']}, score={initial['total_score']}")

    result = graph.invoke(initial)
    print(f"\nFinal State Result:")
    print(f"  Log (Accumulated): {result['log']}")
    print(f"  Total Score (Accumulated): {result['total_score']} (0 + 15 + 25)")
    print(">> NOTICE: Nothing was lost! Both strings and integers accumulated seamlessly.\n")


# ===========================================================================
# DEMO 3: THE CHAT AGENT REDUCER: add_messages
# ===========================================================================
class ChatAgentState(TypedDict):
    """
    Why not use `operator.add` for messages?
    `operator.add` blindly glues lists together without understanding message IDs.
    `add_messages`:
    1. Appends new messages.
    2. Deduplicates messages by message ID.
    3. Replaces messages in-place if an incoming message shares an existing ID!
    """
    messages: Annotated[List[AnyMessage], add_messages]


def mock_user_input(state: ChatAgentState) -> dict:
    print("[Node: user] User asks: 'Hello AI, how is the weather?'")
    return {"messages": [HumanMessage(content="Hello AI, how is the weather?")]}


def mock_ai_response(state: ChatAgentState) -> dict:
    print("[Node: ai] Assistant answers: 'The weather is sunny and 25°C.'")
    return {"messages": [AIMessage(content="The weather is sunny and 25°C.")]}


def demo_add_messages_reducer():
    print("--- 3. The Chat Agent Reducer: add_messages ---")
    builder = StateGraph(ChatAgentState)
    builder.add_node("user", mock_user_input)
    builder.add_node("ai", mock_ai_response)
    builder.add_edge(START, "user")
    builder.add_edge("user", "ai")
    builder.add_edge("ai", END)
    graph = builder.compile()

    res = graph.invoke({"messages": []})
    print(f"\nTotal messages in conversation thread: {len(res['messages'])}")
    for idx, msg in enumerate(res["messages"], start=1):
        print(f"  [{idx}] {msg.type.upper()}: {msg.content}")
    print("\n>> add_messages preserved the entire conversation thread without overwriting!\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_run("LangGraph State Reducers (Overwriting vs Accumulating)", lambda: [
        demo_without_reducer(),
        demo_with_list_and_int_reducer(),
        demo_add_messages_reducer(),
    ])


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 04: State Reducers
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates how LangGraph merges state updates. Compares the default
# overwrite behavior against accumulating reducers using `typing.Annotated`
# with `operator.add` and LangGraph's specialized `add_messages` function.
#
# KEY CONCEPTS USED:
# • Default Overwrite: Without Annotated, any node returning a key replaces it.
# • typing.Annotated[type, reducer]: Attaches reduction metadata to a field.
# • operator.add: Standard library function for list concatenation and integer addition.
# • add_messages: Dedicated LangGraph reducer that merges message lists,
#   tracks message IDs, and handles in-place message updates.
#
# WHY THIS MATTERS FOR AGENTS:
# If you build a chatbot without `add_messages`, your bot will suffer from amnesia
# on turn 2 because the assistant's reply will overwrite the human's question!
#
# THINGS TO REMEMBER:
# • You can write custom reducers: `def my_reducer(old, new): return ...`
# • A reducer is called whenever a node returns that specific key.
#
# COMMON MISTAKES TO AVOID:
# • Using `messages: list` without `Annotated[..., add_messages]` in chatbot states.
# • Using `operator.add` for message arrays, which creates duplicate messages
#   when retrying or streaming.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Run `demo_without_reducer()` and look at the printed output: only the last message survived.
# Then look at `ChatAgentState` to see how `add_messages` fixes it.
# ============================================================
