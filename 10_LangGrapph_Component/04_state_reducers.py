"""
=============================================================================
LANGGRAPH COMPONENT 4: STATE REDUCERS (OVERWRITING VS ACCUMULATING)
=============================================================================
THE PROBLEM:
By default in LangGraph, if node A returns `{"score": 10}` and node B returns
`{"score": 20}`, what is the final score?
-> It is 20! LangGraph OVERWRITES state keys by default.

For conversational agents, this default behavior would be disastrous:
If Step 1 creates [HumanMessage("Hello")], and Step 2 returns [AIMessage("Hi")],
the list would be OVERWRITTEN and the user's initial question would be wiped out!

THE SOLUTION: REDUCERS (via `typing.Annotated`)
Reducers tell LangGraph HOW to combine new node updates with the existing state:
1. `Annotated[list, operator.add]`: Appends new list items to the existing list.
2. `Annotated[int, operator.add]`: Adds the new integer to the existing integer.
3. `Annotated[list[AnyMessage], add_messages]`: The standard agent reducer.
   It appends messages, deduplicates by message ID, and replaces messages if IDs match.
=============================================================================
"""

import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


# ===========================================================================
# 1. GRAPH WITHOUT REDUCER (DEFAULT OVERWRITE BEHAVIOR)
# ===========================================================================
class OverwriteState(TypedDict):
    # Standard list without reducer annotation
    messages: List[str]


def node_step_one(state: OverwriteState) -> dict:
    print("[Node 1] Emitting: ['Step 1 complete']")
    return {"messages": ["Step 1 complete"]}


def node_step_two(state: OverwriteState) -> dict:
    print("[Node 2] Emitting: ['Step 2 complete']")
    return {"messages": ["Step 2 complete"]}


def demo_without_reducer():
    print("--- 1. Graph WITHOUT Reducer (Demonstrating Overwrite) ---")
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
# 2. GRAPH WITH REDUCER (ACCUMULATING / APPENDING)
# ===========================================================================
class AppendingState(TypedDict):
    # Annotated with operator.add: new elements are concatenated to the existing list!
    log: Annotated[List[str], operator.add]
    # Annotated with operator.add for numbers: running counter accumulator!
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
# 3. THE AGENT REDUCER: add_messages
# ===========================================================================
class ChatAgentState(TypedDict):
    # Built-in LangGraph message reducer
    messages: Annotated[List[AnyMessage], add_messages]


def mock_user_input(state: ChatAgentState) -> dict:
    return {"messages": [HumanMessage(content="Hello AI, how is the weather?")]}


def mock_ai_response(state: ChatAgentState) -> dict:
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
    print(f"Total messages accumulated: {len(res['messages'])}")
    for idx, msg in enumerate(res["messages"], start=1):
        print(f"  [{idx}] {msg.type.upper()}: {msg.content}")
    print("\n>> add_messages automatically preserved the entire conversation thread!\n")


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

