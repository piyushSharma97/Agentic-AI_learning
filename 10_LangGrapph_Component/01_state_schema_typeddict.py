"""
=============================================================================
LANGGRAPH COMPONENT 1: STATE SCHEMA WITH TYPEDDICT
=============================================================================
USE
The state schema is the shared notebook every node reads and writes.
TypedDict is the schema you use while learning a graph: each node receives
a dict and returns a dict of the fields it wants to change.

WHY THIS APPROACH
A plain dict has no field names, so `state["nmae"]` is a silent bug.
TypedDict gives the editor those names with no runtime cost.
It does not check values while the program runs. Use file 03 (Pydantic)
when a bad webhook or form must be rejected before any node runs.

WHAT IS A STATE SCHEMA?
In LangGraph, every graph is a state machine. The "State" is the central,
shared data container passed between nodes. Every node reads from the state,
performs work, and returns updates to the state.

WHY TYPEDDICT?
1. Lightweight & Python Standard: Built into typing / typing_extensions.
2. Dictionary Interoperability: Nodes read fields with dict syntax (state["key"])
   and return standard Python dicts to update state.
3. Fast: Zero runtime validation overhead.

CRITICAL GOTCHA TO REMEMBER:
TypedDict only provides static type hints for your IDE and linters (like mypy).
Python does NOT validate types at runtime! Passing {"name": 123} will NOT crash
unless your node code specifically checks for it. (For runtime validation, use Pydantic!)
=============================================================================
"""

import random
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# 1. DEFINE THE STATE SCHEMA
# ---------------------------------------------------------------------------
class PlayerState(TypedDict):
    """
    Schema representing the state of a player in a game selection graph.
    - name: Player's name (string)
    - game: Chosen sport (either 'cricket' or 'badminton')
    - log: List of action descriptions tracking progress
    """
    name: str
    game: Literal["cricket", "badminton"]
    status: str


# ---------------------------------------------------------------------------
# 2. DEFINE NODE FUNCTIONS
# ---------------------------------------------------------------------------
# Each node takes the current state as input and returns a dictionary with
# the keys it wants to update in the state.

def start_session(state: PlayerState) -> dict:
    print(f"\n[Node: start_session] Welcome, {state['name']}!")
    # Use: return only the keys this node changes.
    # Why: LangGraph merges this dict into the state. Returning the whole
    # state would force every node to copy fields it does not touch.
    return {"status": f"{state['name']} has entered the recreation arena."}


def play_cricket(state: PlayerState) -> dict:
    print(f"[Node: play_cricket] {state['name']} picked up the bat!")
    return {
        "game": "cricket",
        "status": f"{state['name']} is batting in a T20 cricket match."
    }


def play_badminton(state: PlayerState) -> dict:
    print(f"[Node: play_badminton] {state['name']} picked up the racquet!")
    return {
        "game": "badminton",
        "status": f"{state['name']} is serving on the indoor badminton court."
    }


# ---------------------------------------------------------------------------
# 3. DEFINE CONDITIONAL ROUTING FUNCTION
# ---------------------------------------------------------------------------
# A conditional edge function inspects the state and returns the name of
# the next node to execute.

def choose_sport(state: PlayerState) -> Literal["play_cricket", "play_badminton"]:
    """
    Routes the workflow conditionally based on an explicit preference
    or a randomized decision.
    """
    selected_game = state.get("game")
    if selected_game in ("cricket", "badminton"):
        print(f"[Router: choose_sport] Player explicitly selected: {selected_game}")
        return f"play_{selected_game}"

    # Default random choice
    choice = "play_cricket" if random.random() < 0.5 else "play_badminton"
    print(f"[Router: choose_sport] No valid game pre-selected. Randomly routed to: {choice}")
    return choice


# ---------------------------------------------------------------------------
# 4. BUILD AND COMPILE THE STATEGRAPH
# ---------------------------------------------------------------------------
def build_typeddict_graph():
    # 1. Initialize graph with the State schema
    builder = StateGraph(PlayerState)

    # 2. Add nodes (Worker functions)
    builder.add_node("start_session", start_session)
    builder.add_node("play_cricket", play_cricket)
    builder.add_node("play_badminton", play_badminton)

    # 3. Add edges (Workflow flow)
    builder.add_edge(START, "start_session")

    # 4. Add conditional routing edge from 'start_session'
    # Use: a function that returns the next node name.
    # Why: a normal edge is a fixed arrow. A conditional edge is used when
    # the next step depends on the state, here the sport the player picked.
    builder.add_conditional_edges(
        "start_session",
        choose_sport,
        {
            "play_cricket": "play_cricket",
            "play_badminton": "play_badminton",
        }
    )

    # 5. Connect game nodes to END
    builder.add_edge("play_cricket", END)
    builder.add_edge("play_badminton", END)

    # 6. Compile the graph
    return builder.compile()


# ---------------------------------------------------------------------------
# 5. EXECUTION & DEMONSTRATION
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_typeddict_graph()
        print_mermaid(graph, "TypedDict Player Graph Topology")

        print("--- Test 1: Player with Explicit Game Preference ('cricket') ---")
        result1 = graph.invoke({"name": "Krish", "game": "cricket", "status": "Initiating"})
        print(f"Final State:\n  Name: {result1['name']}\n  Game: {result1['game']}\n  Status: {result1['status']}\n")

        print("--- Test 2: Player with Explicit Game Preference ('badminton') ---")
        result2 = graph.invoke({"name": "Alice", "game": "badminton", "status": "Initiating"})
        print(f"Final State:\n  Name: {result2['name']}\n  Game: {result2['game']}\n  Status: {result2['status']}\n")

        print("--- Test 3: Player without Pre-selected Game (Random Router) ---")
        result3 = graph.invoke({"name": "Bob", "status": "Initiating"})
        print(f"Final State:\n  Name: {result3['name']}\n  Game: {result3.get('game')}\n  Status: {result3['status']}\n")

        print("--- Test 4: Demonstrating Lack of Runtime Type Enforcement ---")
        # Notice: name is passed as an int (123) instead of str. TypedDict will NOT crash at runtime!
        result4 = graph.invoke({"name": 123, "game": "cricket", "status": "Testing"})
        print(f"TypedDict accepted integer name without crash: {result4['name']} (Type: {type(result4['name'])})")
        print("Reason: TypedDict is only for static analysis (IDE/mypy). For runtime guarantees, use Pydantic!\n")

    safe_run("LangGraph State Schema with TypedDict", demo)


if __name__ == "__main__":
    main()

