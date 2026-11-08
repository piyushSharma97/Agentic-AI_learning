"""
=============================================================================
LANGGRAPH COMPONENT 1: STATE SCHEMA WITH TYPEDDICT
=============================================================================
TOPIC OVERVIEW:
The State Schema is the single source of truth across your entire LangGraph.
It is the shared data container that is passed from node to node. Every node
reads from the state, performs its computation, and returns updates to the state.

WHY TYPEDDICT?
1. Python Standard: Built directly into Python's `typing` / `typing_extensions`.
2. Dictionary Interoperability: Nodes access fields using dictionary syntax:
   `state["name"]`, and return standard Python dictionaries to update state.
3. Zero Overhead: Extremely fast because it does not perform runtime checks.

EXECUTION FLOW:
    [START] 
       │
       ▼
 [start_session] (Emits entry status)
       │
       ▼ (Conditional Router: choose_sport)
     ┌─┴─────────────┐
     │               │
  "cricket"      "badminton"
     │               │
     ▼               ▼
[play_cricket]  [play_badminton]
     │               │
     └───────┬───────┘
             ▼
           [END]

KEY GOTCHA TO REMEMBER:
TypedDict only provides static type hints for your IDE and linters (like mypy).
Python does NOT validate types at runtime! Passing {"name": 123} will NOT crash
unless your code manually checks for it. (For runtime validation, use Pydantic in Topic 03!)
=============================================================================
"""

import sys
from pathlib import Path
import random
from typing import Literal
from typing_extensions import TypedDict

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langgraph.graph import StateGraph, START, END
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE THE STATE SCHEMA
# ---------------------------------------------------------------------------
# The state schema defines all fields that will ever exist in this graph.
# Here we define a PlayerState tracking a player's name, chosen sport, and current status.

class PlayerState(TypedDict):
    """
    Schema representing the state of a player in a game selection graph.
    - name: Player's name (str)
    - game: Chosen sport (either 'cricket' or 'badminton')
    - status: Human-readable log tracking current progress (str)
    """
    name: str
    game: Literal["cricket", "badminton"]
    status: str


# ---------------------------------------------------------------------------
# STEP 2: DEFINE NODE FUNCTIONS (WORKERS)
# ---------------------------------------------------------------------------
# In LangGraph, a node is a regular Python function that:
# 1. Takes the current state dictionary as its input argument: `state: PlayerState`
# 2. Performs some action or calculation
# 3. Returns a DICTIONARY with ONLY the keys it wants to update in the state.
#    (You do NOT need to return the entire state — LangGraph merges your updates!)

def start_session(state: PlayerState) -> dict:
    """
    Node 1: Entry Point.
    Reads player name from state, welcomes them, and updates the status.
    """
    print(f"\n[Node: start_session] Welcome, {state['name']}!")
    # Return only the 'status' key that changed; LangGraph automatically merges this.
    return {"status": f"{state['name']} has entered the recreation arena."}


def play_cricket(state: PlayerState) -> dict:
    """
    Node 2A: Cricket Branch.
    Updates game to 'cricket' and updates the status message.
    """
    print(f"[Node: play_cricket] {state['name']} picked up the bat!")
    return {
        "game": "cricket",
        "status": f"{state['name']} is batting in a T20 cricket match."
    }


def play_badminton(state: PlayerState) -> dict:
    """
    Node 2B: Badminton Branch.
    Updates game to 'badminton' and updates the status message.
    """
    print(f"[Node: play_badminton] {state['name']} picked up the racquet!")
    return {
        "game": "badminton",
        "status": f"{state['name']} is serving on the indoor badminton court."
    }


# ---------------------------------------------------------------------------
# STEP 3: DEFINE CONDITIONAL ROUTING FUNCTION
# ---------------------------------------------------------------------------
# A conditional router function inspects the current state and returns
# a string matching the name of the next node to execute.

def choose_sport(state: PlayerState) -> Literal["play_cricket", "play_badminton"]:
    """
    Conditional edge router:
    - If the user explicitly provided 'cricket' or 'badminton', respect their choice.
    - If no game was specified, select a sport at random (50/50 probability).
    """
    selected_game = state.get("game")
    if selected_game in ("cricket", "badminton"):
        print(f"[Router: choose_sport] Player explicitly selected: {selected_game}")
        return f"play_{selected_game}"

    # Fallback: make a random choice if no sport was pre-selected
    choice = "play_cricket" if random.random() < 0.5 else "play_badminton"
    print(f"[Router: choose_sport] No valid game pre-selected. Randomly routed to: {choice}")
    return choice


# ---------------------------------------------------------------------------
# STEP 4: ASSEMBLE AND COMPILE THE STATEGRAPH
# ---------------------------------------------------------------------------
def build_typeddict_graph():
    """
    Builds the graph structure:
    1. Instantiates StateGraph with PlayerState schema
    2. Registers nodes
    3. Wires static edges and conditional routing edges
    4. Compiles into an executable Runnable graph
    """
    # 1. Initialize graph with the State schema
    builder = StateGraph(PlayerState)

    # 2. Add nodes (Worker functions)
    builder.add_node("start_session", start_session)
    builder.add_node("play_cricket", play_cricket)
    builder.add_node("play_badminton", play_badminton)

    # 3. Add static edge: Graph entry point goes straight to start_session
    builder.add_edge(START, "start_session")

    # 4. Add conditional routing edge from 'start_session'
    # Based on the return value of choose_sport, route to the corresponding node:
    builder.add_conditional_edges(
        "start_session",
        choose_sport,
        {
            "play_cricket": "play_cricket",
            "play_badminton": "play_badminton",
        }
    )

    # 5. Connect both game nodes to END (terminal point of the graph)
    builder.add_edge("play_cricket", END)
    builder.add_edge("play_badminton", END)

    # 6. Compile the graph into an executable runnable
    return builder.compile()


# ---------------------------------------------------------------------------
# STEP 5: EXECUTION & DEMONSTRATION
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_typeddict_graph()
        print_mermaid(graph, "TypedDict Player Graph Topology")

        # Test Case 1: Player explicitly chose cricket
        print("--- Test 1: Player with Explicit Game Preference ('cricket') ---")
        result1 = graph.invoke({"name": "Krish", "game": "cricket", "status": "Initiating"})
        print(f"Final State:\n  Name: {result1['name']}\n  Game: {result1['game']}\n  Status: {result1['status']}\n")

        # Test Case 2: Player explicitly chose badminton
        print("--- Test 2: Player with Explicit Game Preference ('badminton') ---")
        result2 = graph.invoke({"name": "Alice", "game": "badminton", "status": "Initiating"})
        print(f"Final State:\n  Name: {result2['name']}\n  Game: {result2['game']}\n  Status: {result2['status']}\n")

        # Test Case 3: Player without pre-selected game (router will pick randomly)
        print("--- Test 3: Player without Pre-selected Game (Random Router) ---")
        result3 = graph.invoke({"name": "Bob", "status": "Initiating"})
        print(f"Final State:\n  Name: {result3['name']}\n  Game: {result3.get('game')}\n  Status: {result3['status']}\n")

        # Test Case 4: Demonstrating that TypedDict has NO runtime validation!
        print("--- Test 4: Demonstrating Lack of Runtime Type Enforcement ---")
        # Notice: name is passed as an integer (123) instead of string. TypedDict will NOT crash!
        result4 = graph.invoke({"name": 123, "game": "cricket", "status": "Testing"})
        print(f"TypedDict accepted integer name without crash: {result4['name']} (Type: {type(result4['name'])})")
        print("Takeaway: TypedDict is static only (IDE/mypy). For strict runtime validation, use Pydantic (Topic 03)!\n")

    safe_run("LangGraph State Schema with TypedDict", demo)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 01: TypedDict State Schema
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates defining LangGraph state using Python's `TypedDict`. It builds
# a game selection graph where nodes receive state as a dictionary, return
# partial updates, and use a conditional router to branch execution.
#
# KEY CONCEPTS USED:
# • StateGraph(PlayerState): Initializes a graph with a typed dictionary contract.
# • Partial Updates: Node functions return `{"status": "..."}`, modifying only
#   specific keys while LangGraph preserves other untouched keys.
# • add_conditional_edges: Branches workflow execution based on the return
#   string of a router function.
# • Static vs Runtime: TypedDict gives IDE code hints but does not enforce
#   types at runtime.
#
# THINGS TO REMEMBER:
# • Access state values via dictionary indexing: `state["field"]` or `state.get("field")`.
# • Never return unneeded keys; return only what the node modifies.
#
# COMMON MISTAKES TO AVOID:
# • Relying on TypedDict to validate incoming user requests or API JSON.
# • Mismatching dictionary keys returned by the router with the target map in `add_conditional_edges`.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `PlayerState(TypedDict)` -> see how `start_session` reads `state['name']`
# and returns `{"status": ...}` -> see how `build_typeddict_graph` connects them.
# ============================================================
