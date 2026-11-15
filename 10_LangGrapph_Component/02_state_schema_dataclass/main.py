"""
=============================================================================
LANGGRAPH COMPONENT 2: STATE SCHEMA WITH PYTHON DATACLASSES
=============================================================================
TOPIC OVERVIEW:
While Topic 01 used TypedDict (dictionary-based syntax), this module teaches
how to define state using Python's native `@dataclass` decorator.

WHY USE DATACLASSES FOR STATE?
1. Dot Notation Access: Access fields via `state.name` instead of `state["name"]`,
   which prevents string typos and enables full IDE autocomplete.
2. Default Values: Easily define default initial state values (`score: int = 0`).
3. Encapsulated Methods: Add helper methods and computed properties directly on
   the state class (e.g. `state.summary()`).

HOW LANGGRAPH APPLIES UPDATES:
Even though the state is a Dataclass instance, your graph nodes can STILL return
standard Python dictionaries: `{"score": state.score + 5}`. LangGraph automatically
merges those dictionary updates onto the fields of the Dataclass!

FLOW OF EXECUTION:
    [START] 
       │
       ▼
   [warm_up] (Adds +5 score, updates status)
       │
       ▼ (Conditional Router: decide_sport)
     ┌─┴─────────────┐
     │               │
  "cricket"      "badminton"
     │               │
     ▼               ▼
[play_cricket]  [play_badminton]
 (+50 score)     (+30 score)
     │               │
     └───────┬───────┘
             ▼
           [END]
=============================================================================
"""

import sys
from pathlib import Path
import random
from dataclasses import dataclass
from typing import Literal, Optional

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langgraph.graph import StateGraph, START, END
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE THE STATE SCHEMA AS A DATACLASS
# ---------------------------------------------------------------------------
# The @dataclass decorator automatically generates __init__, __repr__, and __eq__.
# Notice that default values (e.g. score=0) can be provided directly!

@dataclass
class PlayerDataClassState:
    """
    Dataclass State Schema representing an active player.
    - name: Required string identifying the player
    - game: Optional sport preference ('cricket' or 'badminton')
    - score: Running score/points tally with default 0
    - status: Operational status string with default 'Initialized'
    """
    name: str
    game: Optional[Literal["cricket", "badminton"]] = None
    score: int = 0
    status: str = "Initialized"

    def summary(self) -> str:
        """Helper method on the state object for clean formatting."""
        return f"Player: {self.name} | Sport: {self.game} | Total Score: {self.score}"


# ---------------------------------------------------------------------------
# STEP 2: DEFINE NODE FUNCTIONS USING DOT NOTATION
# ---------------------------------------------------------------------------
# Each node takes `state: PlayerDataClassState`.
# We read values with dot notation: `state.name`, `state.score`.
# We return standard dictionaries: `{"score": state.score + 5}`.

def warm_up(state: PlayerDataClassState) -> dict:
    """
    Node 1: Entry Worker.
    Simulates warm-up exercises and awards 5 fitness points.
    """
    print(f"\n[Node: warm_up] {state.name} is stretching and warming up...")
    return {
        "status": f"{state.name} finished warm-up.",
        "score": state.score + 5  # Earn 5 warm-up points
    }


def play_cricket(state: PlayerDataClassState) -> dict:
    """
    Node 2A: Cricket Worker.
    Simulates batting and awards 50 points for hitting a boundary.
    """
    print(f"[Node: play_cricket] {state.name} hits a boundary in cricket!")
    return {
        "game": "cricket",
        "score": state.score + 50,
        "status": f"{state.name} scored a boundary in cricket."
    }


def play_badminton(state: PlayerDataClassState) -> dict:
    """
    Node 2B: Badminton Worker.
    Simulates a rally and awards 30 points for landing a smash.
    """
    print(f"[Node: play_badminton] {state.name} lands a smash in badminton!")
    return {
        "game": "badminton",
        "score": state.score + 30,
        "status": f"{state.name} won a rally in badminton."
    }


# ---------------------------------------------------------------------------
# STEP 3: CONDITIONAL ROUTER
# ---------------------------------------------------------------------------
def decide_sport(state: PlayerDataClassState) -> Literal["play_cricket", "play_badminton"]:
    """
    Evaluates state.game using dot notation and returns the name of the next node.
    """
    if state.game == "cricket":
        print(f"[Router: decide_sport] Routing to play_cricket based on state.game = {state.game}")
        return "play_cricket"
    elif state.game == "badminton":
        print(f"[Router: decide_sport] Routing to play_badminton based on state.game = {state.game}")
        return "play_badminton"

    # Fallback random choice if sport is unassigned
    choice = "play_cricket" if random.random() < 0.5 else "play_badminton"
    print(f"[Router: decide_sport] No sport set. Randomly chosen: {choice}")
    return choice


# ---------------------------------------------------------------------------
# STEP 4: BUILD AND COMPILE THE STATEGRAPH
# ---------------------------------------------------------------------------
def build_dataclass_graph():
    """
    Assembles nodes and edges with PlayerDataClassState as the schema.
    """
    builder = StateGraph(PlayerDataClassState)

    builder.add_node("warm_up", warm_up)
    builder.add_node("play_cricket", play_cricket)
    builder.add_node("play_badminton", play_badminton)

    # Execution flow wiring
    builder.add_edge(START, "warm_up")
    builder.add_conditional_edges("warm_up", decide_sport)
    builder.add_edge("play_cricket", END)
    builder.add_edge("play_badminton", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# STEP 5: EXECUTION & DEMONSTRATION
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_dataclass_graph()
        print_mermaid(graph, "Dataclass State Graph Topology")

        # Test Case 1: Invoking with a Dataclass instance
        print("--- Test 1: Invoking with a Dataclass Instance ---")
        initial_state = PlayerDataClassState(name="Piyush", game="cricket")
        result1 = graph.invoke(initial_state)

        # Handling dictionary return representation
        name1 = result1.get("name") if isinstance(result1, dict) else result1.name
        game1 = result1.get("game") if isinstance(result1, dict) else result1.game
        score1 = result1.get("score") if isinstance(result1, dict) else result1.score
        status1 = result1.get("status") if isinstance(result1, dict) else result1.status
        print(f"Final State: {name1} played {game1}")
        print(f"Final Score: {score1} (5 warm-up + 50 cricket)")
        print(f"Status: {status1}\n")

        # Test Case 2: Invoking with a standard dictionary (LangGraph auto-maps it!)
        print("--- Test 2: Invoking with a Standard Dictionary (Auto-coerced) ---")
        result2 = graph.invoke({"name": "Piyush", "game": "badminton"})
        name2 = result2.get("name") if isinstance(result2, dict) else result2.name
        game2 = result2.get("game") if isinstance(result2, dict) else result2.game
        score2 = result2.get("score") if isinstance(result2, dict) else result2.score
        print(f"Final State: {name2} | Sport: {game2} | Score: {score2} (5 warm-up + 30 badminton)\n")

        # Test Case 3: Unspecified game (triggers randomized routing)
        print("--- Test 3: Unspecified Game (Randomized Routing) ---")
        result3 = graph.invoke({"name": "Sam"})
        name3 = result3.get("name") if isinstance(result3, dict) else result3.name
        game3 = result3.get("game") if isinstance(result3, dict) else result3.game
        score3 = result3.get("score") if isinstance(result3, dict) else result3.score
        print(f"Final State: {name3} | Auto-selected Sport: {game3} | Score: {score3}\n")

    safe_run("LangGraph State Schema with Python Dataclasses", demo)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 02: Dataclass State Schema
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates defining LangGraph state using Python's `@dataclass`.
# Shows how nodes read state using dot notation (`state.name`), return partial
# dictionaries for state updates, and how initial input dictionaries are automatically
# coerced by LangGraph into the dataclass contract.
#
# KEY CONCEPTS USED:
# • @dataclass: Standard library class decorator generating initialization logic.
# • Dot Notation: Clean object-oriented field access (`state.score`).
# • Default Values: Initial state values like `score: int = 0` are built right into the schema.
# • Interoperability: Nodes still return partial dictionaries; you don't have to reconstruct
#   the full Dataclass inside every node.
#
# HOW IT CONNECTS TO THE PROJECT:
# Dataclasses represent a middle ground between lightweight `TypedDict` and strictly
# validated `Pydantic` models, offering OOP ergonomics with zero third-party dependencies.
#
# THINGS TO REMEMBER:
# • Like TypedDict, Dataclasses do NOT validate types at runtime. Passing `"10"` to `score: int`
#   will not raise an error.
# • The object returned from `graph.invoke()` may be a dictionary or a Dataclass instance
#   depending on LangGraph version; use `.get()` or `isinstance()` checks when inspecting results.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting default values for optional fields (e.g. `score = 0`), which would require every
#   `invoke()` call to specify every single field.
# • Trying to mutate `state.score += 5` directly without returning `{"score": state.score + 5}`.
#   (Always return updates via dictionary so LangGraph can manage state immutability and snapshots!).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `@dataclass class PlayerDataClassState` -> notice default values -> check how
# `warm_up` accesses `state.score` and returns `{"score": state.score + 5}`.
# ============================================================
