"""
=============================================================================
LANGGRAPH COMPONENT 2: STATE SCHEMA WITH PYTHON DATACLASSES
=============================================================================
WHY DATACLASSES?
Python's built-in `@dataclass` decorator offers an object-oriented approach
to defining LangGraph state:
1. Attribute Access (Dot Notation): Access values with `state.name` instead of
   dictionary keys `state['name']`, preventing typos and improving IDE autocompletion.
2. Default Values: Define default initial state values easily (`score: int = 0`).
3. Class Methods & Properties: You can add helper methods or computed properties
   directly inside the dataclass.

HOW LANGGRAPH HANDLES UPDATES:
Even though the state is a Dataclass instance, your nodes can still return
standard dictionaries like `{"score": state.score + 10}`. LangGraph automatically
applies those dictionary updates to the Dataclass fields!
=============================================================================
"""

import random
from dataclasses import dataclass, field
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# 1. DEFINE THE STATE SCHEMA AS A DATACLASS
# ---------------------------------------------------------------------------
@dataclass
class PlayerDataClassState:
    """
    Dataclass State Schema.
    Notice the clean syntax, default values, and attribute access.
    """
    name: str
    game: Optional[Literal["cricket", "badminton"]] = None
    score: int = 0
    status: str = "Initialized"

    def summary(self) -> str:
        """Helper method on the state object."""
        return f"Player: {self.name} | Sport: {self.game} | Score: {self.score}"


# ---------------------------------------------------------------------------
# 2. DEFINE NODE FUNCTIONS USING DOT NOTATION
# ---------------------------------------------------------------------------
def warm_up(state: PlayerDataClassState) -> dict:
    # Notice: dot notation `state.name` instead of `state['name']`
    print(f"\n[Node: warm_up] {state.name} is stretching and warming up...")
    return {
        "status": f"{state.name} finished warm-up.",
        "score": state.score + 5  # Earn 5 fitness points
    }


def play_cricket(state: PlayerDataClassState) -> dict:
    print(f"[Node: play_cricket] {state.name} hits a boundary in cricket!")
    return {
        "game": "cricket",
        "score": state.score + 50,
        "status": f"{state.name} scored a boundary in cricket."
    }


def play_badminton(state: PlayerDataClassState) -> dict:
    print(f"[Node: play_badminton] {state.name} lands a smash in badminton!")
    return {
        "game": "badminton",
        "score": state.score + 30,
        "status": f"{state.name} won a rally in badminton."
    }


# ---------------------------------------------------------------------------
# 3. CONDITIONAL ROUTER
# ---------------------------------------------------------------------------
def decide_sport(state: PlayerDataClassState) -> Literal["play_cricket", "play_badminton"]:
    if state.game == "cricket":
        return "play_cricket"
    elif state.game == "badminton":
        return "play_badminton"
    
    # 50/50 split if unassigned
    return "play_cricket" if random.random() < 0.5 else "play_badminton"


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_dataclass_graph():
    builder = StateGraph(PlayerDataClassState)

    builder.add_node("warm_up", warm_up)
    builder.add_node("play_cricket", play_cricket)
    builder.add_node("play_badminton", play_badminton)

    builder.add_edge(START, "warm_up")
    builder.add_conditional_edges("warm_up", decide_sport)
    builder.add_edge("play_cricket", END)
    builder.add_edge("play_badminton", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 5. EXECUTION & DEMONSTRATION
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_dataclass_graph()
        print_mermaid(graph, "Dataclass State Graph Topology")

        print("--- Test 1: Invoking with a Dataclass Instance ---")
        # You can instantiate the Dataclass directly
        initial_state = PlayerDataClassState(name="Krish", game="cricket")
        result1 = graph.invoke(initial_state)

        # Result returned from invoke is a dictionary representation of the state
        name1 = result1.get("name") if isinstance(result1, dict) else result1.name
        game1 = result1.get("game") if isinstance(result1, dict) else result1.game
        score1 = result1.get("score") if isinstance(result1, dict) else result1.score
        status1 = result1.get("status") if isinstance(result1, dict) else result1.status
        print(f"Final State: {name1} played {game1}")
        print(f"Final Score: {score1}")
        print(f"Status: {status1}\n")

        print("--- Test 2: Invoking with a Standard Dictionary (Auto-coerced) ---")
        # LangGraph automatically maps dictionary inputs to the Dataclass!
        result2 = graph.invoke({"name": "Piyush", "game": "badminton"})
        name2 = result2.get("name") if isinstance(result2, dict) else result2.name
        game2 = result2.get("game") if isinstance(result2, dict) else result2.game
        score2 = result2.get("score") if isinstance(result2, dict) else result2.score
        print(f"Final State: {name2} | Sport: {game2} | Score: {score2}\n")

        print("--- Test 3: Unspecified Game (Randomized Routing) ---")
        result3 = graph.invoke({"name": "Sam"})
        name3 = result3.get("name") if isinstance(result3, dict) else result3.name
        game3 = result3.get("game") if isinstance(result3, dict) else result3.game
        score3 = result3.get("score") if isinstance(result3, dict) else result3.score
        print(f"Final State: {name3} | Auto-selected Sport: {game3} | Score: {score3}\n")

    safe_run("LangGraph State Schema with Python Dataclasses", demo)


if __name__ == "__main__":
    main()
