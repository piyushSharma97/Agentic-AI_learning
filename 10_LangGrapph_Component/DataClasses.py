"""
=============================================================================
Dataclasses in LangGraph StateGraph
=============================================================================
Python's standard @dataclass provides an object-oriented way to define
state schemas in LangGraph.

Key Advantages:
1. Dot notation: `state.name` instead of `state['name']`
2. Default values: `score: int = 0` directly on the class
3. Direct instantiation: `DataClassState(name="Krish", game="cricket")`
=============================================================================
"""

import random
from dataclasses import dataclass
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


@dataclass
class DataClassState:
    name: str
    game: Optional[Literal["badminton", "cricket"]] = None


def play_game(state: DataClassState):
    print("--- Play Game node has been called ---")
    return {"name": state.name + " wants to play"}


def cricket(state: DataClassState):
    print("--- Cricket node has been called ---")
    return {"name": state.name + " cricket", "game": "cricket"}


def badminton(state: DataClassState):
    print("--- Badminton node has been called ---")
    return {"name": state.name + " badminton", "game": "badminton"}


def decide_play(state: DataClassState) -> Literal["cricket", "badminton"]:
    # 50/50 split or routes by preference
    if state.game in ("cricket", "badminton"):
        return state.game
    return "cricket" if random.random() < 0.5 else "badminton"


def build_graph():
    builder = StateGraph(DataClassState)
    builder.add_node("playgame", play_game)
    builder.add_node("cricket", cricket)
    builder.add_node("badminton", badminton)

    # Graph Flow
    builder.add_edge(START, "playgame")
    builder.add_conditional_edges("playgame", decide_play)
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)

    return builder.compile()


def main():
    def demo():
        graph = build_graph()
        print_mermaid(graph, "Dataclass StateGraph")

        print("--- Invocation 1: Explicit Game ---")
        res1 = graph.invoke(DataClassState(name="Krish", game="cricket"))
        n1 = res1.get("name") if isinstance(res1, dict) else res1.name
        g1 = res1.get("game") if isinstance(res1, dict) else res1.game
        print(f"Result 1: name='{n1}', game='{g1}'\n")

        print("--- Invocation 2: Random Decision ---")
        res2 = graph.invoke(DataClassState(name="Krish"))
        n2 = res2.get("name") if isinstance(res2, dict) else res2.name
        g2 = res2.get("game") if isinstance(res2, dict) else res2.game
        print(f"Result 2: name='{n2}', game='{g2}'\n")

    safe_run("Dataclasses State Schema", demo)


if __name__ == "__main__":
    main()
