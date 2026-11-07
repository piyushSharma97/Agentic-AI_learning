"""
=============================================================================
State Schema in LangGraph: TypedDict vs DataClasses
=============================================================================
When we define a LangGraph StateGraph, we specify a state schema.
The state schema defines the structure and types of data our graph will hold.

1. TypedDict:
   - Uses standard dictionary syntax: `state['name']`
   - Only provides type hints for linters / IDEs (not enforced at runtime!)
   - Lightweight and standard across LangGraph tutorials.

2. Dataclasses:
   - Uses attribute access: `state.name`
   - Supports default values and class methods.

See `01_state_schema_typeddict.py` and `02_state_schema_dataclass.py` for
dedicated deep-dives into each!
=============================================================================
"""

import random
from typing import Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


class TypedDictState(TypedDict):
    name: str
    game: Optional[Literal["cricket", "badminton"]]


def play_game(state: TypedDictState):
    print("--- Play Game node has been called ---")
    return {"name": f"{state['name']} wants to play"}


def cricket(state: TypedDictState):
    print("--- Cricket node has been called ---")
    return {"name": f"{state['name']} cricket", "game": "cricket"}


def badminton(state: TypedDictState):
    print("--- Badminton node has been called ---")
    return {"name": f"{state['name']} badminton", "game": "badminton"}


def decide_play(state: TypedDictState) -> Literal["cricket", "badminton"]:
    if state.get("game") in ("cricket", "badminton"):
        return state["game"]
    return "cricket" if random.random() < 0.5 else "badminton"


def build_graph():
    builder = StateGraph(TypedDictState)
    builder.add_node("playgame", play_game)
    builder.add_node("cricket", cricket)
    builder.add_node("badminton", badminton)

    builder.add_edge(START, "playgame")
    builder.add_conditional_edges("playgame", decide_play)
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)

    return builder.compile()


def main():
    def demo():
        graph = build_graph()
        print_mermaid(graph, "TypedDict StateGraph")

        print("--- Invocation with Dict: ---")
        res1 = graph.invoke({"name": "Krish", "game": "cricket"})
        print(f"Result: {res1}\n")

        print("--- Invocation with integer name (Type hint not enforced at runtime): ---")
        res2 = graph.invoke({"name": 123, "game": "badminton"})
        print(f"Result with int: {res2}\n")

    safe_run("State Schema with TypedDict", demo)


if __name__ == "__main__":
    main()
