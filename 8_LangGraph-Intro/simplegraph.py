# ============================================================
# LANGGRAPH INTRO: SIMPLE STATE GRAPH WITH CONDITIONAL ROUTING
# ============================================================
#
# What is LangGraph?
# ------------------
# LangGraph is a library for building stateful, multi-actor applications
# with LLMs. While traditional chains in LangChain are strictly linear
# Directed Acyclic Graphs (DAGs), LangGraph models workflows as graphs
# (Nodes + Edges + State), allowing:
#   - Cycles / Loops (e.g., agent self-correction or multi-turn tool loops)
#   - Persistence & Memory (saving state across checkpoints)
#   - Human-in-the-loop approvals (pausing and resuming execution)
#
# Core Concepts in this example:
# 1. State: The shared data schema (TypedDict) passed between all nodes.
# 2. Nodes: Regular Python functions that receive the current state and return updates.
# 3. Edges: Direct transitions from one node to the next (including START and END).
# 4. Conditional Edges: Decision functions that route to different nodes based on logic/state.
# 5. Compile & Invoke: Validating the graph structure into a runnable, then executing it.
# ============================================================

from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. DEFINE STATE
# ============================================================
# The State is the single source of truth for the entire graph.
# Every node receives this state as its argument, and any dictionary
# returned by a node updates this state.
#
# Note: Without a reducer (like Annotated[list, add_messages]), returning
# a key will OVERWRITE its existing value in the state dictionary.
class State(TypedDict):
    graph_info: str  # Accumulates text as the state travels through nodes


# ============================================================
# 2. DEFINE NODES
# ============================================================
# Nodes are standard Python functions with the signature:
#     node_name(state: State) -> dict
#
# Rules:
# - Input: The current snapshot of the graph State.
# - Output: A dictionary containing ONLY the keys you want to update.
#   (You do not need to return the entire state).

def start_play(state: State):
    """Entry node: appends planning message to the existing state."""
    print("Start_Play node has been called")

    return {
        "graph_info": state["graph_info"] + " I am planning to play"
    }


def cricket(state: State):
    """Branch node A: appends 'Cricket' to the state."""
    print("My Cricket node has been called")

    return {
        "graph_info": state["graph_info"] + " Cricket"
    }


def badminton(state: State):
    """Branch node B: appends 'Badminton' to the state."""
    print("My badminton node has been called")

    return {
        "graph_info": state["graph_info"] + " Badminton"
    }


# ============================================================
# 3. DEFINE CONDITIONAL ROUTING FUNCTION
# ============================================================
# A conditional routing function inspects the state or external conditions
# and returns the name of the next node to transition to.
#
# The return type annotation Literal["cricket", "badminton"] tells LangGraph
# exactly which node names are valid destinations for this decision.

def random_play(state: State) -> Literal["cricket", "badminton"]:
    """
    Decides whether the next step is cricket or badminton.
    In real applications, this function could check LLM intent classification,
    sentiment scores, tool call requirements, or user preferences.
    """
    import random

    # 50/50 chance to branch to cricket or badminton
    if random.random() > 0.5:
        return "cricket"

    return "badminton"


# ============================================================
# 4. INITIALIZE GRAPH
# ============================================================
# StateGraph takes the State schema class as an argument.
# This binds the graph structure to this specific data contract.
graph = StateGraph(State)


# ============================================================
# 5. REGISTER NODES
# ============================================================
# add_node(node_name, function)
# - node_name: The unique string identifier used when drawing edges.
# - function: The Python callable executed when this node is reached.
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)


# ============================================================
# 6. DEFINE EDGES (WIRING THE FLOW)
# ============================================================
# START -> "start_play":
#   START is a special LangGraph constant representing graph entry.
#   Execution always begins at "start_play".
graph.add_edge(START, "start_play")

# Conditional Edge:
#   After "start_play" completes, LangGraph calls random_play(state).
#   The return value ("cricket" or "badminton") determines which node runs next.
graph.add_conditional_edges(
    "start_play",
    random_play
)

# Terminal Edges:
#   Both "cricket" and "badminton" lead to END, terminating execution.
#   END is a special LangGraph constant representing graph exit.
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)


# ============================================================
# 7. COMPILE GRAPH
# ============================================================
# .compile() validates the graph structure:
#   - Ensures all referenced nodes are registered.
#   - Ensures all execution branches lead toward END.
#   - Returns a CompiledGraph (Pregel runnable) supporting .invoke(), .stream(), etc.
graph_builder = graph.compile()


# ============================================================
# 8. EXECUTE GRAPH
# ============================================================
# .invoke() seeds the initial state and runs the graph to completion.
# It returns the final state dictionary after reaching END.
result = graph_builder.invoke({
    "graph_info": "Hey My name is Piyush"
})

print("\nFinal Result:", result)
