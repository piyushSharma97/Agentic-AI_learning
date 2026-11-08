"""
=============================================================================
LANGGRAPH COMPONENT 7: AGENT MEMORY & CHECKPOINTING (MEMORYSAVER)
=============================================================================
TOPIC OVERVIEW:
Without a checkpointer, an agent is completely stateless between successive
`.invoke()` calls. If you ask:
Turn 1: "Add 12 and 13" -> Agent responds "25"
Turn 2: "Divide that by 5" -> Without memory, the agent asks: "Divide WHAT by 5?"

THE SOLUTION: CHECKPOINTERS & thread_id
LangGraph features a built-in persistence layer. When you compile a graph with
a checkpointer (such as `MemorySaver`), LangGraph automatically saves a state
snapshot after every node execution.

CRITICAL CONCEPT: thread_id
To support multiple concurrent users or conversations, checkpoints are indexed
by `thread_id` inside the `config` dictionary:
`config = {"configurable": {"thread_id": "alice_session_101"}}`

Sessions with DIFFERENT thread_ids are completely isolated from each other!

FLOW OF MULTI-TURN PERSISTENCE:
Turn 1: User ("Add 12 and 13") -> Agent evaluates -> 25 -> Saved to Thread 1
Turn 2: User ("Divide that by 5") -> Graph loads Thread 1 snapshot -> Agent knows
        "that" is 25! -> Evaluates 25/5=5 -> Saved to Thread 1
Thread 2: Bob asks ("What was my number?") -> Graph loads Thread 2 (Empty) ->
        Agent has zero access to Alice's data!
=============================================================================
"""

import sys
from pathlib import Path
from typing import Annotated, List
from typing_extensions import TypedDict

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# STEP 1: DEFINE TOOLS
# ===========================================================================
@tool
def add(a: float, b: float) -> float:
    """Add two numbers a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b safely."""
    if b == 0:
        return "Error: Division by zero."
    return str(round(a / b, 4))


TOOLS = [add, multiply, divide]


# ===========================================================================
# STEP 2: DEFINE AGENT STATE
# ===========================================================================
class PersistentAgentState(TypedDict):
    """State using add_messages so new turns append rather than overwrite."""
    messages: Annotated[List[AnyMessage], add_messages]


# ===========================================================================
# STEP 3: ASSEMBLE AGENT WITH MEMORYSAVER CHECKPOINTER
# ===========================================================================
def build_persistent_agent():
    """
    Wires up ReAct agent and compiles with MemorySaver checkpointer.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: PersistentAgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(PersistentAgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Initialize In-Memory Checkpointer
    checkpointer = MemorySaver()

    # Pass checkpointer to compile()!
    # This enables snapshotting after every node execution.
    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# STEP 4: MULTI-TURN PERSISTENT CONVERSATION DEMO
# ===========================================================================
def demo_multi_turn_memory():
    agent = build_persistent_agent()
    print_mermaid(agent, "Persistent ReAct Agent Graph (with MemorySaver)")

    # Thread 1: User Alice's session
    config_alice = {"configurable": {"thread_id": "alice_thread_1"}}

    print("--- [THREAD 1 - Turn 1 (Alice)] ---")
    query_1 = "Add 12 and 13."
    print(f"Alice: {query_1}")
    res1 = agent.invoke({"messages": [HumanMessage(content=query_1)]}, config=config_alice)
    last_msg_1 = res1["messages"][-1].content
    print(f"Agent: {last_msg_1}\n")

    print("--- [THREAD 1 - Turn 2 (Alice)] (Relies on past memory!) ---")
    query_2 = "Divide that by 5."
    print(f"Alice: {query_2}")
    # Notice: We only pass the new query! The agent retrieves past context from thread_id!
    res2 = agent.invoke({"messages": [HumanMessage(content=query_2)]}, config=config_alice)
    last_msg_2 = res2["messages"][-1].content
    print(f"Agent: {last_msg_2}\n")

    print("--- [THREAD 1 - Turn 3 (Alice)] (Follow-up calculation) ---")
    query_3 = "Multiply that result by 10."
    print(f"Alice: {query_3}")
    res3 = agent.invoke({"messages": [HumanMessage(content=query_3)]}, config=config_alice)
    last_msg_3 = res3["messages"][-1].content
    print(f"Agent: {last_msg_3}\n")

    # Thread 2: Completely Isolated Session for Bob
    config_bob = {"configurable": {"thread_id": "bob_thread_2"}}

    print("--- [THREAD 2 - Turn 1 (Bob)] (Isolated Thread Test) ---")
    query_bob = "What was the last number we calculated?"
    print(f"Bob: {query_bob}")
    res_bob = agent.invoke({"messages": [HumanMessage(content=query_bob)]}, config=config_bob)
    last_msg_bob = res_bob["messages"][-1].content
    print(f"Agent: {last_msg_bob}")
    print(">> Notice: Bob does NOT see Alice's calculations because thread_id isolates sessions!\n")

    # Inspecting State Snapshot directly from checkpointer
    print("--- Inspecting Saved State Snapshot in Checkpointer ---")
    snapshot = agent.get_state(config_alice)
    print(f"Thread 'alice_thread_1' checkpoint message count: {len(snapshot.values['messages'])}")
    print(f"Next expected node in thread: {snapshot.next or 'Completed (END)'}\n")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
def main():
    safe_run("LangGraph Checkpointing & Multi-Turn Memory (MemorySaver)", demo_multi_turn_memory)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 07: Checkpointing & Agent Memory
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates how LangGraph enables stateful, multi-turn memory across
# independent `.invoke()` calls using checkpointers (`MemorySaver`) and
# session partitioning with `thread_id`.
#
# KEY CONCEPTS USED:
# • Checkpointer: Persistence engine that writes state snapshots after every node execution.
# • MemorySaver: Built-in in-memory checkpointer suitable for testing and development.
# • thread_id: Key passed in `config={"configurable": {"thread_id": "..."}}` that identifies
#   a conversation thread.
# • Multi-turn Memory: The caller only sends the new turn; LangGraph automatically loads
#   past turns from the checkpoint.
# • get_state(config): Method to inspect the current state snapshot and next step of any thread.
#
# HOW IT CONNECTS TO THE PROJECT:
# Without checkpointers, chatbots cannot maintain conversations across HTTP requests
# or terminal prompts. This pattern powers all persistent conversational agents in LangGraph.
#
# THINGS TO REMEMBER:
# • Always supply `config={"configurable": {"thread_id": ...}}` when invoking a compiled graph
#   that has a checkpointer attached.
# • MemorySaver stores data in RAM; for durable storage across process restarts,
#   use SqliteSaver or PostgresSaver.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `checkpointer=...` during `builder.compile()`, resulting in stateless behavior.
# • Using the same `thread_id` for two different users, causing their memories to collide.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# 1. `agent = builder.compile(checkpointer=MemorySaver())`
# 2. `config = {"configurable": {"thread_id": "thread_1"}}`
# 3. `agent.invoke({"messages": [...]}, config=config)`
# ============================================================
