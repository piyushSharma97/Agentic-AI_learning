"""
=============================================================================
LANGGRAPH COMPONENT 7: AGENT MEMORY & CHECKPOINTING (MEMORYSAVER)
=============================================================================
THE PROBLEM:
Without a checkpointer, an agent is stateless between successive `.invoke()` calls.
If you ask:
Turn 1: "Add 12 and 13" -> Agent responds "25"
Turn 2: "Divide that by 5" -> Without memory, the agent asks: "Divide WHAT by 5?"

THE SOLUTION: CHECKPOINTERS & MEMORYSAVER
LangGraph features a built-in persistence layer. When you compile a graph with
a checkpointer, LangGraph saves a state checkpoint after every node execution.

CRITICAL CONCEPT: thread_id
To support multiple concurrent users or conversations, checkpoints are indexed
by `thread_id` inside the `config` dictionary:
`config = {"configurable": {"thread_id": "user_session_101"}}`

Sessions with DIFFERENT thread_ids are completely isolated from each other!
=============================================================================
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import shared helper
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# 1. DEFINE TOOLS
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
# 2. DEFINE AGENT STATE
# ===========================================================================
class PersistentAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# ===========================================================================
# 3. BUILD AGENT WITH CHECKPOINTER
# ===========================================================================
def build_persistent_agent():
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

    # Compile WITH the checkpointer to activate persistent memory!
    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# 4. MULTI-TURN PERSISTENT CONVERSATION DEMO
# ===========================================================================
def demo_multi_turn_memory():
    agent = build_persistent_agent()
    print_mermaid(agent, "Persistent ReAct Agent Graph")

    # Thread 1: User Alice
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

    print("--- [THREAD 1 - Turn 3 (Alice)] (Multiply that result by 10) ---")
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

    # Inspecting State Snapshot
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

