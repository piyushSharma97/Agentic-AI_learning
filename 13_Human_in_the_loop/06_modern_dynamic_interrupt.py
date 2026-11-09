"""
=============================================================================
LESSON 06: MODERN DYNAMIC INTERRUPTS & COMMAND (LangGraph 1.x / 0.2+)
=============================================================================

1. STATIC BREAKPOINTS vs DYNAMIC INTERRUPTS:
-----------------------------------------------------------------------------
Static Breakpoints (Lessons 01-05):
- Configured at compile time: `compile(interrupt_before=["tools"])`.
- Coarse-grained: ALWAYS pauses every time that node is reached.
- Resumes with: `graph.stream(None, thread_config)`.

Dynamic Interrupts (Modern LangGraph 1.x):
- Configured DYNAMICALLY inside node code using:
    `from langgraph.types import interrupt, Command`
- Fine-grained: Pause conditionally (e.g., only if amount > $1000, or only if
  operation is division/deletion).
- Passes custom data to the human: `human_reply = interrupt({"question": "Approve transfer?", "amount": 5000})`
- The node execution pauses immediately.
- Resumed with:
    `graph.stream(Command(resume=human_reply), thread_config)`
- The value passed to `resume` becomes the return value of `interrupt()`!

2. HOW IT WORKS UNDER THE HOOD:
-----------------------------------------------------------------------------
  Node executes ──> Reaches interrupt(payload)
                           │
                           ▼
                    Execution pauses!
                    Payload stored in snapshot.tasks[0].interrupts
                           │
                           ▼
                    Human inspects payload & makes decision
                           │
                           ▼
  Resume with: graph.stream(Command(resume={"approved": True}), thread)
                           │
                           ▼
  Node wakes up: `decision = interrupt(...)` receives the resume dictionary!
  Node finishes logic and updates state.
=============================================================================
"""

from typing import TypedDict, Optional, Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from config import print_banner, print_mermaid


# 1. State Definition
class TransactionState(TypedDict):
    sender: str
    recipient: str
    amount: float
    status: str
    human_notes: Optional[str]


# 2. Node Implementations
def initiate_transfer(state: TransactionState) -> Dict[str, Any]:
    """Node 1: Prepares the transaction."""
    print(f"\n[initiate_transfer] Processing transfer of ${state['amount']} from {state['sender']} to {state['recipient']}...")
    return {"status": "pending_approval"}


def human_approval_guard(state: TransactionState) -> Dict[str, Any]:
    """
    Node 2: Dynamic Interruption Node.
    
    If transfer amount exceeds $1,000, dynamically pauses execution and prompts human!
    If transfer is small (< $1,000), auto-approves without interrupting!
    """
    amount = state["amount"]
    
    if amount >= 1000.0:
        print(f"\n[human_approval_guard] High-value transfer (${amount:,.2f}) detected!")
        print("  --> Triggering dynamic interrupt() to pause and request human authorization...")

        # DYNAMIC INTERRUPT:
        # Pauses execution and surfaces this exact dictionary to the caller:
        human_decision = interrupt({
            "prompt": f"Authorize wire transfer of ${amount:,.2f} to {state['recipient']}?",
            "recipient": state["recipient"],
            "amount": amount,
            "security_level": "Tier 1",
        })

        # When resumed via Command(resume=...), human_decision holds the supplied value:
        print(f"\n[human_approval_guard] Resumed! Received Human Decision: {human_decision}")

        if human_decision.get("approved"):
            return {
                "status": "approved",
                "human_notes": human_decision.get("notes", "Approved by authorized officer."),
            }
        else:
            return {
                "status": "rejected",
                "human_notes": human_decision.get("notes", "Rejected by compliance officer."),
            }
    else:
        print(f"\n[human_approval_guard] Transfer under threshold (${amount}). Auto-approved!")
        return {"status": "approved", "human_notes": "Auto-approved (< $1,000 threshold)."}


def execute_settlement(state: TransactionState) -> Dict[str, Any]:
    """Node 3: Finalizes the transaction based on approval status."""
    print(f"\n[execute_settlement] Final Status: {state['status'].upper()}")
    if state["status"] == "approved":
        print(f"  --> Funds of ${state['amount']:,.2f} transferred to {state['recipient']} successfully!")
    else:
        print(f"  --> Transfer cancelled! Reason: {state['human_notes']}")
    return {}


# 3. Build Graph
def build_dynamic_interrupt_graph():
    builder = StateGraph(TransactionState)
    builder.add_node("initiate", initiate_transfer)
    builder.add_node("guard", human_approval_guard)
    builder.add_node("settle", execute_settlement)

    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "guard")
    builder.add_edge("guard", "settle")
    builder.add_edge("settle", END)

    # Note: NO interrupt_before or interrupt_after in compile()!
    # The interrupt is triggered dynamically inside the node function itself!
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def demo_dynamic_interrupt():
    print_banner(
        "LESSON 06",
        "Modern Dynamic Interrupts (interrupt & Command)",
        "Demonstrates conditional in-node interruption and resumption using Command(resume=...)."
    )

    graph = build_dynamic_interrupt_graph()
    print_mermaid(graph, "Dynamic Guard Flow")

    # TEST CASE 1: Transfer of $5,000 (Triggers Dynamic Interrupt)
    thread_1 = {"configurable": {"thread_id": "tx-1001"}}
    input_high = {
        "sender": "Alice Corp",
        "recipient": "Bob Enterprises",
        "amount": 5000.0,
        "status": "new",
        "human_notes": None,
    }

    print("=" * 65)
    print("RUN 1: Initiating High-Value Transfer ($5,000)...")
    print("=" * 65)

    # Stream until interrupt triggers
    for event in graph.stream(input_high, thread_1):
        print(f"Event: {event}")

    # Inspect the snapshot to view the interrupt payload
    state = graph.get_state(thread_1)
    print(f"\nGraph is paused! Next node: {state.next}")

    # Access the interrupt payload sent from inside the node:
    pending_task = state.tasks[0]
    interrupt_payload = pending_task.interrupts[0].value
    print("\n--- Interrupt Payload Extracted from StateSnapshot ---")
    for k, v in interrupt_payload.items():
        print(f"  {k}: {v}")
    print("------------------------------------------------------")

    # Simulate Human Officer Approving the Request
    print("\n[Human Officer]: Approving transaction with compliance note...")
    human_response = {
        "approved": True,
        "notes": "Verified invoices. Approved by Senior Risk Officer.",
    }

    print("\n--- Resuming with Command(resume=human_response) ---")
    # RESUMING A DYNAMIC INTERRUPT:
    for event in graph.stream(Command(resume=human_response), thread_1):
        print(f"Event: {event}")

    final_state_1 = graph.get_state(thread_1)
    print(f"\nRun 1 Completed State Status: {final_state_1.values['status']}")
    print(f"Compliance Notes: {final_state_1.values['human_notes']}")

    # TEST CASE 2: Small Transfer of $250 (Bypasses Interrupt Automatically)
    print("\n" + "=" * 65)
    print("RUN 2: Initiating Low-Value Transfer ($250 - Auto-Approval)...")
    print("=" * 65)

    thread_2 = {"configurable": {"thread_id": "tx-1002"}}
    input_low = {
        "sender": "Charlie",
        "recipient": "Local Cafe",
        "amount": 250.0,
        "status": "new",
        "human_notes": None,
    }

    for event in graph.stream(input_low, thread_2):
        print(f"Event: {event}")

    final_state_2 = graph.get_state(thread_2)
    print(f"\nRun 2 Status: {final_state_2.values['status']} (Zero human intervention required!)")


if __name__ == "__main__":
    demo_dynamic_interrupt()
