"""
=============================================================================
USE CASE 5: AGENTS IN LANGCHAIN
=============================================================================
While a Chain executes a predetermined sequence of steps, an Agent uses the
LLM as a reasoning engine to dynamically decide:
1. Which tools to use (if any)
2. What arguments to pass to those tools
3. In what sequence to execute them
4. When enough information is gathered to return a final response

Key Concepts Covered:
1. Difference between Chains (fixed DAG) and Agents (dynamic ReAct loop)
2. Creating an Agent with `create_agent` (Modern LangChain standard)
3. Multi-tool, multi-step problem solving
4. Inspecting the step-by-step reasoning trace
=============================================================================
"""

import json
from langchain.agents import create_agent
from langchain_core.tools import tool

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Define Agent Tools
# ---------------------------------------------------------------------------
ORDERS_DB = {
    "ORD-101": {"customer": "Sarah Connor", "item": "Smart Watch Pro", "price": 299.00, "status": "delivered", "days_ago": 12},
    "ORD-102": {"customer": "John Doe", "item": "Wireless Earbuds", "price": 89.00, "status": "shipped", "days_ago": 3},
    "ORD-103": {"customer": "Bruce Wayne", "item": "Noise-Cancelling Headphones", "price": 350.00, "status": "delivered", "days_ago": 45},
}


@tool
def lookup_order_details(order_id: str) -> str:
    """
    Search customer order database by order ID (e.g. 'ORD-101', 'ORD-102', 'ORD-103').
    Returns order details including item, price, status, and days since purchase.
    """
    clean_id = order_id.strip().upper()
    order = ORDERS_DB.get(clean_id)
    if order:
        return json.dumps({"order_id": clean_id, **order})
    return json.dumps({"error": f"Order {clean_id} not found."})


@tool
def check_refund_eligibility(days_since_purchase: int, item_price: float) -> str:
    """
    Check if an item is eligible for refund based on company policy (30-day window)
    and calculate restocking fee if applicable.
    """
    if days_since_purchase <= 30:
        restocking_fee = 0.05 * item_price if item_price > 200 else 0.0
        net_refund = item_price - restocking_fee
        return json.dumps({
            "eligible": True,
            "original_price": item_price,
            "restocking_fee": round(restocking_fee, 2),
            "net_refund": round(net_refund, 2),
            "message": "Approved for full refund minus standard restocking fee."
        })
    else:
        return json.dumps({
            "eligible": False,
            "message": f"Ineligible for refund: return window is 30 days, but {days_since_purchase} days have passed."
        })


@tool
def calculate_math(expression: str) -> str:
    """
    Safely evaluate arithmetic mathematical expressions (e.g., '299 - 14.95').
    """
    try:
        # Limited safe arithmetic evaluation
        allowed_chars = set("0123456789+-*/. ()")
        if not set(expression).issubset(allowed_chars):
            return "Error: Invalid characters in arithmetic expression."
        result = eval(expression, {"__builtins__": None}, {})
        return str(round(result, 2))
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# 2. Build and Run the Autonomous Agent
# ---------------------------------------------------------------------------
def demo_autonomous_agent():
    print("--- Autonomous Multi-Step Reasoning Agent ---")
    llm = get_llm()
    
    tools = [lookup_order_details, check_refund_eligibility, calculate_math]
    
    system_prompt = (
        "You are an AI Customer Support Specialist for an electronics store. "
        "When handling customer inquiries, always look up the order details first, "
        "check policy compliance, and provide a polite, professional resolution."
    )
    
    # Create the agent using LangChain's compiled state graph agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    query = "Customer Sarah Connor wants to return order ORD-101. Is she eligible, and what will her net refund be?"
    print(f"User Request: {query}\n")
    
    # Execute agent
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    
    # Display the reasoning trace
    print("--- Step-by-Step Execution Trace ---")
    for idx, msg in enumerate(response["messages"], start=1):
        role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
        
        # Check if message contains tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print(f"Step {idx} [{role.upper()} - TOOL CALL]: Decided to call:")
            for call in tool_calls:
                print(f"   -> Tool: {call['name']} | Args: {call['args']}")
        elif role == "tool":
            print(f"Step {idx} [TOOL OBSERVATION]: Result returned:\n   {msg.content}")
        else:
            print(f"Step {idx} [{role.upper()}]:\n   {msg.content}")
    print()


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("LangChain Agent Demonstrations", lambda: [
        demo_autonomous_agent(),
    ])


if __name__ == "__main__":
    main()

