"""
=============================================================================
TOPIC 112: ROUTING WORKFLOW IN LANGGRAPH
=============================================================================

1. WHAT IS ROUTING?
-------------------
Routing is a workflow pattern that classifies an incoming user query and
directs it to a specialized downstream node or sub-graph.

                 ┌─────────────────────────────┐
                 │  Classification / Router    │
                 │  (Analyzes user intent)     │
                 └──────────────┬──────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │ (technical)         │ (billing)           │ (general)
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Technical Node   │  │ Billing Node     │  │ General FAQ Node │
│ (Code & Debug)   │  │ (Refunds/Invoice)│  │ (Company info)   │
└─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼
                             [ END ]

2. WHY USE ROUTING?
-------------------
- "Specialized Prompts": A single general prompt cannot excel at Python code
  debugging, empathetic refund disputes, and concise marketing FAQs all at once.
- "Cost & Latency Optimization": Route simple questions to small/fast models
  (e.g., gpt-4o-mini or llama-3.1-8b) and reserve high-reasoning models only
  for complex technical engineering requests.
- "Security & Tool Isolation": You only attach sensitive tools (e.g., Stripe API,
  Database Write) to the Billing node, preventing the General Chat node from
  ever accessing them.

3. LANGGRAPH IMPLEMENTATION
---------------------------
- Classifier node categorizes input into a structured label.
- `add_conditional_edges(source, router_function, path_map)` directs the flow
  based on the classifier's verdict.
=============================================================================
"""

from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm, print_mermaid, print_banner, safe_run


# =============================================================================
# 1. STATE DEFINITION
# =============================================================================
class RoutingState(TypedDict):
    """
    State tracks the user query, classification category, and final response.
    """
    user_query: str
    category: Literal["technical", "billing", "general_inquiry"]
    response: str
    routed_expert: str


# =============================================================================
# 2. CLASSIFIER / ROUTER NODE
# =============================================================================
llm = get_llm(temperature=0.0)


def classify_query_node(state: RoutingState) -> dict:
    """
    Analyzes user intent and categorizes it strictly into:
    - 'technical': Bugs, code, errors, API integration, architecture
    - 'billing': Payments, invoices, subscription charges, refund requests
    - 'general_inquiry': Office hours, company info, sales, careers, greetings
    """
    print("[Router] Classifying query intent...")

    prompt = f"""You are a query classification dispatcher.
Classify the following customer query into EXACTLY one of these categories:
- technical
- billing
- general_inquiry

Query: "{state['user_query']}"

Respond with ONLY the category name in lowercase (no punctuation, no explanation)."""

    response = llm.invoke([
        SystemMessage(content="You are a strict, single-word intent classifier."),
        HumanMessage(content=prompt)
    ])

    raw_category = response.content.strip().lower()

    # Fallback sanity check in case model outputs extra characters
    if "tech" in raw_category:
        category = "technical"
    elif "bill" in raw_category or "refund" in raw_category or "invoice" in raw_category:
        category = "billing"
    else:
        category = "general_inquiry"

    print(f"[Router] Detected Category: '{category}'")
    return {"category": category}


# =============================================================================
# 3. SPECIALIZED EXPERT NODES
# =============================================================================
def technical_support_node(state: RoutingState) -> dict:
    """Specialist 1: Handles technical bugs, architecture, and code snippets."""
    print("[Specialist: Technical Support] Handling code/architecture query...")

    prompt = f"""You are a Principal Software Support Engineer.
The user reported this technical problem:
"{state['user_query']}"

Provide a structured, step-by-step diagnostic solution with code advice where relevant.
Be concise and engineering-focused."""

    response = llm.invoke([
        SystemMessage(content="You are an expert technical support engineer."),
        HumanMessage(content=prompt)
    ])

    return {
        "response": response.content.strip(),
        "routed_expert": "Senior Technical Support Engineer"
    }


def billing_support_node(state: RoutingState) -> dict:
    """Specialist 2: Handles invoices, refunds, and subscriptions with empathy & clarity."""
    print("[Specialist: Billing Support] Handling finance/subscription query...")

    prompt = f"""You are a Customer Billing & Finance Representative.
The user has this billing inquiry:
"{state['user_query']}"

Explain our policy clearly:
- 30-day money-back guarantee for annual plans
- Invoices available under Account -> Invoices
- Refunds processed within 3-5 business days
Respond with an empathetic, reassuring, professional tone."""

    response = llm.invoke([
        SystemMessage(content="You are an empathetic, precise billing specialist."),
        HumanMessage(content=prompt)
    ])

    return {
        "response": response.content.strip(),
        "routed_expert": "Customer Billing & Accounts Specialist"
    }


def general_inquiry_node(state: RoutingState) -> dict:
    """Specialist 3: Handles general FAQs, company info, and sales inquiries."""
    print("[Specialist: General FAQ] Handling general company query...")

    prompt = f"""You are a Helpful Customer Experience Ambassador.
The user asked:
"{state['user_query']}"

Provide a warm, welcoming, and concise answer about our company and services."""

    response = llm.invoke([
        SystemMessage(content="You are a warm, courteous customer experience ambassador."),
        HumanMessage(content=prompt)
    ])

    return {
        "response": response.content.strip(),
        "routed_expert": "Customer Experience Ambassador"
    }


# =============================================================================
# 4. CONDITIONAL ROUTING FUNCTION
# =============================================================================
def route_to_specialist(state: RoutingState) -> str:
    """
    Inspects state['category'] and returns the exact node name to execute next.
    """
    category = state.get("category", "general_inquiry")

    if category == "technical":
        return "technical_support"
    elif category == "billing":
        return "billing_support"
    else:
        return "general_inquiry"


# =============================================================================
# 5. GRAPH ASSEMBLY & TOPOLOGY
# =============================================================================
builder = StateGraph(RoutingState)

# Add all nodes
builder.add_node("classifier", classify_query_node)
builder.add_node("technical_support", technical_support_node)
builder.add_node("billing_support", billing_support_node)
builder.add_node("general_inquiry", general_inquiry_node)

# Flow:
# START -> classifier
builder.add_edge(START, "classifier")

# Conditional Routing from classifier to the appropriate specialist
builder.add_conditional_edges(
    "classifier",
    route_to_specialist,
    {
        "technical_support": "technical_support",
        "billing_support": "billing_support",
        "general_inquiry": "general_inquiry",
    }
)

# All specialists lead to END
builder.add_edge("technical_support", END)
builder.add_edge("billing_support", END)
builder.add_edge("general_inquiry", END)

routing_graph = builder.compile()


# =============================================================================
# 6. DEMO EXECUTION ACROSS DIVERSE INPUTS
# =============================================================================
def run_routing_demo():
    print_banner(
        "TOPIC 112",
        "Routing Workflow Pattern",
        "Demonstrates intelligent intent classification and routing to 3 domain specialists:\n"
        "1. Technical Support  |  2. Billing & Refunds  |  3. General FAQ"
    )

    print_mermaid(routing_graph, "Routing Graph Topology")

    # Test cases representing different intents
    test_queries = [
        "I was charged $49 twice on my credit card this morning for my monthly plan. Can I get a refund?",
        "My LangGraph checkpointer is throwing SQLite lock error when running async with max_concurrency=5. How do I fix it?",
        "What are your customer support office hours and where are your headquarters located?"
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: \"{query}\"")
        print(f"{'='*70}")

        result = routing_graph.invoke({
            "user_query": query,
            "category": "general_inquiry",
            "response": "",
            "routed_expert": ""
        })

        print(f"\n[Classification Outcome]: {result['category'].upper()}")
        print(f"[Handled By]: {result['routed_expert']}")
        print(f"[Response]:\n{result['response']}\n")


if __name__ == "__main__":
    safe_run("Routing Workflow", run_routing_demo)
