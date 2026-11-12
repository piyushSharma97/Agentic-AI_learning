"""
=============================================================================
LESSON 112: ROUTING
=============================================================================

The idea
--------
One input, several specialists, and only one of them should run.

    START -> classifier -> technical support
                        -> billing support
                        -> general FAQ
                        -> END

The classifier writes a label into state. A routing function reads that
label and returns the next node name. add_conditional_edges connects the
label to the node.

Routing is different from parallelization. Parallelization runs every
branch. Routing runs exactly one branch.

Why bother
----------
A billing prompt should talk about refunds. A technical prompt should talk
about stack traces. Mixing both into one prompt makes both worse, and it
would also put billing tools in reach of a general chat node.

Run:
    python "12_Different Workflows In LangGraph/112_routing.py"
=============================================================================
"""

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run


Category = Literal["technical", "billing", "general"]


class RouteState(TypedDict):
    query: str
    category: Category
    expert: str
    answer: str


llm = get_llm(temperature=0.0)


def classify(state: RouteState) -> dict:
    print("\n[Router] classifying")
    response = llm.invoke([
        SystemMessage(content="You label support tickets. Reply with one word."),
        HumanMessage(content=(
            "Label the ticket as exactly one of: technical, billing, general.\n\n"
            f"Ticket: {state['query']}"
        )),
    ])
    raw = response.content.strip().lower()
    if "tech" in raw or "code" in raw or "bug" in raw:
        category: Category = "technical"
    elif "bill" in raw or "refund" in raw or "invoice" in raw or "charge" in raw:
        category = "billing"
    else:
        category = "general"
    print(f"[Router] category = {category}")
    return {"category": category}


def technical(state: RouteState) -> dict:
    print("[Expert] technical")
    response = llm.invoke([
        SystemMessage(content="You are a support engineer. Give three short numbered steps."),
        HumanMessage(content=state["query"]),
    ])
    return {"expert": "technical support", "answer": response.content.strip()}


def billing(state: RouteState) -> dict:
    print("[Expert] billing")
    response = llm.invoke([
        SystemMessage(content=(
            "You handle billing. Policy: 30-day refund on annual plans, "
            "3-5 business days to process. Be brief and specific."
        )),
        HumanMessage(content=state["query"]),
    ])
    return {"expert": "billing support", "answer": response.content.strip()}


def general(state: RouteState) -> dict:
    print("[Expert] general")
    response = llm.invoke([
        SystemMessage(content=(
            "You answer company questions. Hours: 9:00-18:00 IST on weekdays. "
            "HQ: Bengaluru. Keep the answer under 4 lines."
        )),
        HumanMessage(content=state["query"]),
    ])
    return {"expert": "general FAQ", "answer": response.content.strip()}


def route(state: RouteState) -> str:
    """Map the label to a node name. Only this node will run."""
    return {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    }[state["category"]]


builder = StateGraph(RouteState)
builder.add_node("classify", classify)
builder.add_node("technical", technical)
builder.add_node("billing", billing)
builder.add_node("general", general)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general",
    },
)
builder.add_edge("technical", END)
builder.add_edge("billing", END)
builder.add_edge("general", END)

graph = builder.compile()


QUERIES = [
    "My card was charged twice for the annual plan. How do I get a refund?",
    "LangGraph says the checkpointer database is locked when two runs share one thread id.",
    "What time does support open, and which city is the office in?",
]


def run_routing() -> None:
    print_banner(
        "LESSON 112",
        "Routing",
        "The classifier picks one specialist.\n"
        "The other specialist nodes do not run for that input.",
    )
    print_mermaid(graph, "Routing graph")

    for query in QUERIES:
        print("\n" + "=" * 70)
        print("QUERY:", query)
        result = graph.invoke({
            "query": query,
            "category": "general",
            "expert": "",
            "answer": "",
        })
        print(f"EXPERT: {result['expert']}")
        print(result["answer"])


if __name__ == "__main__":
    safe_run("Lesson 112 Routing", run_routing)
