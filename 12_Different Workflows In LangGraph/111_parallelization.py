r"""
=============================================================================
LESSON 111: PARALLELIZATION
=============================================================================

The idea
--------
Some work does not depend on other work. Run those pieces at the same time,
then join the results.

Two kinds you will see in this course:

1. Sectioning. Split one job into independent views.
   Market, tech, and security each review the same proposal.
2. Voting. Ask the same question more than once (different prompts or
   models) and keep the answer they agree on.

This file implements sectioning. Voting uses the same fan-out / fan-in shape.
The branches would share a question and a reducer would collect their votes.

        START
       /  |  \
   market tech security      <- fan-out, these three run together
       \  |  /
      synthesizer             <- fan-in, this waits for all three
          |
         END

How LangGraph does it
---------------------
Two or more edges that leave the same node start a parallel super-step.
The node they all point at does not run until every branch has finished.

Parallel nodes must not overwrite the same state key. Put their results in
a list annotated with operator.add. LangGraph concatenates the lists.

    analyses: Annotated[list[str], operator.add]

Each branch returns {"analyses": ["one review"]}. After the join, state
holds all three reviews.

Run:
    python "12_Different Workflows In LangGraph/111_parallelization.py"
=============================================================================
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run


class ReviewState(TypedDict):
    proposal: str
    analyses: Annotated[list[str], operator.add]
    verdict: str


llm = get_llm(temperature=0.2)

PROPOSAL = (
    "A support agent that reads a customer's past tickets and drafts a reply. "
    "A human must approve the reply before it is sent."
)


def _review(role: str, question: str, proposal: str) -> str:
    response = llm.invoke([
        SystemMessage(content=f"You are a {role}. Answer in two sentences."),
        HumanMessage(content=f"Proposal: {proposal}\n\n{question}"),
    ])
    return response.content.strip()


def market_analyst(state: ReviewState) -> dict:
    print("[Branch] market")
    text = _review(
        "market analyst",
        "Who would buy this, and what is the main commercial risk?",
        state["proposal"],
    )
    return {"analyses": [f"MARKET\n{text}"]}


def tech_architect(state: ReviewState) -> dict:
    print("[Branch] tech")
    text = _review(
        "software architect",
        "What is the hardest engineering problem, and which pieces would you build first?",
        state["proposal"],
    )
    return {"analyses": [f"TECH\n{text}"]}


def security_analyst(state: ReviewState) -> dict:
    print("[Branch] security")
    text = _review(
        "security reviewer",
        "What private data is involved, and what must be true before launch?",
        state["proposal"],
    )
    return {"analyses": [f"SECURITY\n{text}"]}


def synthesizer(state: ReviewState) -> dict:
    """Runs only after market, tech, and security have all written into analyses."""
    print("[Join] synthesizer")
    combined = "\n\n".join(state["analyses"])
    response = llm.invoke([
        SystemMessage(content="You write a 4-line decision: verdict, then one line per risk."),
        HumanMessage(content=(
            f"Proposal: {state['proposal']}\n\n"
            f"Reviews:\n{combined}\n\n"
            "End with exactly one of: GO, CONDITIONAL GO, NO-GO."
        )),
    ])
    return {"verdict": response.content.strip()}


builder = StateGraph(ReviewState)
builder.add_node("market", market_analyst)
builder.add_node("tech", tech_architect)
builder.add_node("security", security_analyst)
builder.add_node("synthesizer", synthesizer)

# Fan-out: three edges from START.
builder.add_edge(START, "market")
builder.add_edge(START, "tech")
builder.add_edge(START, "security")

# Fan-in: synthesizer waits until every incoming branch is done.
builder.add_edge("market", "synthesizer")
builder.add_edge("tech", "synthesizer")
builder.add_edge("security", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


def run_parallel() -> None:
    print_banner(
        "LESSON 111",
        "Parallelization",
        "Three reviewers run in one super-step.\n"
        "operator.add collects their writes. The synthesizer runs after all three finish.",
    )
    print_mermaid(graph, "Parallel review graph")
    print(f"Proposal: {PROPOSAL}\n")

    result = graph.invoke({"proposal": PROPOSAL, "analyses": [], "verdict": ""})

    print("\n--- Collected analyses (order is not guaranteed) ---")
    for item in result["analyses"]:
        print(item)
        print("-" * 40)
    print("\n--- Verdict ---")
    print(result["verdict"])


if __name__ == "__main__":
    safe_run("Lesson 111 Parallelization", run_parallel)
