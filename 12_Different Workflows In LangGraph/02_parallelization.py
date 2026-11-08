"""
=============================================================================
TOPIC 111: PARALLELIZATION WORKFLOW IN LANGGRAPH
=============================================================================

1. WHAT IS PARALLELIZATION?
---------------------------
Parallelization is a workflow pattern where multiple independent tasks or
analyses are executed simultaneously rather than sequentially.

There are two primary sub-patterns of parallelization:

A. SECTIONING:
   Breaking a complex evaluation into independent dimensions evaluated
   simultaneously.
   Example:
                    ┌─────────────────────────┐
                    │ Market Analyst (Node A) │
              ┌────►│                         ├─────┐
              │     └─────────────────────────┘     │
   [Input]    │     ┌─────────────────────────┐     │     ┌────────────────────────┐
   Proposal ──┼────►│ Tech Architect (Node B) ├─────┼────►│ Executive Synthesizer  │
              │     └─────────────────────────┘     │     │ (Aggregates & Decides) │
              │     ┌─────────────────────────┐     │     └────────────────────────┘
              └────►│ Security Analyst (Node C)─────┘
                    └─────────────────────────┘
                    (Fan-Out: Runs Concurrently)   (Fan-In: Waits for All Branches)

B. VOTING / ENSEMBLING:
   Running the exact same query through multiple distinct prompts, temperatures,
   or models to check consensus and select the highest-confidence answer.

2. WHY USE PARALLELIZATION?
---------------------------
- Wall-Clock Latency: 3 independent calls taking 2 seconds each take ~2 seconds
  in parallel, instead of 6 seconds sequentially.
- Specialization: Each parallel branch operates with its own distinct persona,
  system prompt, and focus area without cross-contamination.
- Reducers Handle Concurrency: In LangGraph, parallel nodes can safely return
  updates to an `Annotated[list, operator.add]` state field. LangGraph merges
  them automatically without race conditions.
=============================================================================
"""

import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm, print_mermaid, print_banner, safe_run


# =============================================================================
# 1. STATE DEFINITION WITH REDUCERS
# =============================================================================
class ParallelState(TypedDict):
    """
    State for parallel multi-analyst review.
    KEY CONCEPT: `Annotated[List[str], operator.add]`
    When Node A, Node B, and Node C complete in parallel, their returned list
    items are CONCATENATED together by `operator.add` rather than overwriting each other!
    """
    proposal: str                                                 # The startup/product idea
    analyses: Annotated[List[str], operator.add]                  # Accumulated reviews from all parallel nodes
    final_verdict: str                                            # Synthesizer's consolidated decision


# =============================================================================
# 2. PARALLEL SPECIALIST NODES (FAN-OUT BRANCHES)
# =============================================================================
llm = get_llm(temperature=0.2)


def market_analyst_node(state: ParallelState) -> dict:
    """Parallel Branch 1: Evaluates business model, market size, and competition."""
    print("[Parallel Branch 1] Market Analyst is reviewing...")

    prompt = f"""You are an expert Venture Capital Market Analyst.
Analyze the following startup proposal:
'{state['proposal']}'

Provide a concise 2-sentence assessment covering:
1. Target market size & opportunity
2. Key commercial risk or competitor challenge."""

    response = llm.invoke([
        SystemMessage(content="You are a sharp, data-driven market analyst."),
        HumanMessage(content=prompt)
    ])

    return {"analyses": [f"📊 [MARKET ANALYSIS]:\n{response.content.strip()}"]}


def tech_architect_node(state: ParallelState) -> dict:
    """Parallel Branch 2: Evaluates technical feasibility, stack, and complexity."""
    print("[Parallel Branch 2] Technical Architect is reviewing...")

    prompt = f"""You are a Principal Cloud Solutions Architect.
Analyze the following startup proposal:
'{state['proposal']}'

Provide a concise 2-sentence assessment covering:
1. Technical feasibility & primary engineering bottleneck
2. Recommended core infrastructure or AI framework."""

    response = llm.invoke([
        SystemMessage(content="You are a realistic, senior software systems architect."),
        HumanMessage(content=prompt)
    ])

    return {"analyses": [f"💻 [TECHNICAL FEASIBILITY]:\n{response.content.strip()}"]}


def security_analyst_node(state: ParallelState) -> dict:
    """Parallel Branch 3: Evaluates data privacy, security, and compliance risks."""
    print("[Parallel Branch 3] Security & Compliance Officer is reviewing...")

    prompt = f"""You are a Chief Information Security Officer (CISO).
Analyze the following startup proposal:
'{state['proposal']}'

Provide a concise 2-sentence assessment covering:
1. Primary data privacy (GDPR/HIPAA) or AI safety risk
2. Critical security guardrail required before launch."""

    response = llm.invoke([
        SystemMessage(content="You are a strict enterprise security and compliance auditor."),
        HumanMessage(content=prompt)
    ])

    return {"analyses": [f"🔒 [SECURITY & COMPLIANCE]:\n{response.content.strip()}"]}


# =============================================================================
# 3. SYNTHESIS NODE (FAN-IN AGGREGATION)
# =============================================================================
def executive_synthesizer_node(state: ParallelState) -> dict:
    """
    Fan-In Aggregator:
    This node automatically waits until ALL parallel branches (Market, Tech, Security)
    have finished. It synthesizes their findings into an executive decision.
    """
    print("[Fan-In Aggregator] Synthesizing all 3 analyses into Executive Verdict...")

    combined_analyses = "\n\n".join(state["analyses"])

    prompt = f"""You are an Executive Board Member.
Proposal: '{state['proposal']}'

Here are the independent evaluations from your 3 domain experts:
{combined_analyses}

Provide an Executive Summary:
1. Verdict: [GO / CONDITIONAL GO / NO-GO]
2. 2-sentence justification synthesizing the market, tech, and security trade-offs.
3. Top action item for the team."""

    response = llm.invoke([
        SystemMessage(content="You make decisive, balanced executive investment decisions."),
        HumanMessage(content=prompt)
    ])

    return {"final_verdict": response.content.strip()}


# =============================================================================
# 4. GRAPH ASSEMBLY & TOPOLOGY (FAN-OUT / FAN-IN)
# =============================================================================
builder = StateGraph(ParallelState)

# Register the nodes
builder.add_node("market_analyst", market_analyst_node)
builder.add_node("tech_architect", tech_architect_node)
builder.add_node("security_analyst", security_analyst_node)
builder.add_node("synthesizer", executive_synthesizer_node)

# --- FAN-OUT ---
# Adding multiple edges originating from START triggers concurrent execution:
builder.add_edge(START, "market_analyst")
builder.add_edge(START, "tech_architect")
builder.add_edge(START, "security_analyst")

# --- FAN-IN ---
# All three parallel branches route to the single synthesizer node.
# LangGraph guarantees the synthesizer will not run until all 3 nodes resolve!
builder.add_edge("market_analyst", "synthesizer")
builder.add_edge("tech_architect", "synthesizer")
builder.add_edge("security_analyst", "synthesizer")

# Finish
builder.add_edge("synthesizer", END)

parallel_graph = builder.compile()


# =============================================================================
# 5. DEMO EXECUTION
# =============================================================================
def run_parallelization_demo():
    print_banner(
        "TOPIC 111",
        "Parallelization Workflow (Fan-Out / Fan-In)",
        "Demonstrates concurrent execution of 3 independent specialist nodes\n"
        "(Market, Tech, Security) followed by a synchronized Executive Synthesizer."
    )

    print_mermaid(parallel_graph, "Parallel Workflow Topology")

    sample_proposal = (
        "MedAI Assistant: An autonomous multi-modal agent that analyzes private patient "
        "medical health records and hospital imaging scans to automatically suggest differential diagnoses "
        "and generate prescription drafts directly into pharmacy networks."
    )

    print(f"Startup Proposal Under Review:\n\"{sample_proposal}\"\n")

    initial_input: ParallelState = {
        "proposal": sample_proposal,
        "analyses": [],
        "final_verdict": ""
    }

    result = parallel_graph.invoke(initial_input)

    print("\n" + "=" * 60)
    print("ALL CONCURRENT ANALYSES (Aggregated via State Reducer):")
    print("=" * 60)
    for analysis in result["analyses"]:
        print(analysis)
        print("-" * 50)

    print("\n" + "=" * 60)
    print("FINAL EXECUTIVE SYNTHESIS:")
    print("=" * 60)
    print(result["final_verdict"])


if __name__ == "__main__":
    safe_run("Parallelization Workflow", run_parallelization_demo)
