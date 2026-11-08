"""
=============================================================================
LANGGRAPH COMPONENT 5: CHAINS & WORKFLOWS IN LANGGRAPH
=============================================================================
TOPIC OVERVIEW:
In basic LCEL, pipelines are linear and black-box: `prompt | llm | parser`.
In LangGraph, chains are stateful DAGs (Directed Acyclic Graphs) that allow:
1. Intermediate State Inspection: Inspect state between every single step.
2. Resilience: If step 3 fails, resume from step 2 without re-running step 1.
3. Native Parallel Execution: Fan-out to multiple worker nodes concurrently,
   then fan-in to an aggregator node.

PATTERNS DEMONSTRATED:
Pattern A: Sequential Multi-Stage Processing
   [START] ──> [drafter] ──> [critic] ──> [editor] ──> [END]

Pattern B: Parallel Fan-Out / Fan-In
                ┌──> [tech_analyst]   ──┐
   [START] ─────┤                       ├────> [synthesizer] ──> [END]
                └──> [market_analyst] ──┘
=============================================================================
"""

import sys
from pathlib import Path
from typing_extensions import TypedDict

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# PATTERN A: SEQUENTIAL MULTI-STAGE REFINEMENT CHAIN
# ===========================================================================
class EditorialChainState(TypedDict):
    """
    Schema for the sequential multi-stage workflow.
    Notice that each stage writes to its own dedicated field!
    """
    topic: str
    target_audience: str
    draft: str
    critique: str
    polished_version: str


def draft_content(state: EditorialChainState) -> dict:
    """Stage 1: Generates a 2-sentence draft on the given topic."""
    print("\n[Node 1: Drafter] Generating initial content draft...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise tech blogger. Write a 2-sentence intro draft."),
        ("human", "Topic: {topic}\nTarget Audience: {target_audience}")
    ])
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"topic": state["topic"], "target_audience": state["target_audience"]})
    print(f"Draft Generated:\n{draft.strip()}\n")
    return {"draft": draft}


def critique_content(state: EditorialChainState) -> dict:
    """Stage 2: Reviews the draft and provides actionable editorial feedback."""
    print("[Node 2: Critic] Reviewing draft for clarity and audience resonance...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an editorial reviewer. Provide 1 actionable improvement tip."),
        ("human", "Audience: {target_audience}\nDraft:\n{draft}")
    ])
    chain = prompt | llm | StrOutputParser()
    critique = chain.invoke({"target_audience": state["target_audience"], "draft": state["draft"]})
    print(f"Critique Output:\n{critique.strip()}\n")
    return {"critique": critique}


def polish_content(state: EditorialChainState) -> dict:
    """Stage 3: Applies critique to produce the final polished deliverable."""
    print("[Node 3: Editor] Polishing final version based on editorial feedback...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a master copyeditor. Produce the final refined 2-sentence version."),
        ("human", "Draft:\n{draft}\n\nCritique Feedback:\n{critique}")
    ])
    chain = prompt | llm | StrOutputParser()
    polished = chain.invoke({"draft": state["draft"], "critique": state["critique"]})
    print(f"Polished Version:\n{polished.strip()}\n")
    return {"polished_version": polished}


def build_sequential_chain_graph():
    """Wires up the sequential pipeline: START -> drafter -> critic -> editor -> END"""
    builder = StateGraph(EditorialChainState)
    builder.add_node("drafter", draft_content)
    builder.add_node("critic", critique_content)
    builder.add_node("editor", polish_content)

    builder.add_edge(START, "drafter")
    builder.add_edge("drafter", "critic")
    builder.add_edge("critic", "editor")
    builder.add_edge("editor", END)

    return builder.compile()


# ===========================================================================
# PATTERN B: PARALLEL FAN-OUT / FAN-IN WORKFLOW
# ===========================================================================
class DualAnalysisState(TypedDict):
    """
    Schema for parallel workflow. Two analysts write independent keys,
    and the executive synthesizer reads both.
    """
    company_name: str
    tech_analysis: str
    market_analysis: str
    executive_synthesis: str


def analyze_technology(state: DualAnalysisState) -> dict:
    """Parallel Worker A: Analyzes technical moat and IP."""
    print("[Parallel Branch A: Tech Analyst] Evaluating technology stack and IP...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Assess the technological moat of {company} in 1 sentence.")
    res = (prompt | llm | StrOutputParser()).invoke({"company": state["company_name"]})
    return {"tech_analysis": res}


def analyze_market(state: DualAnalysisState) -> dict:
    """Parallel Worker B: Analyzes market demand and competition."""
    print("[Parallel Branch B: Market Analyst] Evaluating market demand and competition...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Assess the market competition and demand for {company} in 1 sentence.")
    res = (prompt | llm | StrOutputParser()).invoke({"company": state["company_name"]})
    return {"market_analysis": res}


def synthesize_executive_summary(state: DualAnalysisState) -> dict:
    """Fan-In Aggregator: Combines both parallel insights into a final memo."""
    print("\n[Fan-In Aggregator: Executive] Synthesizing parallel insights into final memo...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Synthesize into a 2-sentence investment memo:\nTech Moat: {tech}\nMarket Demand: {market}"
    )
    res = (prompt | llm | StrOutputParser()).invoke({
        "tech": state["tech_analysis"],
        "market": state["market_analysis"]
    })
    return {"executive_synthesis": res}


def build_parallel_chain_graph():
    """
    Wires up Fan-Out / Fan-In:
    - Fan-out: START connects to BOTH tech_analyst and market_analyst
    - Fan-in: BOTH tech_analyst and market_analyst connect to synthesizer
    """
    builder = StateGraph(DualAnalysisState)

    builder.add_node("tech_analyst", analyze_technology)
    builder.add_node("market_analyst", analyze_market)
    builder.add_node("synthesizer", synthesize_executive_summary)

    # Fan-Out
    builder.add_edge(START, "tech_analyst")
    builder.add_edge(START, "market_analyst")

    # Fan-In (LangGraph ensures synthesizer waits for BOTH parallel workers)
    builder.add_edge("tech_analyst", "synthesizer")
    builder.add_edge("market_analyst", "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder.compile()


# ===========================================================================
# DEMONSTRATION RUNNER
# ===========================================================================
def main():
    def demo():
        print("=== 1. Sequential Chain: Draft -> Critique -> Polish ===")
        seq_graph = build_sequential_chain_graph()
        print_mermaid(seq_graph, "Sequential Chain Topology")
        seq_res = seq_graph.invoke({
            "topic": "Zero-Knowledge Proofs in Blockchain",
            "target_audience": "Undergraduate Computer Science Students",
            "draft": "",
            "critique": "",
            "polished_version": ""
        })
        print(f"Final Polished Deliverable:\n{seq_res['polished_version'].strip()}\n")

        print("=== 2. Parallel Fan-Out / Fan-In Workflow ===")
        par_graph = build_parallel_chain_graph()
        print_mermaid(par_graph, "Parallel Fan-Out / Fan-In Topology")
        par_res = par_graph.invoke({
            "company_name": "NVIDIA",
            "tech_analysis": "",
            "market_analysis": "",
            "executive_synthesis": ""
        })
        print(f"\nFinal Executive Memo:\n{par_res['executive_synthesis'].strip()}\n")

    safe_run("Chains and Multi-Stage Workflows in LangGraph", demo)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 05: Chains and Workflows
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates two fundamental workflow architectures in LangGraph:
# 1. Sequential Chain: Linear progression through multi-stage refinement.
# 2. Parallel Fan-Out / Fan-In: Concurrent execution of independent worker
#    nodes followed by an aggregator node that synchronizes their findings.
#
# KEY CONCEPTS USED:
# • Dedicated State Keys: Each node stores its result in a separate field,
#   allowing complete auditability and inspection of reasoning steps.
# • Fan-Out: Creating multiple outgoing edges from a single node (e.g. START -> A, START -> B).
# • Fan-In: Wiring multiple incoming edges into an aggregator node (A -> C, B -> C).
#   LangGraph guarantees C only executes after BOTH A and B complete.
#
# HOW IT CONNECTS TO THE PROJECT:
# This bridges simple single-call chains to complex multi-agent architectures
# where agents perform research from multiple perspectives before deciding.
#
# THINGS TO REMEMBER:
# • Parallel nodes must not overwrite the exact same state key unless using a reducer.
# • Always assign separate keys (`tech_analysis`, `market_analysis`) for parallel branches.
#
# COMMON MISTAKES TO AVOID:
# • Trying to run parallel workers that depend on each other's outputs.
# • Collapsing all intermediate outputs into a single string, losing provenance.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `build_parallel_chain_graph()`: notice how `START` connects to both analysts,
# and both analysts connect to `synthesizer`.
# ============================================================
