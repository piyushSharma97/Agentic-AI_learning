"""
=============================================================================
LANGGRAPH COMPONENT 5: CHAINS & WORKFLOWS IN LANGGRAPH
=============================================================================
WHY BUILD CHAINS IN LANGGRAPH INSTEAD OF BASIC LCEL?
In basic LCEL (Prompt | LLM | Parser), the execution is a black-box pipeline.
In LangGraph, chains are stateful DAGs (Directed Acyclic Graphs) offering:
1. Intermediate State Inspection: Inspect and log state between every node.
2. Resilience & Checkpointing: If step 3 of 5 fails, resume from step 2 without re-running step 1.
3. Native Parallel Execution: Fan-out to multiple worker nodes concurrently, then fan-in to an aggregator.
4. Human-in-the-Loop: Halt execution at any point for human review.

Key Patterns Demonstrated:
1. Pattern A: Sequential Multi-Stage Processing (Draft -> Critique -> Polish)
2. Pattern B: Parallel Fan-Out / Fan-In (Tech Analyst + Financial Analyst -> Synthesis)
=============================================================================
"""

from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import shared helper
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# PATTERN A: SEQUENTIAL MULTI-STAGE REFINEMENT CHAIN
# ===========================================================================
class EditorialChainState(TypedDict):
    topic: str
    target_audience: str
    draft: str
    critique: str
    polished_version: str


def draft_content(state: EditorialChainState) -> dict:
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
    company_name: str
    tech_analysis: str
    market_analysis: str
    executive_synthesis: str


def analyze_technology(state: DualAnalysisState) -> dict:
    print("[Parallel Branch A: Tech Analyst] Evaluating technology stack and IP...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Assess the technological moat of {company} in 1 sentence.")
    res = (prompt | llm | StrOutputParser()).invoke({"company": state["company_name"]})
    return {"tech_analysis": res}


def analyze_market(state: DualAnalysisState) -> dict:
    print("[Parallel Branch B: Market Analyst] Evaluating market demand and competition...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Assess the market competition and demand for {company} in 1 sentence.")
    res = (prompt | llm | StrOutputParser()).invoke({"company": state["company_name"]})
    return {"market_analysis": res}


def synthesize_executive_summary(state: DualAnalysisState) -> dict:
    print("\n[Fan-In Aggregator: Executive] Synthesizing parallel insights into final memo...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Synthesize into an investment memo:\nTech Moat: {tech}\nMarket Demand: {market}"
    )
    res = (prompt | llm | StrOutputParser()).invoke({
        "tech": state["tech_analysis"],
        "market": state["market_analysis"]
    })
    return {"executive_synthesis": res}


def build_parallel_chain_graph():
    builder = StateGraph(DualAnalysisState)

    # Add parallel worker nodes
    builder.add_node("tech_analyst", analyze_technology)
    builder.add_node("market_analyst", analyze_market)
    builder.add_node("synthesizer", synthesize_executive_summary)

    # Fan-out: START connects to BOTH nodes simultaneously
    builder.add_edge(START, "tech_analyst")
    builder.add_edge(START, "market_analyst")

    # Fan-in: BOTH parallel nodes feed into the synthesizer
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

