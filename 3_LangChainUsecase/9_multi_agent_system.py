"""
=============================================================================
USE CASE 9: MULTI-AGENT SYSTEMS IN LANGCHAIN & LANGGRAPH
=============================================================================
Single-agent systems often suffer from cognitive overload, hallucination,
and prompt drift when asked to research, write, verify, and format complex tasks.

Multi-Agent Systems decompose complex workflows into specialized, autonomous roles:
1. Researcher Agent: Gathers facts, data points, and technical specifics.
2. Writer Agent: Synthesizes raw research into structured, compelling drafts.
3. Critic / Reviewer Agent: Audits the draft against quality criteria and signs off.

Key Concepts Covered:
1. Multi-Agent separation of concerns
2. State Graph architecture (`StateGraph`, `START`, `END`)
3. Passing and mutating shared workflow state (`TypedDict`)
4. Peer-to-Peer collaborative pipeline with agent handoffs
=============================================================================
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Define Shared Graph State
# ---------------------------------------------------------------------------
class MultiAgentState(TypedDict):
    topic: str
    target_audience: str
    research_notes: str
    draft_article: str
    review_feedback: str
    final_status: str


# ---------------------------------------------------------------------------
# 2. Agent 1: The Researcher
# ---------------------------------------------------------------------------
def researcher_node(state: MultiAgentState) -> dict:
    print("\n[Agent 1: Researcher] Analyzing topic and gathering core facts...")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a specialized Research Agent. "
            "Given a topic and target audience, identify exactly 3 key technical trends, "
            "facts, or architectural considerations. Format as bullet points."
        )),
        ("human", "Topic: {topic}\nTarget Audience: {target_audience}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    notes = chain.invoke({
        "topic": state["topic"],
        "target_audience": state["target_audience"]
    })
    
    print(f"[Researcher Findings]:\n{notes.strip()}\n")
    return {"research_notes": notes}


# ---------------------------------------------------------------------------
# 3. Agent 2: The Writer
# ---------------------------------------------------------------------------
def writer_node(state: MultiAgentState) -> dict:
    print("\n[Agent 2: Writer] Synthesizing research notes into an executive article...")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Technical Writer. "
            "Using the provided research notes, craft an engaging 2-paragraph executive brief "
            "tailored for the specified audience."
        )),
        ("human", "Topic: {topic}\nAudience: {target_audience}\nResearch Notes:\n{research_notes}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "topic": state["topic"],
        "target_audience": state["target_audience"],
        "research_notes": state["research_notes"]
    })
    
    print(f"[Writer Draft]:\n{draft.strip()}\n")
    return {"draft_article": draft}


# ---------------------------------------------------------------------------
# 4. Agent 3: The Critic / Quality Reviewer
# ---------------------------------------------------------------------------
def critic_node(state: MultiAgentState) -> dict:
    print("\n[Agent 3: Quality Critic] Auditing draft for clarity, tone, and rigor...")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an Editorial Reviewer and Quality Auditor. "
            "Evaluate the draft based on clarity, tone, and audience fit. "
            "Provide a 1-sentence verdict and rating (APPROVED or NEEDS_REVISION)."
        )),
        ("human", "Audience: {target_audience}\nDraft Article:\n{draft_article}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    feedback = chain.invoke({
        "target_audience": state["target_audience"],
        "draft_article": state["draft_article"]
    })
    
    print(f"[Critic Feedback]:\n{feedback.strip()}\n")
    return {
        "review_feedback": feedback,
        "final_status": "APPROVED"
    }


# ---------------------------------------------------------------------------
# 5. Build and Compile the Multi-Agent StateGraph
# ---------------------------------------------------------------------------
def build_multi_agent_graph():
    builder = StateGraph(MultiAgentState)
    
    # Register agent nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    
    # Define agent workflow edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_edge("critic", END)
    
    return builder.compile()


# ---------------------------------------------------------------------------
# 6. Execute Multi-Agent Workflow
# ---------------------------------------------------------------------------
def demo_multi_agent_collaboration():
    print("--- Multi-Agent Collaborative System (Researcher -> Writer -> Critic) ---")
    graph = build_multi_agent_graph()
    
    initial_input: MultiAgentState = {
        "topic": "Edge AI and On-Device Small Language Models (SLMs)",
        "target_audience": "Chief Technology Officers (CTOs)",
        "research_notes": "",
        "draft_article": "",
        "review_feedback": "",
        "final_status": ""
    }
    
    final_output = graph.invoke(initial_input)
    
    print("=" * 60)
    print("FINAL MULTI-AGENT DELIVERABLE:")
    print("=" * 60)
    print(f"Status: {final_output['final_status']}")
    print(f"\nFinal Article:\n{final_output['draft_article'].strip()}")
    print(f"\nAudit Report:\n{final_output['review_feedback'].strip()}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Multi-Agent System Demonstrations", lambda: [
        demo_multi_agent_collaboration(),
    ])


if __name__ == "__main__":
    main()

