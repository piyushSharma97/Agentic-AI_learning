"""
=============================================================================
LESSON 110: PROMPT CHAINING WITH LANGGRAPH
=============================================================================

Lesson 109 called three functions in order:

    outline = generate_outline(...)
    if outline_is_usable(outline):
        draft = write_draft(...)
        final_post = polish(...)

This lesson is the same pipeline, written as a graph.

    START -> generate_outline -> gate -> write_draft -> polish -> END
                                   |
                                   +-> END   (outline rejected)

What LangGraph adds
-------------------
1. State. A TypedDict holds topic, outline, draft, and the final post.
   A node returns only the keys it changes. LangGraph merges them in.
2. Edges. add_edge("write_draft", "polish") is the arrow between steps.
3. A conditional edge. The gate is a function that returns the next node
   name. It does not call the model.

Run:
    python "12_Different Workflows In LangGraph/110_prompt_chaining_langgraph.py"
=============================================================================
"""

from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run


class ChainState(TypedDict):
    """Every node reads this dict and returns a partial update."""

    topic: str
    audience: str
    outline: Optional[str]
    draft: Optional[str]
    final_post: Optional[str]
    outline_ok: bool


llm = get_llm(temperature=0.3)


def generate_outline(state: ChainState) -> dict:
    print("[Node] generate_outline")
    response = llm.invoke([
        SystemMessage(content="You write short technical outlines. No prose paragraphs."),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Audience: {state['audience']}\n\n"
            "Write exactly 3 numbered sections. "
            "Each section is a title plus one sentence on what it will cover."
        )),
    ])
    outline = response.content.strip()
    outline_ok = len(outline) > 40 and "2" in outline and "3" in outline
    return {"outline": outline, "outline_ok": outline_ok}


def write_draft(state: ChainState) -> dict:
    print("[Node] write_draft")
    response = llm.invoke([
        SystemMessage(content="You write clear technical drafts. Follow the outline."),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Audience: {state['audience']}\n\n"
            f"Outline:\n{state['outline']}\n\n"
            "Write exactly 3 short paragraphs, one per outline section."
        )),
    ])
    return {"draft": response.content.strip()}


def polish(state: ChainState) -> dict:
    print("[Node] polish")
    response = llm.invoke([
        SystemMessage(content="You are an editor. You improve clarity and do not add new ideas."),
        HumanMessage(content=(
            f"Audience: {state['audience']}\n\n"
            "Add a one-line title, keep the three paragraphs, "
            "then add two bullet takeaways.\n\n"
            f"Draft:\n{state['draft']}"
        )),
    ])
    return {"final_post": response.content.strip()}


def gate(state: ChainState) -> str:
    """Return the name of the next node. This function does not call the model."""
    if state["outline_ok"]:
        print("[Gate] Outline accepted. Next node: write_draft")
        return "write_draft"
    print("[Gate] Outline rejected. Next node: END")
    return END


builder = StateGraph(ChainState)
builder.add_node("generate_outline", generate_outline)
builder.add_node("write_draft", write_draft)
builder.add_node("polish", polish)

builder.add_edge(START, "generate_outline")
builder.add_conditional_edges(
    "generate_outline",
    gate,
    {"write_draft": "write_draft", END: END},
)
builder.add_edge("write_draft", "polish")
builder.add_edge("polish", END)

chain = builder.compile()


def run_prompt_chain() -> None:
    print_banner(
        "LESSON 110",
        "Prompt Chaining in LangGraph",
        "Same three steps as lesson 109.\n"
        "State stores each result. Edges decide the order. The gate is a conditional edge.",
    )
    print_mermaid(chain, "Prompt chaining graph")

    result = chain.invoke({
        "topic": "Why a LangGraph agent needs a checkpointer",
        "audience": "Engineers learning agent workflows",
        "outline": None,
        "draft": None,
        "final_post": None,
        "outline_ok": False,
    })

    print("\n--- State after the graph finishes ---")
    print("outline_ok:", result["outline_ok"])
    print("\nOUTLINE\n", result["outline"])
    print("\nDRAFT\n", result["draft"])
    print("\nFINAL POST\n", result["final_post"])


if __name__ == "__main__":
    safe_run("Lesson 110 Prompt Chaining in LangGraph", run_prompt_chain)
