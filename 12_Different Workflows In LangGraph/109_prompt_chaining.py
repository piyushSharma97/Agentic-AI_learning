"""
=============================================================================
LESSON 109: PROMPT CHAINING
=============================================================================

The idea
--------
A hard request is easier if you split it into a fixed sequence of small
prompts. The output of one step becomes the input of the next step.

    topic
      -> 1. outline
      -> gate: is the outline good enough?
      -> 2. draft
      -> 3. polish
      -> finished post

Why a chain beats one giant prompt
----------------------------------
Each call has one job, so the model is less likely to skip structure,
wander off topic, or mix drafting with editing. You can also stop early.
If the outline is empty or unstructured, there is no point spending tokens
on a draft.

What this file is
-----------------
Plain Python. Three functions, called in order, plus an if-statement gate.
Lesson 110 builds the same pipeline as a LangGraph graph.

Run:
    python "12_Different Workflows In LangGraph/109_prompt_chaining.py"
=============================================================================
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm, print_banner, safe_run


llm = get_llm(temperature=0.3)


def generate_outline(topic: str, audience: str) -> str:
    """Step 1. One job: structure only. Do not write the article yet."""
    print("\n[Step 1] Generating an outline...")
    response = llm.invoke([
        SystemMessage(content="You write short technical outlines. No prose paragraphs."),
        HumanMessage(content=(
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            "Write exactly 3 numbered sections. "
            "Each section is a title plus one sentence on what it will cover."
        )),
    ])
    return response.content.strip()


def outline_is_usable(outline: str) -> bool:
    """
    The gate sits between steps. It is ordinary Python, not another prompt.
    A failed gate saves the cost of every later model call.
    """
    return len(outline) > 40 and "1" in outline and "2" in outline and "3" in outline


def write_draft(topic: str, audience: str, outline: str) -> str:
    """Step 2. One job: turn the approved outline into a short draft."""
    print("[Step 2] Writing the draft from the outline...")
    response = llm.invoke([
        SystemMessage(content="You write clear technical drafts. Follow the outline."),
        HumanMessage(content=(
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Outline:\n{outline}\n\n"
            "Write exactly 3 short paragraphs, one per outline section."
        )),
    ])
    return response.content.strip()


def polish(audience: str, draft: str) -> str:
    """Step 3. One job: edit. Do not add new sections."""
    print("[Step 3] Polishing the draft...")
    response = llm.invoke([
        SystemMessage(content="You are an editor. You improve clarity and do not add new ideas."),
        HumanMessage(content=(
            f"Audience: {audience}\n\n"
            "Add a one-line title, keep the three paragraphs, "
            "then add two bullet takeaways.\n\n"
            f"Draft:\n{draft}"
        )),
    ])
    return response.content.strip()


def run_prompt_chain() -> None:
    print_banner(
        "LESSON 109",
        "Prompt Chaining",
        "A fixed sequence of focused prompts.\n"
        "Each function does one job. A Python if-statement can stop the chain early.",
    )

    topic = "Why a LangGraph agent needs a checkpointer"
    audience = "Engineers learning agent workflows"

    print(f"Topic: {topic}")
    print(f"Audience: {audience}")

    outline = generate_outline(topic, audience)
    print("\n--- Outline ---")
    print(outline)

    if not outline_is_usable(outline):
        print("\nGate failed. Draft and polish never run.")
        return

    print("\nGate passed. Continuing to the draft.")
    draft = write_draft(topic, audience, outline)
    print("\n--- Draft ---")
    print(draft)

    final_post = polish(audience, draft)
    print("\n--- Polished post ---")
    print(final_post)

    print(
        "\nWhat to notice: step 2 only saw the outline, and step 3 only edited "
        "the draft. That hand-off is prompt chaining."
    )


if __name__ == "__main__":
    safe_run("Lesson 109 Prompt Chaining", run_prompt_chain)
