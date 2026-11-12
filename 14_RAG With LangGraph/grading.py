"""
A small yes/no check used by both RAG graphs.

The grader sees one question and one block of context. It does not write
the answer. The graph uses the boolean to choose the next node.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Relevance(BaseModel):
    """Structured output keeps the grader from adding a paragraph."""

    binary_score: Literal["yes", "no"] = Field(
        description="yes only if the context contains the facts needed to answer the question"
    )


def is_relevant(llm, question: str, context: str) -> bool:
    """Return True when this context can answer the question."""
    prompt = (
        "You grade retrieval for a RAG system.\n"
        "Score yes only when the context states facts that answer the question.\n"
        "Score no when the topic is different, or the context is only vaguely related.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context[:1500]}\n\n"
        "Reply with the structured score."
    )
    try:
        result = llm.with_structured_output(Relevance).invoke(prompt)
        if isinstance(result, Relevance):
            score = result.binary_score
        else:
            score = str(result["binary_score"])
    except Exception as exc:
        # Some models reject structured output. A single yes/no word is enough.
        print(f"[grade] structured output unavailable ({exc}). Reading a yes/no word.")
        text = llm.invoke(prompt + "\nReply with one word: yes or no.").content.lower()
        score = "yes" if "yes" in text else "no"

    print(f"[grade] {score}")
    return score.strip().lower() == "yes"
