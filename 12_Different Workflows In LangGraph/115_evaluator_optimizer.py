"""
=============================================================================
LESSON 115: EVALUATOR-OPTIMIZER
=============================================================================

The idea
--------
A model is often better at criticizing a draft than at producing a finished
draft in one shot. So you loop:

    generator writes a solution
         |
    evaluator scores it and writes feedback
         |
         +-- score is high enough, or retries are used up --> END
         |
         +-- otherwise, send the feedback back to the generator

That edge from evaluator back to generator is a cycle. The other workflows
in this folder are acyclic. This one is not.

Always store an iteration count and a max. A cycle without a ceiling will
keep calling the model.

State
-----
task, solution, feedback, score, passed, iteration, max_iterations.

The generator increments iteration. On the first pass there is no feedback,
so it writes a fresh solution. Later passes must address the feedback.

The evaluator uses temperature 0. The pass/fail decision in code uses the
numeric score. Do not trust a separate "passed" boolean from the model if
it disagrees with the score.

Run:
    python "12_Different Workflows In LangGraph/115_evaluator_optimizer.py"
=============================================================================
"""

import json

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run


PASSING_SCORE = 8


class LoopState(TypedDict):
    task: str
    solution: str
    feedback: str
    score: int
    passed: bool
    iteration: int
    max_iterations: int


writer = get_llm(temperature=0.3)
critic = get_llm(temperature=0.0)


def generator(state: LoopState) -> dict:
    iteration = state["iteration"] + 1
    if iteration == 1 or not state["feedback"]:
        print(f"\n[Generator] iteration {iteration}: first draft")
        prompt = (
            f"Task:\n{state['task']}\n\n"
            "Write one Python function. Include type hints and a docstring. "
            "No tests, no explanation outside the code."
        )
    else:
        print(f"\n[Generator] iteration {iteration}: revise using feedback")
        prompt = (
            f"Task:\n{state['task']}\n\n"
            f"Current code:\n{state['solution']}\n\n"
            f"Fix every point in this feedback:\n{state['feedback']}\n\n"
            "Return the full function only."
        )
    response = writer.invoke([
        SystemMessage(content="You write small, correct Python functions."),
        HumanMessage(content=prompt),
    ])
    return {"solution": response.content.strip(), "iteration": iteration}


def _parse_review(raw: str) -> tuple[int, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        score = int(data.get("score", 5))
        feedback = str(data.get("feedback", "No feedback."))
        return max(1, min(10, score)), feedback
    except (json.JSONDecodeError, TypeError, ValueError):
        return 5, "The review was not valid JSON. Make the function easier to check."


def evaluator(state: LoopState) -> dict:
    print(f"[Evaluator] reviewing iteration {state['iteration']}")
    response = critic.invoke([
        SystemMessage(content="You review code and reply with JSON only."),
        HumanMessage(content=(
            f"Task:\n{state['task']}\n\n"
            f"Code:\n{state['solution']}\n\n"
            "Score from 1 to 10. A score of 8 or more means the function "
            "matches every requirement, including empty input and duplicates.\n"
            '{"score": 7, "feedback": "what to fix"}'
        )),
    ])
    score, feedback = _parse_review(response.content)
    passed = score >= PASSING_SCORE
    print(f"[Evaluator] score {score}/10  passed={passed}")
    print(f"[Evaluator] {feedback}")
    return {"score": score, "feedback": feedback, "passed": passed}


def decide(state: LoopState) -> str:
    if state["passed"]:
        print("[Loop] quality bar met. Stop.")
        return END
    if state["iteration"] >= state["max_iterations"]:
        print("[Loop] max iterations reached. Stop with the latest draft.")
        return END
    print("[Loop] send feedback back to the generator.")
    return "generator"


builder = StateGraph(LoopState)
builder.add_node("generator", generator)
builder.add_node("evaluator", evaluator)
builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator", decide, {"generator": "generator", END: END})
graph = builder.compile()


TASK = (
    "Write normalize_names(raw: list[str]) -> list[str]. "
    "Strip each item, collapse repeated spaces, title-case it, drop blanks, "
    "and drop duplicates while keeping the first occurrence. "
    "Return an empty list when the input is empty."
)


def run_loop() -> None:
    print_banner(
        "LESSON 115",
        "Evaluator-Optimizer",
        "The generator drafts. The evaluator scores.\n"
        "A conditional edge either stops or loops back with feedback.",
    )
    print_mermaid(graph, "Evaluator-optimizer cycle")
    print(f"Task: {TASK}\n")

    result = graph.invoke({
        "task": TASK,
        "solution": "",
        "feedback": "",
        "score": 0,
        "passed": False,
        "iteration": 0,
        "max_iterations": 3,
    })

    print("\n--- Final ---")
    print(f"iterations: {result['iteration']}")
    print(f"score:      {result['score']}/10")
    print(f"passed:     {result['passed']}")
    print(result["solution"])


if __name__ == "__main__":
    safe_run("Lesson 115 Evaluator-Optimizer", run_loop)
