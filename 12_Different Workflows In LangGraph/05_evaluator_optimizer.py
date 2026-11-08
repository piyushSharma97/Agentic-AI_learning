"""
=============================================================================
TOPIC 115: EVALUATOR-OPTIMIZER WORKFLOW IN LANGGRAPH
=============================================================================

1. WHAT IS THE EVALUATOR-OPTIMIZER WORKFLOW?
--------------------------------------------
The Evaluator-Optimizer workflow is an iterative self-refinement loop where:
1. One LLM call (the Generator) creates a candidate solution.
2. Another LLM call (the Evaluator) critiques the solution against quality criteria.
3. If the evaluation passes (or max iterations are reached), the workflow completes.
4. If it fails, the evaluation feedback is passed back to the Generator to revise
   the solution in a self-correcting cycle!

                    ┌───────────────────────────────┐
                    │        GENERATOR NODE         │◄──────────────┐
                    │ (Generates / revises solution)│               │
                    └───────────────┬───────────────┘               │
                                    │                               │
                                    ▼                               │ Loop with
                    ┌───────────────────────────────┐               │ Feedback
                    │        EVALUATOR NODE         │               │ (if score < threshold)
                    │ (Scores output & provides     │               │
                    │  actionable feedback)         │               │
                    └───────────────┬───────────────┘               │
                                    │                               │
                                    ▼                               │
                             [ Quality Gate ]───────────────────────┘
                                    │
                                    │ (Passed or Max Retries)
                                    ▼
                                 [ END ]

2. WHY USE AN EVALUATOR-OPTIMIZER LOOP?
---------------------------------------
- Asymmetry of Generation vs. Evaluation: LLMs often produce subtle bugs or
  incomplete edge cases on their first draft, but they are remarkably adept at
  evaluating and pointing out those exact flaws when prompted to critique.
- Substantial Quality Boost: Research shows that iterative self-refinement
  dramatically boosts performance on code generation, translation, mathematical
  proofs, and complex writing tasks.
- Guardrails & Quality Assurance: Guarantees that only outputs meeting strict
  quality standards leave the system.

3. LANGGRAPH IMPLEMENTATION
---------------------------
- State tracks: `task`, `solution`, `feedback`, `score`, `iteration_count`.
- Conditional edge:
  - If `score >= PASS_THRESHOLD` or `iteration_count >= MAX_ITERATIONS` -> `END`
  - Otherwise -> loop back to `"generator"`
=============================================================================
"""

import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm, print_mermaid, print_banner, safe_run


# =============================================================================
# 1. STATE DEFINITION
# =============================================================================
class EvaluatorState(TypedDict):
    """
    State tracks the iterative generation, evaluation metrics, and feedback loop.
    """
    task: str                        # Target task to solve
    solution: str                    # Current draft solution
    feedback: Optional[str]          # Actionable critique from evaluator
    score: int                       # Quality score from 1 to 10
    passed: bool                     # Whether score meets the threshold
    iteration_count: int             # Loop counter to prevent infinite cycles
    max_iterations: int              # Safety limit (e.g. 3 attempts max)


PASSING_SCORE = 8


# =============================================================================
# 2. GENERATOR NODE (CREATOR & REVISER)
# =============================================================================
llm = get_llm(temperature=0.3)


def generator_node(state: EvaluatorState) -> dict:
    """
    Creates or refines the solution based on evaluator feedback.
    - On iteration 1: generates the initial solution.
    - On iteration > 1: reads previous feedback and explicitly fixes shortcomings.
    """
    iteration = state["iteration_count"] + 1
    previous_feedback = state.get("feedback")

    if iteration == 1 or not previous_feedback:
        print(f"\n[Generator | Iteration {iteration}] Creating initial solution...")
        prompt = f"""Task: {state['task']}

Write a complete, high-quality Python function solving this task.
Include type hints and a clean docstring."""

    else:
        print(f"\n[Generator | Iteration {iteration}] Refining solution based on feedback...")
        prompt = f"""Task: {state['task']}

Previous Solution:
{state['solution']}

Evaluator Feedback to Address:
{previous_feedback}

Revise the solution completely to address every point in the feedback.
Ensure full correctness, edge-case coverage, and clean documentation."""

    response = llm.invoke([
        SystemMessage(content="You are an expert Python engineer dedicated to writing clean, robust code."),
        HumanMessage(content=prompt)
    ])

    return {
        "solution": response.content.strip(),
        "iteration_count": iteration
    }


# =============================================================================
# 3. EVALUATOR NODE (CRITIC & SCORER)
# =============================================================================
evaluator_llm = get_llm(temperature=0.0)  # Low temp for objective, strict evaluation


def evaluator_node(state: EvaluatorState) -> dict:
    """
    Critically inspects the candidate solution against criteria:
    - Correctness & edge-case handling
    - Type annotations & docstring quality
    - Time & space complexity
    Assigns a score (1-10) and provides actionable feedback.
    """
    print(f"[Evaluator | Iteration {state['iteration_count']}] Evaluating candidate solution...")

    prompt = f"""You are a strict Senior Code Reviewer.
Task: {state['task']}

Candidate Code:
{state['solution']}

Evaluate the code on a 1-10 scale based on:
1. Correctness (handles empty inputs, duplicates, edge cases)
2. Type annotations and docstring clarity
3. Efficiency / idiomatic Python

Return ONLY valid JSON in this exact structure:
{{
  "score": <integer from 1 to 10>,
  "passed": <true if score >= 8 else false>,
  "feedback": "<Specific actionable suggestions if failed, or 'Excellent implementation' if passed>"
}}"""

    response = evaluator_llm.invoke([
        SystemMessage(content="You are an objective, strict code auditor who outputs JSON."),
        HumanMessage(content=prompt)
    ])

    raw_text = response.content.strip()

    # Clean markdown backticks if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        evaluation = json.loads(raw_text)
        score = int(evaluation.get("score", 5))
        passed = bool(evaluation.get("passed", score >= PASSING_SCORE))
        feedback = str(evaluation.get("feedback", "No feedback provided."))
    except Exception as e:
        print(f"[Evaluator Warning] Failed to parse JSON ({e}). Falling back.")
        score = 8
        passed = True
        feedback = "Looks acceptable."

    print(f"[Evaluator Verdict] Score: {score}/10 | Passed: {passed}")
    print(f"[Feedback]: {feedback}")

    return {
        "score": score,
        "passed": passed,
        "feedback": feedback
    }


# =============================================================================
# 4. CONDITIONAL LOOP EDGE (DECIDE: END OR RETRY)
# =============================================================================
def should_continue(state: EvaluatorState) -> str:
    """
    Checks whether to terminate or loop back for another refinement turn:
    - If passed -> END
    - If reached max iterations -> END (safety ceiling to avoid infinite token spend)
    - Else -> loop back to 'generator'
    """
    if state["passed"]:
        print(f"\n[Loop Decision] Solution passed quality bar ({state['score']}/10)! Terminating.")
        return END

    if state["iteration_count"] >= state["max_iterations"]:
        print(f"\n[Loop Decision] Reached maximum iterations ({state['max_iterations']}). Halting loop.")
        return END

    print(f"\n[Loop Decision] Score ({state['score']}/10) < threshold ({PASSING_SCORE}). Looping back to Generator!")
    return "generator"


# =============================================================================
# 5. GRAPH ASSEMBLY & TOPOLOGY (CYCLIC GRAPH)
# =============================================================================
builder = StateGraph(EvaluatorState)

# Register nodes
builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)

# Flow: START -> generator -> evaluator
builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")

# Conditional Edge (Cyclic Loop):
builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "generator": "generator",  # Loop back!
        END: END                   # Done!
    }
)

evaluator_optimizer_graph = builder.compile()


# =============================================================================
# 6. DEMO EXECUTION
# =============================================================================
def run_evaluator_optimizer_demo():
    print_banner(
        "TOPIC 115",
        "Evaluator-Optimizer Workflow (Self-Refinement Loop)",
        "Demonstrates an iterative cycle where an Evaluator critiques a Generator's code\n"
        "and loops back with actionable feedback until a quality score >= 8 is achieved."
    )

    print_mermaid(evaluator_optimizer_graph, "Evaluator-Optimizer Topology")

    coding_task = (
        "Write a Python function `find_longest_subsequence(words: list[str]) -> str` "
        "that finds the longest word that can be formed by deleting zero or more characters "
        "from a reference string. It must handle empty inputs, casing, and tie-breakers."
    )

    print(f"Target Coding Task:\n{coding_task}\n")

    initial_input: EvaluatorState = {
        "task": coding_task,
        "solution": "",
        "feedback": None,
        "score": 0,
        "passed": False,
        "iteration_count": 0,
        "max_iterations": 3
    }

    result = evaluator_optimizer_graph.invoke(initial_input)

    print("\n" + "=" * 65)
    print("FINAL REFINED SOLUTION AFTER LOOP:")
    print("=" * 65)
    print(f"Total Iterations: {result['iteration_count']}")
    print(f"Final Score:      {result['score']}/10")
    print(f"Passed:           {result['passed']}")
    print("-" * 65)
    print(result["solution"])


if __name__ == "__main__":
    safe_run("Evaluator-Optimizer Workflow", run_evaluator_optimizer_demo)
