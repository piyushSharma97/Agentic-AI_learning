"""
=============================================================================
LESSON 113: ORCHESTRATOR-WORKER
=============================================================================

The idea
--------
Prompt chaining and parallelization use a graph you drew in advance.
The steps are known before the user speaks.

An orchestrator-worker flow does not know the steps yet.

    1. Orchestrator reads the goal and writes a plan (a list of subtasks).
    2. A worker carries out each subtask.
    3. A synthesizer stitches the worker outputs into one deliverable.

        goal
          |
          v
     orchestrator  -- "do these 3 things"
          |
          v
        workers
          |
          v
      synthesizer

When to use it
--------------
Use it when two different goals need different steps.
"Design a call-summary pipeline" and "plan a conference workshop"
should not be forced through the same three hardcoded nodes.

What this file does, and what it does not
-----------------------------------------
The workers run inside one node, in a for-loop. That is the easiest way
to see the pattern: plan, do each task, combine. The workers are not
separate graph nodes, and they do not run at the same time.

Lesson 114 turns each planned task into a real parallel node with Send.

Run:
    python "12_Different Workflows In LangGraph/113_orchestrator_worker.py"
=============================================================================
"""

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run


class SubTask(TypedDict):
    task_id: int
    title: str
    instruction: str


class WorkerResult(TypedDict):
    task_id: int
    title: str
    output: str


class FlowState(TypedDict):
    objective: str
    plan: list[SubTask]
    results: list[WorkerResult]
    report: str


llm = get_llm(temperature=0.2)


def _parse_plan(raw: str) -> list[SubTask]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        plan = json.loads(text)
        if isinstance(plan, list) and len(plan) >= 1:
            return plan
    except json.JSONDecodeError:
        pass
    return [
        {"task_id": 1, "title": "Capture", "instruction": "How audio enters the pipeline."},
        {"task_id": 2, "title": "Summarize", "instruction": "How a call becomes a short summary."},
        {"task_id": 3, "title": "Deliver", "instruction": "Where the summary is stored and who reads it."},
    ]


def orchestrator(state: FlowState) -> dict:
    print("[Orchestrator] planning subtasks")
    response = llm.invoke([
        SystemMessage(content="You output only a JSON array. No markdown."),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n\n"
            "Break it into exactly 3 non-overlapping subtasks.\n"
            '[{"task_id": 1, "title": "...", "instruction": "one sentence"},'
            ' {"task_id": 2, "title": "...", "instruction": "one sentence"},'
            ' {"task_id": 3, "title": "...", "instruction": "one sentence"}]'
        )),
    ])
    plan = _parse_plan(response.content)
    for task in plan:
        print(f"  {task['task_id']}. {task['title']}")
    return {"plan": plan}


def workers(state: FlowState) -> dict:
    """
    One graph node walks the plan. Easy to read.
    Lesson 114 replaces this loop with one graph task per subtask.
    """
    print("[Workers] running the plan in order")
    results: list[WorkerResult] = []
    for task in state["plan"]:
        print(f"  working on {task['task_id']}: {task['title']}")
        response = llm.invoke([
            SystemMessage(content="You complete one subtask in a single short paragraph."),
            HumanMessage(content=(
                f"Objective: {state['objective']}\n"
                f"Subtask: {task['title']}\n"
                f"Instruction: {task['instruction']}"
            )),
        ])
        results.append({
            "task_id": task["task_id"],
            "title": task["title"],
            "output": response.content.strip(),
        })
    return {"results": results}


def synthesizer(state: FlowState) -> dict:
    print("[Synthesizer] combining worker output")
    sections = "\n\n".join(
        f"{item['task_id']}. {item['title']}\n{item['output']}"
        for item in state["results"]
    )
    response = llm.invoke([
        SystemMessage(content="You write a short brief: 2-sentence summary, then 3 next steps."),
        HumanMessage(content=f"Objective: {state['objective']}\n\nSections:\n{sections}"),
    ])
    return {"report": response.content.strip()}


builder = StateGraph(FlowState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("workers", workers)
builder.add_node("synthesizer", synthesizer)
builder.add_edge(START, "orchestrator")
builder.add_edge("orchestrator", "workers")
builder.add_edge("workers", "synthesizer")
builder.add_edge("synthesizer", END)
graph = builder.compile()


def run_orchestrator() -> None:
    print_banner(
        "LESSON 113",
        "Orchestrator-Worker",
        "The plan is created at runtime.\n"
        "This version runs workers one after another inside a single node.",
    )
    print_mermaid(graph, "Orchestrator graph (workers are still one node)")

    objective = "Design a low-latency pipeline that turns a support call into a 5-line summary."
    print(f"Objective: {objective}\n")

    result = graph.invoke({
        "objective": objective,
        "plan": [],
        "results": [],
        "report": "",
    })

    print("\n--- Worker results ---")
    for item in result["results"]:
        print(f"\n{item['task_id']}. {item['title']}\n{item['output']}")
    print("\n--- Report ---")
    print(result["report"])


if __name__ == "__main__":
    safe_run("Lesson 113 Orchestrator-Worker", run_orchestrator)
