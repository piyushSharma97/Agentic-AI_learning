r"""
=============================================================================
LESSON 114: ORCHESTRATOR-WORKER IN LANGGRAPH
=============================================================================

Lesson 113 planned three tasks and ran them in a for-loop inside one node.
The graph shape never changed. It was always:

    orchestrator -> workers -> synthesizer

This lesson makes the workers real graph tasks. After the orchestrator
writes the plan, a conditional edge returns one Send per task:

    Send("worker", {"objective": ..., "task": one_subtask})

LangGraph then runs a copy of the worker node for every Send, in parallel.
The worker writes into worker_results. That field uses operator.add, so
the parallel writes are concatenated. The synthesizer waits until every
worker copy has finished.

Order is not guaranteed. Sort by task_id before you present the report.

        START
          |
     orchestrator          plan = [task1, task2, task3]
          |
          |  Send, Send, Send
         /|\
     worker copies         each sees only its own task
         \|/
     synthesizer
          |
         END

Run:
    python "12_Different Workflows In LangGraph/114_orchestrator_worker_langgraph.py"
=============================================================================
"""

import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

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
    worker_results: Annotated[list[WorkerResult], operator.add]
    report: str


class WorkerState(TypedDict):
    """The payload Send delivers. It is smaller than FlowState."""

    objective: str
    task: SubTask


llm = get_llm(temperature=0.2)

FALLBACK_PLAN: list[SubTask] = [
    {"task_id": 1, "title": "Capture", "instruction": "How audio enters the pipeline."},
    {"task_id": 2, "title": "Summarize", "instruction": "How a call becomes a short summary."},
    {"task_id": 3, "title": "Deliver", "instruction": "Where the summary is stored and who reads it."},
]


def _parse_plan(raw: str) -> list[SubTask]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        plan = json.loads(text)
        if isinstance(plan, list) and plan:
            return plan
    except json.JSONDecodeError:
        pass
    return FALLBACK_PLAN


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


def assign_workers(state: FlowState) -> list[Send]:
    """One Send per planned task. The length of this list is decided at runtime."""
    print(f"[Assign] sending {len(state['plan'])} workers")
    return [
        Send("worker", {"objective": state["objective"], "task": task})
        for task in state["plan"]
    ]


def worker(state: WorkerState) -> dict:
    task = state["task"]
    print(f"[Worker] task {task['task_id']}: {task['title']}")
    response = llm.invoke([
        SystemMessage(content="You complete one subtask in a single short paragraph."),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n"
            f"Subtask: {task['title']}\n"
            f"Instruction: {task['instruction']}"
        )),
    ])
    result: WorkerResult = {
        "task_id": task["task_id"],
        "title": task["title"],
        "output": response.content.strip(),
    }
    return {"worker_results": [result]}


def synthesizer(state: FlowState) -> dict:
    print("[Synthesizer] all workers finished")
    ordered = sorted(state["worker_results"], key=lambda item: item["task_id"])
    sections = "\n\n".join(
        f"{item['task_id']}. {item['title']}\n{item['output']}" for item in ordered
    )
    response = llm.invoke([
        SystemMessage(content="You write a short brief: 2-sentence summary, then 3 next steps."),
        HumanMessage(content=f"Objective: {state['objective']}\n\nSections:\n{sections}"),
    ])
    return {"report": response.content.strip()}


builder = StateGraph(FlowState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("worker", worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
builder.add_edge("worker", "synthesizer")
builder.add_edge("synthesizer", END)
graph = builder.compile()


def run_orchestrator() -> None:
    print_banner(
        "LESSON 114",
        "Orchestrator-Worker in LangGraph",
        "Send creates one parallel worker per subtask.\n"
        "operator.add collects their results. Sort by task_id before synthesis.",
    )
    print_mermaid(graph, "Dynamic worker graph")

    objective = "Design a low-latency pipeline that turns a support call into a 5-line summary."
    print(f"Objective: {objective}\n")

    result = graph.invoke({
        "objective": objective,
        "plan": [],
        "worker_results": [],
        "report": "",
    })

    print("\n--- Worker results (sorted) ---")
    for item in sorted(result["worker_results"], key=lambda row: row["task_id"]):
        print(f"\n{item['task_id']}. {item['title']}\n{item['output']}")
    print("\n--- Report ---")
    print(result["report"])


if __name__ == "__main__":
    safe_run("Lesson 114 Orchestrator-Worker in LangGraph", run_orchestrator)
