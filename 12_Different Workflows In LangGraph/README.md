# Different workflows in LangGraph

Section 14 of the course, lessons 109–115. Each lesson is one file. Read the docstring at the top, then run the file. The printout shows which node is running.

Shared model setup lives in `config.py`. It loads the project `.env` and picks Groq, Google, or OpenAI from whichever key is set.

## Study order

| Lesson | File | What you are learning |
| --- | --- | --- |
| 109 | `109_prompt_chaining.py` | Split one job into a fixed sequence of small prompts, with a Python `if` as a gate. No graph yet. |
| 110 | `110_prompt_chaining_langgraph.py` | The same sequence as a `StateGraph`: state, edges, and `add_conditional_edges`. |
| 111 | `111_parallelization.py` | Fan-out and fan-in. Several reviewers run together. `operator.add` collects their writes. |
| 112 | `112_routing.py` | A classifier picks one specialist. The other branches do not run. |
| 113 | `113_orchestrator_worker.py` | The steps are not known in advance. An orchestrator writes a plan, then a loop runs it. |
| 114 | `114_orchestrator_worker_langgraph.py` | The same plan, but each subtask is a parallel graph task created with `Send`. |
| 115 | `115_evaluator_optimizer.py` | A cycle. The generator drafts, the evaluator scores, and a low score sends feedback back. |

Lessons 109 and 113 are the ideas. Lessons 110 and 114 are those ideas built with LangGraph.

## Which pattern to pick

| Pattern | Shape | Pick it when |
| --- | --- | --- |
| Prompt chaining | A then B then C | The steps are always the same, and each step should see a clean input. |
| Parallelization | A, B, and C together, then join | The branches do not need each other's output. |
| Routing | One of A, B, or C | The input belongs to a single specialist. |
| Orchestrator-worker | Plan, then N workers, then join | You cannot name the steps until you have read the request. |
| Evaluator-optimizer | Draft, score, maybe draft again | A critic pass will catch mistakes a single generation misses. |

## How to run

From the project root, with the virtual environment active:

```powershell
python "12_Different Workflows In LangGraph/109_prompt_chaining.py"
python "12_Different Workflows In LangGraph/110_prompt_chaining_langgraph.py"
python "12_Different Workflows In LangGraph/111_parallelization.py"
python "12_Different Workflows In LangGraph/112_routing.py"
python "12_Different Workflows In LangGraph/113_orchestrator_worker.py"
python "12_Different Workflows In LangGraph/114_orchestrator_worker_langgraph.py"
python "12_Different Workflows In LangGraph/115_evaluator_optimizer.py"
```

## What to look at in the code

- **State.** Each graph has a `TypedDict`. A node returns only the keys it changes.
- **Edges.** `add_edge` is a fixed next step. `add_conditional_edges` asks a function for the next step.
- **Reducers.** Parallel nodes share a list field marked `Annotated[list, operator.add]`. Without that, the last write would replace the others.
- **Send.** Lesson 114 returns `Send("worker", payload)` once per planned task. The worker node then runs once per payload.
- **A cycle needs a stop.** Lesson 115 stops when the score is at least 8, or when `iteration` reaches `max_iterations`.
