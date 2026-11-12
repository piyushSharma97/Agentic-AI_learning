# Agentic AI

Course notes for LangChain, LangGraph, and LlamaIndex. Each numbered folder is one part of the course. `langchain-imports.md` is the import index: the list at the top links to each import in the file.

Keys live in the root `.env`. That file is gitignored.

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
python "9_LlamaIndex/01_quickstart_rag.py"
```

Status is a study marker for the folder as it sits today: **understood**, **copied**, **to review**, or **skipped for now**.

| Folder | What it covers | Status |
| --- | --- | --- |
| `1_Python` | Python from variables and control flow through files, classes, logging, SQLite, FastAPI, and Pydantic | to review |
| `2_Langchain` | Document loaders, text splitters, embeddings, and Chroma and FAISS stores | copied |
| `3_LangChainUsecase` | Ten runnable use cases: prompts, structured output, memory, tools, agents, LCEL, RAG, model switching, multi-agent, and evaluation | understood |
| `4_GenAI` | Small OpenAI and Ollama apps, including a web-page RAG demo | copied |
| `5_LCEL` | One prompt, model, and parser chain written with LCEL | copied |
| `6_Building AI agents With Conver` | A chatbot built in seven steps: one message, history, prompts, language input, and trimming | understood |
| `7_Langchain Version V1` | LangChain v1 models, tools, messages, structured output, `create_agent`, and middleware | to review |
| `8_LangGraph-Intro` | A first `StateGraph`, then a chatbot whose messages use `add_messages` | copied |
| `9_LlamaIndex` | Five lessons: build an index, documents versus nodes, `Settings`, retrieval and chat, and an agent with tools | understood |
| `10_LangGrapph_Component` | LangGraph state schemas, reducers, chains, ReAct, checkpointers, search tools, and streaming | understood |
| `11_debugging_langGraph` | One tool-calling graph and a LangGraph Studio `langgraph.json` | to review |
| `12_Different Workflows In LangGraph` | Prompt chaining, parallelization, routing, orchestrator-worker, and evaluator-optimizer | understood |
| `13_Human_in_the_loop` | Pause a run, approve or edit a tool call, change state, replay a checkpoint, and interrupt from inside a node | understood |
| `14_RAG With LangGraph` | Agentic, corrective, and adaptive RAG over a local note corpus | to review |
| `extra` | `speech.txt`, sample text for the earlier loader lessons | skipped for now |
| `src` | Package stub. `main()` prints one hello line | skipped for now |
