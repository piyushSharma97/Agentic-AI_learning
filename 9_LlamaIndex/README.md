# LlamaIndex

Five lessons. Each file is already one topic, so they stay side by side instead of moving into extra folders. `data/company_info.txt` is the handbook used by lessons 01, 04, and 05.

`config.py` loads the repo-root `.env` before LlamaIndex starts. Run a file from any working directory.

Every example inside a lesson is called from `main()`. Comment that call to skip it. `run_all.py` calls every lesson.

```powershell
python "9_LlamaIndex/01_quickstart_rag.py"
python "9_LlamaIndex/run_all.py"
```

## Lessons

| File | Topic | What runs |
| --- | --- | --- |
| `01_quickstart_rag.py` | Build an index and ask it a question | In-memory documents, then the handbook file |
| `02_documents_and_nodes.py` | Document versus node | Metadata, sentence splitting, an index built from nodes |
| `03_settings_and_models.py` | `Settings` | Defaults, then OpenAI, then Groq, each followed by one query |
| `04_retrieval_and_chat_engine.py` | Three ways to ask | Retriever, one-shot query engine, multi-turn chat engine |
| `05_agent_with_tools.py` | Agent with tools | Handbook search, then handbook plus the budget calculator |

## Review of each file

**01_quickstart_rag.py.** The five RAG stages are clear, and the two examples were already functions. Example 2 was commented out, so a normal run never read `data/`. Both examples now run. The directory path is taken from the file location, so it still finds `data/` when you launch the script from the repo root.

**02_documents_and_nodes.py.** Examples 1 and 2 do not call a model. They print a document and the chunks. Example 3 embeds those chunks and asks one question. It was commented out. It now runs with the others.

**03_settings_and_models.py.** `Settings` is the current way to set the global LLM, embedding model, and chunk size. The Groq example used `llama-3.3-70b-versatile`, which Groq no longer serves. It now uses `GROQ_MODEL` from `.env`, or `openai/gpt-oss-120b`. OpenAI is configured and queried first. Groq is then configured and queried, so you can see the chat model change while the embedding model stays OpenAI.

**04_retrieval_and_chat_engine.py.** One index is shared by three examples. The retriever does not write an answer. The query engine answers one question and forgets it, which is why "Where was she located?" has no prior turn. The chat engine keeps that turn. The query and chat examples were commented out. All three now run.

**05_agent_with_tools.py.** `ReActAgent.from_tools()` and `agent.chat()` are the old API. This install builds the agent with `ReActAgent(tools=..., llm=...)` and runs it with `agent.run(user_msg=...)`. The two tasks are separate functions, and `main()` calls both. The calculator description tells the model when a team-size question should use the function instead of only the handbook.

## What I did not split further

A folder per file would only add a path change. The topic boundary is already the file. Shared handbook text stays in `data/`.
