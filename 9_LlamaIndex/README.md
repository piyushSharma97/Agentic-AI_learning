# 🦙 LlamaIndex Learning Module

A structured, beginner-to-advanced hands-on guide for mastering **LlamaIndex** (v0.10+) for Retrieval-Augmented Generation (RAG) and Agentic Data Workflows.

---

## 📦 Prerequisites & Installation

To run the examples in this folder, install the required packages into your virtual environment:

```bash
# Core LlamaIndex
pip install llama-index

# Providers (install based on your API keys)
pip install llama-index-llms-openai llama-index-embeddings-openai
pip install llama-index-llms-groq
```

Ensure your root `.env` file contains your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🗂️ Module Curriculum

| File | Concept | What You Will Learn |
| :--- | :--- | :--- |
| **[`01_quickstart_rag.py`](./01_quickstart_rag.py)** | **RAG Fundamentals** | • 5 stages of RAG (Loading, Chunking, Indexing, Retrieval, Synthesis)<br>• In-memory `Document` creation<br>• `SimpleDirectoryReader` for files<br>• Inspecting retrieved source chunks and similarity scores |
| **[`02_documents_and_nodes.py`](./02_documents_and_nodes.py)** | **Data Architecture** | • `Document` vs. `Node` distinction<br>• `SentenceSplitter` chunking strategy<br>• Managing `chunk_size` and `chunk_overlap`<br>• Parent-child node relationships and metadata filtering |
| **[`03_settings_and_models.py`](./03_settings_and_models.py)** | **Modern Configuration** | • Modern `Settings` global object (replacing legacy `ServiceContext`)<br>• Swapping LLMs (OpenAI, Groq, Ollama)<br>• Swapping embedding models (OpenAI, HuggingFace)<br>• Setting global chunking parameters |
| **[`04_retrieval_and_chat_engine.py`](./04_retrieval_and_chat_engine.py)** | **Retrievers & Conversational RAG** | • Pure `Retriever` (semantic search without LLM generation)<br>• `QueryEngine` (one-shot Q&A)<br>• `ChatEngine` (multi-turn conversation with memory and query condensation) |
| **[`05_agent_with_tools.py`](./05_agent_with_tools.py)** | **Agentic RAG** | • Wrapping RAG pipelines into `QueryEngineTool`<br>• Wrapping custom Python functions into `FunctionTool`<br>• Building a `ReActAgent` that reasons across documents and actions |

---

## 🧠 Mental Model: LangChain vs. LlamaIndex

```
┌────────────────────────────────────────────────────────┐
│                      YOUR APPLICATION                  │
├───────────────────────────┬────────────────────────────┤
│         LANGCHAIN         │         LLAMAININDEX       │
│  (Orchestration & Logic)  │  (Data & Retrieval Engine) │
├───────────────────────────┼────────────────────────────┤
│ • Multi-agent workflows   │ • Data connectors (100+)   │
│ • State machines (LangGraph)│ • Smart chunking & parsing │
│ • Custom prompt chaining  │ • High-precision indexing  │
│ • Broad ecosystem tools   │ • Advanced retrieval & RAG │
└───────────────────────────┴────────────────────────────┘
```

They are complementary: you can use **LlamaIndex** to index and query your internal documentation, and expose it as a tool inside a **LangGraph / LangChain** agent.

