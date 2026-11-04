# 3_LangChainUsecase: Complete LangChain Engineering Guide

This folder contains runnable implementations and end-to-end demonstrations of the **10 core LangChain architectural use cases**.

Every example follows modern LangChain conventions (v0.3 / v1.x with LCEL and LangGraph), connects automatically to your preferred LLM provider (`GROQ_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY` defined in `.env`), and includes comprehensive docstrings and reasoning traces.

---

## Directory Overview

| # | File | Use Case | Key LangChain Primitives |
|---|---|---|---|
| **01** | `1_prompt_management.py` | **Prompt Management** | `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShotChatMessagePromptTemplate` |
| **02** | `2_structured_output.py` | **Structured Output** | `llm.with_structured_output()`, Pydantic v2 `BaseModel`, `TypedDict`, `PydanticOutputParser` |
| **03** | `3_memory.py` | **Conversational Memory** | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory`, `trim_messages` |
| **04** | `4_tool_calling.py` | **Tool Calling** | `@tool` decorator, `llm.bind_tools()`, `AIMessage.tool_calls`, `ToolMessage` |
| **05** | `5_agent.py` | **Agents (ReAct Loop)** | `create_agent()`, autonomous reasoning, dynamic tool invocation, multi-step problem solving |
| **06** | `6_chain.py` | **Chains & LCEL** | Pipe operator `\|`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch`, `.stream()` |
| **07** | `7_document_processing.py` | **Document Processing & RAG** | `Document`, `RecursiveCharacterTextSplitter`, `InMemoryVectorStore`, embeddings, grounded retrieval |
| **08** | `8_model_switching.py` | **Model Switching & Failover** | `llm.with_fallbacks()`, `configurable_alternatives`, runtime provider swap, complexity router |
| **09** | `9_multi_agent_system.py` | **Multi-Agent Systems** | LangGraph `StateGraph`, Researcher $\to$ Writer $\to$ Quality Critic collaborative pipeline |
| **10** | `10_evaluation.py` | **LLM Evaluation** | LLM-as-a-Judge, factual accuracy, faithfulness, relevance rubrics, scorecard generation |

---

## Quick Start: How to Run

### 1. Run Any Example Directly
```powershell
python 3_LangChainUsecase/1_prompt_management.py
python 3_LangChainUsecase/2_structured_output.py
python 3_LangChainUsecase/3_memory.py
python 3_LangChainUsecase/4_tool_calling.py
python 3_LangChainUsecase/5_agent.py
python 3_LangChainUsecase/6_chain.py
python 3_LangChainUsecase/7_document_processing.py
python 3_LangChainUsecase/8_model_switching.py
python 3_LangChainUsecase/9_multi_agent_system.py
python 3_LangChainUsecase/10_evaluation.py
```

### 2. Run via the Interactive Menu
```powershell
python 3_LangChainUsecase/run_all.py
```
Or pass an argument directly:
```powershell
python 3_LangChainUsecase/run_all.py 5       # Runs Use Case 5 (Agent)
python 3_LangChainUsecase/run_all.py all     # Runs all 10 use cases
```

---

## Architectural Breakdown: How Each Concept Works

### 1. Prompt Management
- **Why it matters:** Instead of brittle string concatenations (`"Hello " + name`), LangChain separates prompt logic, variable validation, and formatting.
- **Chat Prompts:** Models expect structured lists of role-tagged messages (`SystemMessage`, `HumanMessage`, `AIMessage`).
- **MessagesPlaceholder:** Reserves a slot in the prompt where past message history can be injected dynamically at execution time.
- **Few-Shot Prompting:** Injects formatted demonstrations directly into the prompt context to guide difficult formatting or classification tasks.

### 2. Structured Output
- **Why it matters:** Downstream systems, APIs, and databases require strict types (ints, booleans, nested lists), not markdown prose.
- **Mechanism:** `llm.with_structured_output(Schema)` passes JSON Schema directly into model function-calling or schema-enforcement parameters, guaranteeing outputs that match Pydantic or TypedDict definitions.

### 3. Conversational Memory
- **Why it matters:** LLMs are stateless HTTP APIs; without memory, each request forgets previous interactions.
- **Modern LCEL Approach:** `RunnableWithMessageHistory` wraps an LCEL chain and automatically pulls past messages for a given `session_id`, feeds them into the prompt's `MessagesPlaceholder`, and appends the new exchange to the history store.
- **Context Window Management:** `trim_messages` cuts off older dialogue while preserving the critical system prompt to stay within model token constraints.

### 4. Tool Calling
- **Under the hood:** The LLM does *not* execute code. Instead:
  1. The developer binds tool schemas (`@tool`).
  2. The LLM determines a tool is needed and returns a structured `tool_calls` JSON payload.
  3. The local client executes the function in Python.
  4. The result is wrapped into a `ToolMessage` and returned to the model for final synthesis.

### 5. Agents
- **Chain vs Agent:** A chain executes a hardcoded sequence ($A \to B \to C$). An Agent uses the LLM as a decision engine to repeatedly observe, reason, choose tools, and terminate when satisfied.
- **ReAct Loop:** *Reason* (Thought) $\to$ *Act* (Tool call) $\to$ *Observe* (Tool response) $\to$ *Conclude* (Final answer).

### 6. Chains & LCEL (LangChain Expression Language)
- **Declarative Composition:** Components are connected using the unix-style pipe operator `|`:
  `Chain = Prompt | LLM | StrOutputParser`
- **Parallelism:** `RunnableParallel` executes independent branches concurrently.
- **Custom Functions:** `RunnableLambda` injects pure Python functions into the data stream.
- **Branching:** `RunnableBranch` conditionally routes inputs based on predicate functions.

### 7. Document Processing & RAG
- **Pipeline:** Documents $\to$ Chunking (`RecursiveCharacterTextSplitter`) $\to$ Embeddings $\to$ Vector Store Indexing $\to$ Semantic Retrieval $\to$ Grounded LLM Generation.
- **Context Preservation:** `chunk_overlap` ensures sentences cut at chunk boundaries maintain semantic coherence.

### 8. Model Switching
- **Resilience:** `.with_fallbacks([backup])` switches to a secondary provider when rate limits (HTTP 429) or downtime occur.
- **Cost / Latency Routing:** Cheap, ultra-fast models handle high-volume classification, while heavyweight frontier models handle complex reasoning.
- **Dynamic Config:** `configurable_alternatives` allows runtime selection of models via request configurations.

### 9. Multi-Agent Systems
- **Decomposition:** Breaks complex tasks into specialized personas:
  - **Researcher:** Unearths facts and technical specifications.
  - **Writer:** Synthesizes clear, well-structured drafts.
  - **Critic:** Audits compliance, quality, and tone.
- **StateGraph:** Nodes represent agents and edges represent handoffs, mutating a shared typed state dictionary.

### 10. Evaluation
- **LLM-as-a-Judge:** Uses an evaluator LLM with strict rubrics to assess candidate answers against ground truth.
- **Core Dimensions:**
  - *Factual Accuracy:* Does it match true facts?
  - *Faithfulness:* Is it free of hallucination?
  - *Relevance:* Does it answer the user's specific prompt directly?
- **Quantitative Scorecards:** Produces structured scores and pass/fail verdicts to automate regression testing across model versions.

