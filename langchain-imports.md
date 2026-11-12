# LangChain and LangGraph imports used in this project

Every LangChain and LangGraph import used across the course folders, with what it is for and a small example.

Folders scanned:

- `2_Langchain`
- `3_LangChainUsecase`
- `4_GenAI`
- `5_LCEL`
- `6_Building AI agents With Conver`
- `7_Langchain Version V1`
- `8_LangGraph-Intro`
- `10_LangGrapph_Component`
- `11_debugging_langGraph`
- `12_Different Workflows In LangGraph`
- `13_Human_in_the_loop`
- `14_RAG With LangGraph`
- `dummy.py`

---

<a id="contents"></a>

## Contents

Click an import to jump to that section.

### langchain

- [`langchain`](#langchain) — Check the installed version
- [`langchain.agents.create_agent`](#langchain-agents-create-agent) — Build a v1 tool-calling agent
- [`langchain.agents.middleware.SummarizationMiddleware`](#langchain-agents-middleware-summarizationmiddleware) — Summarize long chats
- [`langchain.agents.middleware.HumanInTheLoopMiddleware`](#langchain-agents-middleware-humanintheloopmiddleware) — Pause a tool call for approval
- [`langchain.agents.structured_output.ToolStrategy`](#langchain-agents-structured-output-toolstrategy) — Agent replies as a schema
- [`langchain.chat_models.init_chat_model`](#langchain-chat-models-init-chat-model) — One call for any provider
- [`langchain.tools.tool`](#langchain-tools-tool) — Turn a function into a tool (v1 package)
- [`langchain.chains.create_retrieval_chain`](#langchain-chains-create-retrieval-chain) — RAG: retrieve, then answer
- [`langchain.chains.combine_documents.create_stuff_documents_chain`](#langchain-chains-combine-documents-create-stuff-documents-chain) — Put retrieved chunks into one prompt

### langchain_core

- [`langchain_core.messages.HumanMessage`](#langchain-core-messages-humanmessage) — User turn
- [`langchain_core.messages.SystemMessage`](#langchain-core-messages-systemmessage) — Role and rules
- [`langchain_core.messages.AIMessage`](#langchain-core-messages-aimessage) — Assistant turn
- [`langchain_core.messages.ToolMessage`](#langchain-core-messages-toolmessage) — Tool result sent back to the model
- [`langchain_core.messages.AnyMessage`](#langchain-core-messages-anymessage) — Type for any chat message
- [`langchain_core.messages.trim_messages`](#langchain-core-messages-trim-messages) — Cut history to a token budget
- [`langchain_core.prompts.ChatPromptTemplate`](#langchain-core-prompts-chatprompttemplate) — Prompt with placeholders
- [`langchain_core.prompts.MessagesPlaceholder`](#langchain-core-prompts-messagesplaceholder) — Slot for chat history
- [`langchain_core.output_parsers.StrOutputParser`](#langchain-core-output-parsers-stroutputparser) — Message to a plain string
- [`langchain_core.output_parsers.PydanticOutputParser`](#langchain-core-output-parsers-pydanticoutputparser) — Parse model text into a Pydantic object
- [`langchain_core.chat_history.BaseChatMessageHistory`](#langchain-core-chat-history-basechatmessagehistory) — History interface
- [`langchain_core.chat_history.InMemoryChatMessageHistory`](#langchain-core-chat-history-inmemorychatmessagehistory) — RAM chat history for one session
- [`langchain_core.runnables.RunnablePassthrough`](#langchain-core-runnables-runnablepassthrough) — Pass or add keys in an LCEL chain
- [`langchain_core.runnables.RunnableLambda`](#langchain-core-runnables-runnablelambda) — Put a Python function in an LCEL chain
- [`langchain_core.runnables.history.RunnableWithMessageHistory`](#langchain-core-runnables-history-runnablewithmessagehistory) — Load and save history around a chain
- [`langchain_core.documents.Document`](#langchain-core-documents-document) — Text plus metadata for loaders and search
- [`langchain_core.vectorstores.InMemoryVectorStore`](#langchain-core-vectorstores-inmemoryvectorstore) — Vector store that lives in process memory
- [`langchain_core.tools.tool`](#langchain-core-tools-tool) — Turn a function into a tool (core package)
- [`langchain_core.tools.StructuredTool`](#langchain-core-tools-structuredtool) — Build a tool from an existing function

### langchain_community

- [`langchain_community.document_loaders.TextLoader`](#langchain-community-document-loaders-textloader) — Load a text file
- [`langchain_community.document_loaders.PyPDFLoader`](#langchain-community-document-loaders-pypdfloader) — Load a PDF
- [`langchain_community.document_loaders.WebBaseLoader`](#langchain-community-document-loaders-webbaseloader) — Load a web page
- [`langchain_community.vectorstores.Chroma`](#langchain-community-vectorstores-chroma) — Older Chroma vector store import
- [`langchain_community.vectorstores.FAISS`](#langchain-community-vectorstores-faiss) — Local FAISS index
- [`langchain_community.chat_message_histories.ChatMessageHistory`](#langchain-community-chat-message-histories-chatmessagehistory) — In-memory session history

### langchain_text_splitters

- [`langchain_text_splitters.CharacterTextSplitter`](#langchain-text-splitters-charactertextsplitter) — Split on one separator
- [`langchain_text_splitters.RecursiveCharacterTextSplitter`](#langchain-text-splitters-recursivecharactertextsplitter) — Split by paragraphs, then lines, then words

### langchain_openai

- [`langchain_openai.ChatOpenAI`](#langchain-openai-chatopenai) — OpenAI chat model
- [`langchain_openai.OpenAIEmbeddings`](#langchain-openai-openaiembeddings) — OpenAI embeddings

### langchain_google_genai

- [`langchain_google_genai.ChatGoogleGenerativeAI`](#langchain-google-genai-chatgooglegenerativeai) — Gemini chat model

### langchain_groq

- [`langchain_groq.ChatGroq`](#langchain-groq-chatgroq) — Groq chat model

### langchain_ollama

- [`langchain_ollama.ChatOllama`](#langchain-ollama-chatollama) — Local Ollama chat model
- [`langchain_ollama.OllamaEmbeddings`](#langchain-ollama-ollamaembeddings) — Local Ollama embeddings

### langchain_huggingface

- [`langchain_huggingface.HuggingFaceEmbeddings`](#langchain-huggingface-huggingfaceembeddings) — Local Hugging Face embeddings

### langchain_chroma

- [`langchain_chroma.Chroma`](#langchain-chroma-chroma) — Current Chroma package

### langgraph

- [`langgraph.graph.StateGraph`](#langgraph-graph-stategraph) — Build a graph of nodes, edges, and state
- [`langgraph.graph.START`](#langgraph-graph-start) — Edge target for the first node
- [`langgraph.graph.END`](#langgraph-graph-end) — Edge target that finishes the run
- [`langgraph.graph.MessagesState`](#langgraph-graph-messagesstate) — Built-in state with a messages list
- [`langgraph.graph.message.add_messages`](#langgraph-graph-message-add-messages) — Append chat messages instead of overwriting them
- [`langgraph.prebuilt.ToolNode`](#langgraph-prebuilt-toolnode) — Run tool calls and write ToolMessages
- [`langgraph.prebuilt.tools_condition`](#langgraph-prebuilt-tools-condition) — Route to tools or END from the last message
- [`langgraph.checkpoint.memory.MemorySaver`](#langgraph-checkpoint-memory-memorysaver) — Save graph state in RAM by thread id
- [`langgraph.types.Send`](#langgraph-types-send) — Fan out one node copy per item
- [`langgraph.types.interrupt`](#langgraph-types-interrupt) — Pause inside a node until a person answers
- [`langgraph.types.Command`](#langgraph-types-command) — Resume a paused graph with a value
- [`langgraph.checkpoint.memory.InMemorySaver`](#langgraph-checkpoint-memory-inmemorysaver) — Checkpointer used by create_agent

<a id="langchain"></a>

## `langchain`

**Import**

```python
import langchain
```

**Used in:** `7_Langchain Version V1/1-langchainintro.py`, `7_Langchain Version V1/2-modelintegration.py`

**Use case:** Import the package itself so you can check that LangChain is installed and print its version.

**Example**

```python
import langchain

print(langchain.__version__)
```

---

<a id="langchain-agents-create-agent"></a>

## `langchain.agents.create_agent`

**Import**

```python
from langchain.agents import create_agent
```

**Used in:** `7_Langchain Version V1/1-langchainintro.py`, `7_Langchain Version V1/create_agent.py`, `7_Langchain Version V1/5-structuredoutput.py`, `7_Langchain Version V1/6-middleWare.py`

**Use case:** Build a LangChain v1 agent that can call tools, follow a system prompt, and return a message history. This is the v1 replacement for the older agent constructors.

**Example**

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[add],
    system_prompt="You are a helpful math assistant.",
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 25 + 35?"}]
})
print(response["messages"][-1].content)
```

---

<a id="langchain-agents-middleware-summarizationmiddleware"></a>

## `langchain.agents.middleware.SummarizationMiddleware`

**Import**

```python
from langchain.agents.middleware import SummarizationMiddleware
```

**Used in:** `7_Langchain Version V1/6-middleWare.py`

**Use case:** Compress old chat turns when a conversation gets long, so the agent stays inside the model context window. Recent messages stay as-is; older ones are summarized.

**Example**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10),  # summarize after 10 messages
            keep=("messages", 4),       # keep the last 4 messages raw
        )
    ],
)
```

---

<a id="langchain-agents-middleware-humanintheloopmiddleware"></a>

## `langchain.agents.middleware.HumanInTheLoopMiddleware`

**Import**

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
```

**Used in:** `7_Langchain Version V1/6-middleWare.py`

**Use case:** Pause a v1 agent before a chosen tool runs, so a person can approve, reject, or edit the call. Tools set to `False` in `interrupt_on` still run on their own. This needs a checkpointer and the same `thread_id` when you resume with `Command`.

**Example**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-4o-mini",
    tools=[send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
            }
        )
    ],
)

config = {"configurable": {"thread_id": "hitl-approve"}}
paused = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a hello email to ada@example.com"}]},
    config=config,
)
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
```

---

<a id="langchain-agents-structured-output-toolstrategy"></a>

## `langchain.agents.structured_output.ToolStrategy`

**Import**

```python
from langchain.agents.structured_output import ToolStrategy
```

**Used in:** `7_Langchain Version V1/5-structuredoutput.py`

**Use case:** Force an agent to return data that matches a schema (Pydantic, TypedDict, or dataclass) by using the model's tool-calling feature. Useful when the next step of your app needs fields, not free text.

**Example**

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class TravelPlan(BaseModel):
    destination: str = Field(description="Recommended destination")
    duration_days: int = Field(description="Number of days", gt=0)

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[],
    response_format=ToolStrategy(TravelPlan),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Plan a 5 day trip to Japan."}]
})
print(result["structured_response"])
```

---

<a id="langchain-chat-models-init-chat-model"></a>

## `langchain.chat_models.init_chat_model`

**Import**

```python
from langchain.chat_models import init_chat_model
```

**Used in:** `7_Langchain Version V1/2-modelintegration.py`, `7_Langchain Version V1/3-tools.py`, `7_Langchain Version V1/4-messages.py`, `7_Langchain Version V1/5-structuredoutput.py`

**Use case:** Create a chat model from one string in `provider:model` form. You can switch Groq, OpenAI, Google, and others without changing the rest of the code.

**Example**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:openai/gpt-oss-120b")
response = model.invoke("Explain Artificial Intelligence in one sentence.")
print(response.content)
```

---

<a id="langchain-tools-tool"></a>

## `langchain.tools.tool`

**Import**

```python
from langchain.tools import tool
```

**Used in:** `7_Langchain Version V1/3-tools.py`, `7_Langchain Version V1/create_agent.py`

**Use case:** Turn a normal Python function into a tool the model can call. The function name, docstring, and type hints become the tool schema.

**Example**

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a city."""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])
```

---

<a id="langchain-chains-create-retrieval-chain"></a>

## `langchain.chains.create_retrieval_chain`

**Import**

```python
from langchain.chains import create_retrieval_chain
```

**Used in:** `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Wire a retriever to a document-answering chain. The chain searches a vector store, then answers using the retrieved chunks. This is the RAG (retrieval-augmented generation) pattern.

**Example**

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

result = rag_chain.invoke({"input": "What is this page about?"})
print(result["answer"])
```

---

<a id="langchain-chains-combine-documents-create-stuff-documents-chain"></a>

## `langchain.chains.combine_documents.create_stuff_documents_chain`

**Import**

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
```

**Used in:** `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Stuff all retrieved documents into one prompt and ask the model to answer from that context. "Stuff" means the chunks are placed together in the `{context}` variable.

**Example**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {input}"
)
document_chain = create_stuff_documents_chain(llm, prompt)
```

---

<a id="langchain-core-messages-humanmessage"></a>

## `langchain_core.messages.HumanMessage`

**Import**

```python
from langchain_core.messages import HumanMessage
```

**Used in:** `5_LCEL/simple.py`, `6_Building AI agents With Conver` (basic chat, manual conversation, session memory, prompt template, trim history, chatbot), `7_Langchain Version V1/3-tools.py`, `7_Langchain Version V1/4-messages.py`, `7_Langchain Version V1/6-middleWare.py`

**Use case:** Represent one user turn in a chat. Pass a list of messages when you need roles, history, or metadata such as `name` and `id`.

**Example**

```python
from langchain_core.messages import HumanMessage

response = model.invoke([
    HumanMessage(content="What is LangChain?", name="alice")
])
print(response.content)
```

---

<a id="langchain-core-messages-systemmessage"></a>

## `langchain_core.messages.SystemMessage`

**Import**

```python
from langchain_core.messages import SystemMessage
```

**Used in:** `5_LCEL/simple.py`, `6_Building AI agents With Conver/06_trim_history/main.py`, `6_Building AI agents With Conver/chatbot.py`, `7_Langchain Version V1/4-messages.py`, `7_Langchain Version V1/6-middleWare.py`

**Use case:** Set the model's role and rules before the user speaks. The model treats this as instructions, not as a user question.

**Example**

```python
from langchain_core.messages import SystemMessage, HumanMessage

response = model.invoke([
    SystemMessage(content="You are a senior Python developer. Give short code examples."),
    HumanMessage(content="How do I create a REST API?"),
])
print(response.content)
```

---

<a id="langchain-core-messages-aimessage"></a>

## `langchain_core.messages.AIMessage`

**Import**

```python
from langchain_core.messages import AIMessage
```

**Used in:** `6_Building AI agents With Conver/02_manual_conversation/main.py`, `6_Building AI agents With Conver/06_trim_history/main.py`, `6_Building AI agents With Conver/chatbot.py`, `7_Langchain Version V1/4-messages.py`

**Use case:** Represent an assistant turn. Models return this automatically. You also build one yourself when you want to replay past answers or show a tool call.

**Example**

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Can you help me?"),
    AIMessage(content="Yes, what do you need?"),
    HumanMessage(content="What is 2 + 2?"),
]
response = model.invoke(messages)
```

---

<a id="langchain-core-messages-toolmessage"></a>

## `langchain_core.messages.ToolMessage`

**Import**

```python
from langchain_core.messages import ToolMessage
```

**Used in:** `7_Langchain Version V1/4-messages.py`

**Use case:** Send a tool's result back to the model. `tool_call_id` must match the id on the model's tool call so the model knows which request this result belongs to.

**Example**

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

messages = [
    HumanMessage(content="What's the weather in San Francisco?"),
    AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather",
            "args": {"location": "San Francisco"},
            "id": "call_123",
        }],
    ),
    ToolMessage(
        content="Sunny, 72°F",
        tool_call_id="call_123",
        name="get_weather",
    ),
]
response = model.invoke(messages)
```

---

<a id="langchain-core-messages-anymessage"></a>

## `langchain_core.messages.AnyMessage`

**Import**

```python
from langchain_core.messages import AnyMessage
```

**Used in:** `8_LangGraph-Intro/chatbot.py`, `10_LangGrapph_Component/04_state_reducers/main.py`, `10_LangGrapph_Component/06_react_agent/main.py`, `10_LangGrapph_Component/12_simple_chatbot/main.py`

**Use case:** Type hint for "any chat message": human, AI, system, or tool. Pair it with `add_messages` when the state field must accept every message kind the graph will store.

**Example**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

---

<a id="langchain-core-messages-trim-messages"></a>

## `langchain_core.messages.trim_messages`

**Import**

```python
from langchain_core.messages import trim_messages
```

**Used in:** `6_Building AI agents With Conver/06_trim_history/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** Drop older messages before calling the model so the prompt stays under a token limit. You choose which end to trim and how many tokens to keep.

**Example**

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",          # keep the newest messages
    token_counter=model,
    include_system=True,      # always keep the system message
    start_on="human",
)
response = model.invoke(trimmed)
```

---

<a id="langchain-core-prompts-chatprompttemplate"></a>

## `langchain_core.prompts.ChatPromptTemplate`

**Import**

```python
from langchain_core.prompts import ChatPromptTemplate
```

**Used in:** `4_GenAI/01_openai/1-GettingStarted.py`, `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/app.py`, `4_GenAI/2_ollama/Intro.py`, `5_LCEL/simple.py`, `6_Building AI agents With Conver/04_prompt_template/main.py`, `6_Building AI agents With Conver/05_language_input/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`, `dummy.py`

**Use case:** Build a reusable chat prompt with placeholders such as `{topic}` or `{context}`. You fill the placeholders at runtime instead of concatenating strings.

**Example**

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in {language}."),
    ("human", "{question}"),
])

chain = prompt | model | parser
print(chain.invoke({"language": "Hindi", "question": "What is LCEL?"}))
```

---

<a id="langchain-core-prompts-messagesplaceholder"></a>

## `langchain_core.prompts.MessagesPlaceholder`

**Import**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

**Used in:** `6_Building AI agents With Conver/04_prompt_template/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** Insert a list of past chat messages into a prompt. The placeholder name (usually `history` or `messages`) is where memory gets injected.

**Example**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

---

<a id="langchain-core-output-parsers-stroutputparser"></a>

## `langchain_core.output_parsers.StrOutputParser`

**Import**

```python
from langchain_core.output_parsers import StrOutputParser
```

**Used in:** `4_GenAI/01_openai/1-GettingStarted.py`, `4_GenAI/2_ollama/app.py`, `5_LCEL/simple.py`, `dummy.py`

**Use case:** Take the model's `AIMessage` and return only the text string. This is the last step in most LCEL chains (`prompt | model | parser`).

**Example**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Explain {topic} in one line.")
chain = prompt | model | StrOutputParser()
print(chain.invoke({"topic": "embeddings"}))
```

---

<a id="langchain-core-output-parsers-pydanticoutputparser"></a>

## `langchain_core.output_parsers.PydanticOutputParser`

**Import**

```python
from langchain_core.output_parsers import PydanticOutputParser
```

**Used in:** `3_LangChainUsecase/2_structured_output.py`

**Use case:** Ask the model for text that matches a Pydantic model, then parse that text into the model. `get_format_instructions()` is the block you paste into the prompt. Newer code often uses `model.with_structured_output(...)` instead, which is what the same lesson recommends first.

**Example**

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class ActionItem(BaseModel):
    task: str = Field(description="What to do")
    owner: str = Field(description="Who does it")

parser = PydanticOutputParser(pydantic_object=ActionItem)
prompt = ChatPromptTemplate.from_template(
    "Extract an action item.\n{format_instructions}\n\nNotes:\n{notes}"
)
chain = prompt | llm | parser
item = chain.invoke({
    "notes": "Priya will send the recap by Friday.",
    "format_instructions": parser.get_format_instructions(),
})
```

---

<a id="langchain-core-chat-history-basechatmessagehistory"></a>

## `langchain_core.chat_history.BaseChatMessageHistory`

**Import**

```python
from langchain_core.chat_history import BaseChatMessageHistory
```

**Used in:** `6_Building AI agents With Conver/03_session_memory/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** Type hint for any chat-history store. Your `get_session_history` function returns this type so `RunnableWithMessageHistory` can save and load messages per session.

**Example**

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

---

<a id="langchain-core-chat-history-inmemorychatmessagehistory"></a>

## `langchain_core.chat_history.InMemoryChatMessageHistory`

**Import**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
```

**Used in:** `3_LangChainUsecase/3_memory.py`

**Use case:** A ready-made chat history that lives in RAM and already implements `BaseChatMessageHistory`. Use one instance per session id, then hand it to `RunnableWithMessageHistory`. The history is gone when the process stops.

**Example**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

---

<a id="langchain-core-runnables-runnablepassthrough"></a>

## `langchain_core.runnables.RunnablePassthrough`

**Import**

```python
from langchain_core.runnables import RunnablePassthrough
```

**Used in:** `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** Pass a value through an LCEL chain unchanged, or assign extra keys while keeping the original input. In this project it carries the trimmed history into the prompt.

**Example**

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(history=lambda x: trim_history(x["history"]))
    | prompt
    | model
)
```

---

<a id="langchain-core-runnables-runnablelambda"></a>

## `langchain_core.runnables.RunnableLambda`

**Import**

```python
from langchain_core.runnables import RunnableLambda
```

**Used in:** `3_LangChainUsecase/6_chain.py`, `3_LangChainUsecase/8_model_switching.py`

**Use case:** Wrap a normal Python function so it can sit in an LCEL chain with `|`. The same lesson also imports `RunnableParallel` (run several branches on one input) and `RunnableBranch` (pick a branch with a condition).

**Example**

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

def word_count(text: str) -> int:
    return len(text.split())

chain = RunnableParallel(
    answer=prompt | llm | parser,
    words=parser | RunnableLambda(word_count),
)
```

---

<a id="langchain-core-runnables-history-runnablewithmessagehistory"></a>

## `langchain_core.runnables.history.RunnableWithMessageHistory`

**Import**

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
```

**Used in:** `6_Building AI agents With Conver/03_session_memory/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** Wrap a chain so it loads chat history before the call and saves the new user and assistant messages after the call. Each user gets a `session_id`.

**Example**

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "My name is Piyush."},
    config={"configurable": {"session_id": "user-1"}},
)
```

---

<a id="langchain-core-documents-document"></a>

## `langchain_core.documents.Document`

**Import**

```python
from langchain_core.documents import Document
```

**Used in:** `3_LangChainUsecase/7_document_processing.py`, `14_RAG With LangGraph/corpus.py`, `14_RAG With LangGraph/2-CorrectiveRAG.py`, `14_RAG With LangGraph/3-AdaptiveRAG.py`

**Use case:** One unit of text plus metadata. Loaders, splitters, and vector stores all pass `Document` objects around. `page_content` is the text. `metadata` holds the source name, page, or anything you want to show with a retrieved chunk.

**Example**

```python
from langchain_core.documents import Document

note = Document(
    page_content="LangGraph saves state after each super-step.",
    metadata={"source": "langgraph-checkpointer"},
)
```

---

<a id="langchain-core-vectorstores-inmemoryvectorstore"></a>

## `langchain_core.vectorstores.InMemoryVectorStore`

**Import**

```python
from langchain_core.vectorstores import InMemoryVectorStore
```

**Used in:** `3_LangChainUsecase/7_document_processing.py`, `14_RAG With LangGraph/corpus.py`, `14_RAG With LangGraph/3-AdaptiveRAG.py`

**Use case:** A vector store that keeps embeddings in process memory. It has the same `add_documents` and `as_retriever` methods as FAISS or Chroma, so a RAG graph can be written before you pick a database. The index disappears when the process stops.

**Example**

```python
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(embedding=embeddings)
store.add_documents(chunks)
retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How does a checkpointer work?")
```

---

<a id="langchain-core-tools-tool"></a>

## `langchain_core.tools.tool`

**Import**

```python
from langchain_core.tools import tool
```

**Used in:** `3_LangChainUsecase/4_tool_calling.py`, `3_LangChainUsecase/5_agent.py`, `10_LangGrapph_Component/06_react_agent/main.py`, `10_LangGrapph_Component/09_chain_with_tools/main.py`, `11_debugging_langGraph/openAi.py`, `13_Human_in_the_loop/config.py`

**Use case:** Same job as `langchain.tools.tool`: turn a typed Python function into a tool schema. The core import is the one the LangGraph lessons use. The v1 package re-exports it for `create_agent`.

**Example**

```python
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers a and b."""
    return a + b
```

---

<a id="langchain-core-tools-structuredtool"></a>

## `langchain_core.tools.StructuredTool`

**Import**

```python
from langchain_core.tools import StructuredTool
```

**Used in:** `14_RAG With LangGraph/corpus.py`

**Use case:** Build a tool from a function you already wrote, and set the name and description yourself. Useful when the function is created inside another function, so the `@tool` decorator is awkward.

**Example**

```python
from langchain_core.tools import StructuredTool

def search(query: str) -> str:
    return retriever.invoke(query)[0].page_content

search_notes = StructuredTool.from_function(
    func=search,
    name="search_notes",
    description="Search the local LangGraph notes.",
)
```

---

<a id="langchain-community-document-loaders-textloader"></a>

## `langchain_community.document_loaders.TextLoader`

**Import**

```python
from langchain_community.document_loaders import TextLoader
```

**Used in:** `2_Langchain/03_DataIngestion/01_DataIngestion.py`, `2_Langchain/04_Data Trnasformer/CharacterTextsplitter.py`, `2_Langchain/5_Embedding/embedding.py`, `2_Langchain/5_Embedding/huggingface.py`, `2_Langchain/6_VectorStore/chromaDB.py`, `2_Langchain/6_VectorStore/Faiss.py`, `dummy.py`

**Use case:** Read a `.txt` file into LangChain `Document` objects (page content plus metadata). This is the first step before splitting and embedding.

**Example**

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/notes.txt", encoding="utf-8")
docs = loader.load()
print(docs[0].page_content[:200])
```

---

<a id="langchain-community-document-loaders-pypdfloader"></a>

## `langchain_community.document_loaders.PyPDFLoader`

**Import**

```python
from langchain_community.document_loaders import PyPDFLoader
```

**Used in:** `2_Langchain/03_DataIngestion/01_DataIngestion.py`, `2_Langchain/04_Data Trnasformer/RecuriveCharactertextsplitte.py`

**Use case:** Load a PDF. Each page becomes one `Document`, with the page number stored in metadata.

**Example**

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
pages = loader.load()
print(pages[0].metadata)
print(pages[0].page_content[:200])
```

---

<a id="langchain-community-document-loaders-webbaseloader"></a>

## `langchain_community.document_loaders.WebBaseLoader`

**Import**

```python
from langchain_community.document_loaders import WebBaseLoader
```

**Used in:** `2_Langchain/03_DataIngestion/01_DataIngestion.py`, `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Download a web page and turn the visible text into documents. Often paired with BeautifulSoup (`bs4`) to keep only the tags you care about.

**Example**

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.langchain.com/")
docs = loader.load()
print(docs[0].page_content[:300])
```

---

<a id="langchain-community-vectorstores-chroma"></a>

## `langchain_community.vectorstores.Chroma`

**Import**

```python
from langchain_community.vectorstores import Chroma
```

**Used in:** `2_Langchain/5_Embedding/embedding.py`, `2_Langchain/5_Embedding/huggingface.py`, `dummy.py`

**Use case:** Store document embeddings and search them by similarity. The community import is the older Chroma integration used in the embedding notebooks. Newer code in this repo uses `langchain_chroma.Chroma` instead.

**Example**

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
results = db.similarity_search("What is an agent?", k=3)
```

---

<a id="langchain-community-vectorstores-faiss"></a>

## `langchain_community.vectorstores.FAISS`

**Import**

```python
from langchain_community.vectorstores import FAISS
```

**Used in:** `2_Langchain/6_VectorStore/Faiss.py`, `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Build an in-memory vector index with FAISS. Good for local demos and RAG apps that do not need a separate database server. You can save the index to disk and load it later.

**Example**

```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

loaded = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = loaded.as_retriever()
```

---

<a id="langchain-community-chat-message-histories-chatmessagehistory"></a>

## `langchain_community.chat_message_histories.ChatMessageHistory`

**Import**

```python
from langchain_community.chat_message_histories import ChatMessageHistory
```

**Used in:** `6_Building AI agents With Conver/03_session_memory/main.py`, `6_Building AI agents With Conver/07_complete_chatbot/main.py`, `6_Building AI agents With Conver/chatbot.py`

**Use case:** An in-memory list of chat messages for one session. It implements `BaseChatMessageHistory`, so it works with `RunnableWithMessageHistory`. Data is lost when the process stops.

**Example**

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hi, I am Piyush.")
history.add_ai_message("Hello Piyush.")
print(history.messages)
```

---

<a id="langchain-text-splitters-charactertextsplitter"></a>

## `langchain_text_splitters.CharacterTextSplitter`

**Import**

```python
from langchain_text_splitters import CharacterTextSplitter
```

**Used in:** `2_Langchain/04_Data Trnasformer/CharacterTextsplitter.py`

**Use case:** Split text on a single separator (often `"\n\n"` or `""`). Chunks have a fixed character size and an overlap so a sentence is not cut off from the next chunk.

**Example**

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(len(chunks))
```

---

<a id="langchain-text-splitters-recursivecharactertextsplitter"></a>

## `langchain_text_splitters.RecursiveCharacterTextSplitter`

**Import**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**Used in:** `2_Langchain/04_Data Trnasformer/RecuriveCharactertextsplitte.py`, `2_Langchain/5_Embedding/embedding.py`, `2_Langchain/5_Embedding/huggingface.py`, `2_Langchain/6_VectorStore/chromaDB.py`, `2_Langchain/6_VectorStore/Faiss.py`, `4_GenAI/01_openai/Simpleapp.py`, `4_GenAI/2_ollama/Intro.py`, `dummy.py`

**Use case:** Split text by trying larger separators first (`\n\n`, then `\n`, then space). This is the usual splitter for RAG because it keeps paragraphs together when it can.

**Example**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
```

---

<a id="langchain-openai-chatopenai"></a>

## `langchain_openai.ChatOpenAI`

**Import**

```python
from langchain_openai import ChatOpenAI
```

**Used in:** `4_GenAI/01_openai/1-GettingStarted.py`, `4_GenAI/01_openai/Simpleapp.py`, `7_Langchain Version V1/2-modelintegration.py`, `dummy.py`

**Use case:** Chat model client for the OpenAI API (and OpenAI-compatible endpoints). You pick a model name such as `gpt-4o-mini` and call `invoke`.

**Example**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(llm.invoke("Say hello in one sentence.").content)
```

---

<a id="langchain-openai-openaiembeddings"></a>

## `langchain_openai.OpenAIEmbeddings`

**Import**

```python
from langchain_openai import OpenAIEmbeddings
```

**Used in:** `2_Langchain/5_Embedding/embedding.py`, `4_GenAI/01_openai/Simpleapp.py`, `dummy.py`

**Use case:** Turn text into vectors with an OpenAI embedding model. Those vectors are what Chroma and FAISS search over.

**Example**

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("What is a vector store?")
print(len(vector))
```

---

<a id="langchain-google-genai-chatgooglegenerativeai"></a>

## `langchain_google_genai.ChatGoogleGenerativeAI`

**Import**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
```

**Used in:** `7_Langchain Version V1/2-modelintegration.py`

**Use case:** Chat with a Google Gemini model through LangChain. Needs `GOOGLE_API_KEY`.

**Example**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
print(llm.invoke("Explain embeddings in one sentence.").content)
```

---

<a id="langchain-groq-chatgroq"></a>

## `langchain_groq.ChatGroq`

**Import**

```python
from langchain_groq import ChatGroq
```

**Used in:** `5_LCEL/simple.py`, `6_Building AI agents With Conver` (all chatbot steps and `chatbot.py`), `7_Langchain Version V1/2-modelintegration.py`

**Use case:** Chat with a model hosted on Groq. This project uses it for the LCEL demo and the conversation chatbot because responses are fast. Needs `GROQ_API_KEY`.

**Example**

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print(llm.invoke("What is a prompt template?").content)
```

---

<a id="langchain-ollama-chatollama"></a>

## `langchain_ollama.ChatOllama`

**Import**

```python
from langchain_ollama import ChatOllama
```

**Used in:** `4_GenAI/2_ollama/app.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Chat with a model running locally in Ollama (for example `llama3.2`). No cloud API key is required once Ollama is running.

**Example**

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
print(llm.invoke("Explain RAG in one sentence.").content)
```

---

<a id="langchain-ollama-ollamaembeddings"></a>

## `langchain_ollama.OllamaEmbeddings`

**Import**

```python
from langchain_ollama import OllamaEmbeddings
```

**Used in:** `2_Langchain/5_Embedding/ollamaemnedding.py`, `2_Langchain/6_VectorStore/chromaDB.py`, `4_GenAI/2_ollama/Intro.py`

**Use case:** Create embeddings with a local Ollama embedding model (for example `nomic-embed-text`). Used when you want the vector store to stay on your machine.

**Example**

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("local embeddings")
print(len(vector))
```

---

<a id="langchain-huggingface-huggingfaceembeddings"></a>

## `langchain_huggingface.HuggingFaceEmbeddings`

**Import**

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

**Used in:** `2_Langchain/5_Embedding/huggingface.py`, `2_Langchain/6_VectorStore/Faiss.py`

**Use case:** Create embeddings with a Hugging Face sentence-transformer model running locally. No API key. The first run downloads the model.

**Example**

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector = embeddings.embed_query("What is FAISS?")
print(len(vector))
```

---

<a id="langchain-chroma-chroma"></a>

## `langchain_chroma.Chroma`

**Import**

```python
from langchain_chroma import Chroma
```

**Used in:** `2_Langchain/6_VectorStore/chromaDB.py`

**Use case:** The current first-party Chroma package. Same job as `langchain_community.vectorstores.Chroma`: save embeddings and run similarity search. Prefer this import in new code.

**Example**

```python
from langchain_chroma import Chroma

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_store",
)
print(db.similarity_search("LangChain vector store", k=2))
```

---

<a id="langgraph-graph-stategraph"></a>

## `langgraph.graph.StateGraph`

**Import**

```python
from langgraph.graph import StateGraph
```

**Used in:** `8_LangGraph-Intro/simplegraph.py`, `8_LangGraph-Intro/chatbot.py`, `10_LangGrapph_Component`, `12_Different Workflows In LangGraph`, `13_Human_in_the_loop`, `14_RAG With LangGraph`

**Use case:** The builder for a LangGraph workflow. You pass a state schema (`TypedDict`, `dataclass`, Pydantic model, or `MessagesState`), add nodes, add edges, then `compile()` to get a runnable graph. `11_debugging_langGraph/openAi.py` imports the same class from `langgraph.graph.state`.

**Example**

```python
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    graph_info: str

def start_play(state: State) -> dict:
    return {"graph_info": state["graph_info"] + " I am planning to play"}

builder = StateGraph(State)
builder.add_node("start_play", start_play)
builder.add_edge(START, "start_play")
builder.add_edge("start_play", END)
graph = builder.compile()
print(graph.invoke({"graph_info": "Start."}))
```

---

<a id="langgraph-graph-start"></a>

## `langgraph.graph.START`

**Import**

```python
from langgraph.graph import START
```

**Used in:** every LangGraph lesson that builds a `StateGraph`

**Use case:** The virtual entry point. `add_edge(START, "first_node")` tells the graph which real node runs first. `START` is not a node you define.

**Example**

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
```

---

<a id="langgraph-graph-end"></a>

## `langgraph.graph.END`

**Import**

```python
from langgraph.graph import END
```

**Used in:** every LangGraph lesson that builds a `StateGraph`

**Use case:** The virtual stop point. An edge to `END` finishes the run. Conditional routers return `END` when the model has a final answer and should not call another tool.

**Example**

```python
from langgraph.graph import END, StateGraph

builder.add_edge("assistant", END)
# or, inside a router:
def route(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

---

<a id="langgraph-graph-messagesstate"></a>

## `langgraph.graph.MessagesState`

**Import**

```python
from langgraph.graph import MessagesState
```

**Used in:** `13_Human_in_the_loop/Humanintheloop.py`, `13_Human_in_the_loop/01_breakpoints_and_approval.py` through `05_time_travel_and_replay.py`

**Use case:** A built-in state schema whose only field is `messages`, already wired to the `add_messages` reducer. Use it for a chatbot or tool agent so you do not declare that `TypedDict` yourself.

**Example**

```python
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode

def assistant(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
```

---

<a id="langgraph-graph-message-add-messages"></a>

## `langgraph.graph.message.add_messages`

**Import**

```python
from langgraph.graph.message import add_messages
```

**Used in:** `8_LangGraph-Intro/chatbot.py`, `10_LangGrapph_Component/04_state_reducers/main.py`, `10_LangGrapph_Component/06_react_agent/main.py`, `11_debugging_langGraph/openAi.py`, `14_RAG With LangGraph/1-AgenticRAG.py`

**Use case:** Reducer for a messages list. A node that returns `{"messages": [response]}` appends that message instead of replacing the whole list. It also updates a message in place when the id matches.

**Example**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

---

<a id="langgraph-prebuilt-toolnode"></a>

## `langgraph.prebuilt.ToolNode`

**Import**

```python
from langgraph.prebuilt import ToolNode
```

**Used in:** `10_LangGrapph_Component/06_react_agent/main.py`, `10_LangGrapph_Component/09_chain_with_tools/main.py`, `11_debugging_langGraph/openAi.py`, `13_Human_in_the_loop`, `14_RAG With LangGraph/1-AgenticRAG.py`

**Use case:** A prebuilt node that reads tool calls on the last `AIMessage`, runs the matching Python functions, and writes `ToolMessage` results back into state. You pass the same tool list you bound to the model.

**Example**

```python
from langgraph.prebuilt import ToolNode

builder.add_node("tools", ToolNode([add, multiply]))
builder.add_edge("tools", "assistant")
```

---

<a id="langgraph-prebuilt-tools-condition"></a>

## `langgraph.prebuilt.tools_condition`

**Import**

```python
from langgraph.prebuilt import tools_condition
```

**Used in:** `10_LangGrapph_Component/06_react_agent/main.py`, `10_LangGrapph_Component/09_chain_with_tools/main.py`, `13_Human_in_the_loop`, `14_RAG With LangGraph/1-AgenticRAG.py`

**Use case:** A prebuilt router. If the last message has `tool_calls`, it sends the graph to the node named `"tools"`. Otherwise it sends the graph to `END`. Your tools node must be named `tools` for this function to find it.

**Example**

```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
```

---

<a id="langgraph-checkpoint-memory-memorysaver"></a>

## `langgraph.checkpoint.memory.MemorySaver`

**Import**

```python
from langgraph.checkpoint.memory import MemorySaver
```

**Used in:** `10_LangGrapph_Component/07_agent_memory_checkpointer/main.py`, `10_LangGrapph_Component/12_simple_chatbot/main.py`, `10_LangGrapph_Component/13_streaming/main.py`, `13_Human_in_the_loop`

**Use case:** Checkpointer that stores each super-step in RAM, keyed by `thread_id`. Pass the same config on later calls to continue the conversation, pause on `interrupt_before`, or replay a checkpoint. State is gone when the process stops. `InMemorySaver` is the newer name for the same idea.

**Example**

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
)
config = {"configurable": {"thread_id": "chat-1"}}
graph.invoke({"messages": [("user", "What is 2 + 2?")]}, config)
```

---

<a id="langgraph-types-send"></a>

## `langgraph.types.Send`

**Import**

```python
from langgraph.types import Send
```

**Used in:** `12_Different Workflows In LangGraph/114_orchestrator_worker_langgraph.py`

**Use case:** Start one copy of a node per item, at runtime. A conditional edge returns a list of `Send(node_name, payload)` and LangGraph runs those copies in parallel. The payload is the state that copy sees, which can be smaller than the parent state.

**Example**

```python
from langgraph.types import Send

def assign_workers(state):
    return [
        Send("worker", {"objective": state["objective"], "task": task})
        for task in state["plan"]
    ]

builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
```

---

<a id="langgraph-types-interrupt"></a>

## `langgraph.types.interrupt`

**Import**

```python
from langgraph.types import interrupt
```

**Used in:** `13_Human_in_the_loop/06_modern_dynamic_interrupt.py`

**Use case:** Pause from inside a node, and only when your own condition is true. The value you pass is shown to the caller. When the graph resumes, `interrupt(...)` returns whatever you passed to `Command(resume=...)`. The graph must be compiled with a checkpointer.

**Example**

```python
from langgraph.types import interrupt

def human_approval_guard(state):
    if state["amount"] < 1000:
        return {"status": "approved"}
    decision = interrupt({
        "prompt": f"Approve ${state['amount']} to {state['recipient']}?",
        "amount": state["amount"],
    })
    if decision.get("approved"):
        return {"status": "approved"}
    return {"status": "rejected"}
```

---

<a id="langgraph-types-command"></a>

## `langgraph.types.Command`

**Import**

```python
from langgraph.types import Command
```

**Used in:** `7_Langchain Version V1/6-middleWare.py`, `13_Human_in_the_loop/06_modern_dynamic_interrupt.py`

**Use case:** A control object you pass back into `invoke` or `stream` to resume a paused graph. `Command(resume=value)` becomes the return value of the `interrupt()` that paused. The same `thread_id` is required. Human-in-the-loop middleware resumes with a decision payload in this object.

**Example**

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "transfer-1"}}
graph.invoke({"sender": "Ada", "recipient": "Grace", "amount": 5000}, config)
graph.invoke(Command(resume={"approved": True, "notes": "Looks fine."}), config)
```

---

<a id="langgraph-checkpoint-memory-inmemorysaver"></a>

## `langgraph.checkpoint.memory.InMemorySaver`

**Import**

```python
from langgraph.checkpoint.memory import InMemorySaver
```

**Used in:** `7_Langchain Version V1/6-middleWare.py`

**Use case:** LangGraph checkpointer that stores agent state in RAM. `create_agent` uses it so a `thread_id` can resume the same conversation, including after summarization middleware runs. State is gone when the process stops.

**Example**

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "test-1"}}
agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that my name is Piyush."}]},
    config=config,
)
```

---


[Back to contents](#contents)
