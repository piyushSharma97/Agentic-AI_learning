# LangChain imports used in this project

Every LangChain-related import found across the course folders, with what it is for and a small example.

Folders scanned:

- `2_Langchain`
- `4_GenAI`
- `5_LCEL`
- `6_Building AI agents With Conver`
- `7_Langchain Version V1`
- `dummy.py`

---

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

## Quick index

| Import | Package | Main job |
| --- | --- | --- |
| `langchain` | `langchain` | Check the installed version |
| `create_agent` | `langchain.agents` | Build a v1 tool-calling agent |
| `SummarizationMiddleware` | `langchain.agents.middleware` | Summarize long chats |
| `ToolStrategy` | `langchain.agents.structured_output` | Agent replies as a schema |
| `init_chat_model` | `langchain.chat_models` | One call for any provider |
| `tool` | `langchain.tools` | Turn a function into a tool |
| `create_retrieval_chain` | `langchain.chains` | RAG: retrieve, then answer |
| `create_stuff_documents_chain` | `langchain.chains.combine_documents` | Put chunks into one prompt |
| `HumanMessage` | `langchain_core.messages` | User turn |
| `SystemMessage` | `langchain_core.messages` | Role and rules |
| `AIMessage` | `langchain_core.messages` | Assistant turn |
| `ToolMessage` | `langchain_core.messages` | Tool result sent back to the model |
| `trim_messages` | `langchain_core.messages` | Cut history to a token budget |
| `ChatPromptTemplate` | `langchain_core.prompts` | Prompt with placeholders |
| `MessagesPlaceholder` | `langchain_core.prompts` | Slot for chat history |
| `StrOutputParser` | `langchain_core.output_parsers` | Message to plain string |
| `BaseChatMessageHistory` | `langchain_core.chat_history` | History interface |
| `RunnablePassthrough` | `langchain_core.runnables` | Pass or add keys in LCEL |
| `RunnableWithMessageHistory` | `langchain_core.runnables.history` | Auto load and save history |
| `TextLoader` | `langchain_community.document_loaders` | Load a text file |
| `PyPDFLoader` | `langchain_community.document_loaders` | Load a PDF |
| `WebBaseLoader` | `langchain_community.document_loaders` | Load a web page |
| `Chroma` | `langchain_community.vectorstores` | Older Chroma vector store |
| `FAISS` | `langchain_community.vectorstores` | Local FAISS index |
| `ChatMessageHistory` | `langchain_community.chat_message_histories` | In-memory session history |
| `CharacterTextSplitter` | `langchain_text_splitters` | Split on one separator |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Split by paragraphs, then lines |
| `ChatOpenAI` | `langchain_openai` | OpenAI chat model |
| `OpenAIEmbeddings` | `langchain_openai` | OpenAI embeddings |
| `ChatGoogleGenerativeAI` | `langchain_google_genai` | Gemini chat model |
| `ChatGroq` | `langchain_groq` | Groq chat model |
| `ChatOllama` | `langchain_ollama` | Local Ollama chat model |
| `OllamaEmbeddings` | `langchain_ollama` | Local Ollama embeddings |
| `HuggingFaceEmbeddings` | `langchain_huggingface` | Local Hugging Face embeddings |
| `Chroma` | `langchain_chroma` | Current Chroma package |
| `InMemorySaver` | `langgraph.checkpoint.memory` | Remember agent state by thread id |
