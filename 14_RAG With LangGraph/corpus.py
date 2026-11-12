"""
The knowledge base both lessons retrieve from.

The lecture downloads live documentation with WebBaseLoader and stores it
in FAISS. Those pages change, and faiss is not installed in this project.
A short local corpus teaches the same steps, and you can read every chunk:

    notes -> split -> embed -> vector store -> retriever

InMemoryVectorStore has the same .as_retriever() method the graph uses.
Swap in FAISS.from_documents later if you install faiss-cpu. The graph
code does not change.
"""

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_embeddings


# Each item is (source name, text). The source is stored in metadata so a
# retrieved chunk can be traced back to the note it came from.
LANGGRAPH_NOTES = [
    (
        "langgraph-overview",
        "LangGraph is a library for building stateful agent workflows as a graph. "
        "A node is a Python function that reads the current state and returns a partial update. "
        "An edge says which node runs next. A conditional edge chooses the next node from the state, "
        "for example by checking whether the model asked to call a tool. "
        "The graph is compiled before it is invoked. Compilation checks that every edge points at a real node.",
    ),
    (
        "langgraph-checkpointer",
        "A checkpointer saves the graph state after each super-step. "
        "The saved snapshot is a checkpoint, and a thread id selects which conversation the checkpoint belongs to. "
        "With a checkpointer, the next user message can continue from the previous state instead of starting blank. "
        "MemorySaver stores checkpoints in process memory. A database checkpointer keeps them after the process exits. "
        "Interrupt and human approval also depend on a checkpointer, because the graph must pause and later resume.",
    ),
]

LANGCHAIN_NOTES = [
    (
        "langchain-overview",
        "LangChain is a toolkit for calling chat models, prompts, and tools from Python. "
        "A prompt template fills variables into a fixed instruction. "
        "A chat model returns a message. Tools are functions the model can ask the application to run, "
        "such as a retriever or a web search. LangChain components are often composed in a chain, "
        "but a chain does not by itself remember earlier turns or branch on a condition.",
    ),
    (
        "langchain-chat-history",
        "A LangChain chatbot remembers earlier turns by keeping a list of messages. "
        "HumanMessage is the user, AIMessage is the model, and SystemMessage is the standing instruction. "
        "The application sends that list back on the next call. "
        "If the list grows past the context window, older messages are trimmed or summarized. "
        "Message history is application state. The model does not store it between separate API calls by itself.",
    ),
]

# Narrow on purpose. Corrective RAG should fail this corpus for a current-events
# question and then go to the web. If this corpus already answered everything,
# the web-search branch would never run.
AGENT_NOTES = [
    (
        "agent-loop",
        "An LLM agent is a model inside a loop. The model reads a goal, chooses an action, "
        "observes the result, and chooses again. A planner proposes the next steps before they run. "
        "A tool is an action with a name, arguments, and a text observation. "
        "The loop stops when the model returns a final answer or a step limit is reached.",
    ),
    (
        "agent-memory",
        "Agent memory is the information the loop can still see on a later step. "
        "Short-term memory is the message list of the current task. "
        "Long-term memory is an external store, such as a vector index of past tasks, "
        "that the agent retrieves when a new task looks related. "
        "Writing every thought into long-term memory makes later retrieval noisy.",
    ),
]


def build_retriever(notes: list[tuple[str, str]], label: str, k: int = 2):
    """
    Turn notes into a retriever.

    chunk_size is in characters. Overlap keeps a sentence that sits on a
    boundary from being cut in half with no copy in the next chunk.
    k is how many chunks a query returns.
    """
    documents = [
        Document(page_content=text, metadata={"source": source})
        for source, text in notes
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    print(f"[index:{label}] {len(documents)} notes -> {len(chunks)} chunks")

    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": k})


def preview_retrieval(retriever, query: str) -> None:
    """Show the raw chunks before any graph runs, so retrieval is visible on its own."""
    print(f"\n[retrieve] query: {query}")
    docs = retriever.invoke(query)
    if not docs:
        print("  no chunks")
        return
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        snippet = " ".join(doc.page_content.split())[:180]
        print(f"  - {source}: {snippet}")


def make_retriever_tool(retriever, name: str, description: str):
    """
    A retriever is not a tool until it has a name and a description.
    The agent reads those two strings when it decides which tool to call.
    The function returns plain text, which ToolNode wraps in a ToolMessage.
    """

    def search(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No matching notes."
        blocks = []
        for doc in docs:
            source = doc.metadata.get("source", "notes")
            blocks.append(f"[{source}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    return StructuredTool.from_function(func=search, name=name, description=description)
