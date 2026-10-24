# os lets this script read and set environment variables, including optional tracing settings.
import os
# python-dotenv loads settings from a local .env file so credentials need not be written in code.
from dotenv import load_dotenv

# WebBaseLoader downloads a web page and turns its text into documents for later retrieval.
from langchain_community.document_loaders import WebBaseLoader
# RecursiveCharacterTextSplitter breaks long text into smaller, overlapping pieces for useful retrieval.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# FAISS stores embedding vectors and quickly finds the vectors most similar to a question.
from langchain_community.vectorstores import FAISS
# ChatOllama runs a local chat model; OllamaEmbeddings converts text into vectors using a local model.
from langchain_ollama import ChatOllama, OllamaEmbeddings
# ChatPromptTemplate builds the structured system and user messages sent to the chat model.
from langchain_core.prompts import ChatPromptTemplate
# This helper builds a chain that places retrieved documents into the prompt for the language model.
from langchain.chains.combine_documents import create_stuff_documents_chain
# This helper connects document retrieval with answering the user's question.
from langchain.chains import create_retrieval_chain

# --------------------------------------------------
# 1. Environment variables and LangSmith tracking
# --------------------------------------------------


# Load any local .env values into the process environment before reading configuration.
load_dotenv()

# Read the optional LangSmith key; when present, enable tracing and select a project name.
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT", "ollama-rag-project"
    )




# --------------------------------------------------
# 2. Load data from website
# --------------------------------------------------

# This is the source page whose content will be searched to answer the question.
url = (
    "https://docs.smith.langchain.com/"
    "tutorials/Administrators/manage_spend"
)

# The loader fetches the page and represents its content as LangChain documents.
loader = WebBaseLoader(url)
docs = loader.load()

# Stop early rather than building a search index with no source content.
if not docs:
    raise ValueError("No Documents are loaded")


print(f"Loaded {len(docs)} documents")


# --------------------------------------------------
# 3. Split documents into chunks
# --------------------------------------------------


# Chunking makes long pages easier to search; overlap preserves context across chunk boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
# Each smaller document keeps source metadata, which can be useful when inspecting retrieval results.
documents = text_splitter.split_documents(docs)

print(f"Created {len(documents)} chunks")

# --------------------------------------------------
# 4. Create local Ollama embeddings
# --------------------------------------------------

# An embedding is a numeric vector that captures text meaning so related text can be compared.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# --------------------------------------------------
# 5. Store embeddings in FAISS
# --------------------------------------------------

# FAISS builds a local searchable index from the document vectors; k=4 asks for four likely matches.
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# --------------------------------------------------
# 6. Initialize local Ollama chat model
# --------------------------------------------------
# This local language model writes the answer; temperature=0 favors consistent, less-random responses.
llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
)

# --------------------------------------------------
# 7. Create prompt template
# --------------------------------------------------

# The system message limits answers to retrieved evidence; {context} and {input} are filled by the chain.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using only the
retrieved context below.

If the answer is not available in the context,
say that you do not have enough information.

Context:
{context}""",
        ),
        ("human", "{input}"),
    ]
)
# --------------------------------------------------
# 8. Create document chain
# --------------------------------------------------

# "Stuff" combines the retrieved chunks into the context sent to the model in one prompt.
document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)

# --------------------------------------------------
# 9. Create retrieval chain
# --------------------------------------------------
# This chain first finds relevant chunks, then passes them to the document-answering chain.
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
)
# --------------------------------------------------
# 10. Ask a question
# --------------------------------------------------
# The question guides vector search and becomes the human input in the prompt.
query = (
    "What are the two usage limits in LangSmith, "
    "and what do they measure?"
)

# Invoke runs retrieval and answer generation together; the result includes both answer and context.
response = retrieval_chain.invoke({"input": query})

# Print the generated answer and the source chunks so retrieval can be inspected.
print("\nANSWER:")
print(response["answer"])

print("\nRETRIEVED CONTEXT:")

for i, doc in enumerate(response["context"], start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)

# ============================================================
# REVISION NOTES — Intro.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script downloads a LangSmith documentation page, breaks it into searchable chunks, and indexes those chunks locally.
# It retrieves the chunks most related to a question and asks a local Ollama chat model to answer using that context.
#
# KEY CONCEPTS USED:
# • Chunking: splitting long text into smaller overlapping pieces so searches can find relevant passages.
# • Embeddings: numeric vectors that represent text meaning and let the program compare related text.
# • Retrieval-augmented generation (RAG): retrieving relevant source text and giving it to a language model to ground its answer.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is the retrieval-based Ollama example in 4_GenAI/2_ollama; unlike app.py in the same folder, which answers direct chat questions, this script first searches a web page.
#
# THINGS TO REMEMBER:
# • Ollama must be running locally, with both gemma:2b and nomic-embed-text available.
# • The LangSmith API key is optional; it enables tracing but is not needed for the local retrieval and answer flow.
# • Retrieved context is printed after the answer so you can inspect what information the model received.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting to install or run the Ollama models, which are separate from the Python packages.
# • Assuming the model knows the page contents without retrieval; the prompt is designed to answer from the supplied context.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow retrieval_chain.invoke({"input": query}): it retrieves matching chunks and then generates an answer from them.
# ============================================================