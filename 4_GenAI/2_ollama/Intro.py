"""
100% Local, Private RAG: Ollama, FAISS, Nomic Embeddings & Gemma

Why Build a 100% Local RAG System?
1. Zero API Costs: Unlimited queries, embeddings, and completions without recurring bills or tokens fees.
2. Complete Data Confidentiality: Sensitive internal documents, medical records, or proprietary IP
   never travel over the public internet to third-party cloud servers.
3. Air-Gapped Operation: Works completely offline on your laptop or on-prem servers once models are downloaded.

Local Tech Stack Used Here:
• Document Loader: WebBaseLoader (fetches documentation).
• Text Splitter: RecursiveCharacterTextSplitter (chunks text into manageable pieces).
• Embedding Model: OllamaEmbeddings with `nomic-embed-text` (superb 768-dim embeddings with 8192 context window).
• Vector Store: FAISS (high-speed local in-memory vector index).
• LLM: ChatOllama with `gemma:2b` (Google's efficient 2-billion parameter model).
• Chains: create_stuff_documents_chain and create_retrieval_chain.
"""

import os
from dotenv import load_dotenv

# WebBaseLoader scrapes web content into LangChain Document objects.
from langchain_community.document_loaders import WebBaseLoader

# RecursiveCharacterTextSplitter splits documents at natural sentence boundaries.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS creates an in-memory vector index for rapid nearest-neighbor retrieval.
from langchain_community.vectorstores import FAISS

# ChatOllama runs local chat models; OllamaEmbeddings runs local embedding models.
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ChatPromptTemplate builds structured prompts.
from langchain_core.prompts import ChatPromptTemplate

# create_stuff_documents_chain stuffs retrieved documents into prompt context.
from langchain.chains.combine_documents import create_stuff_documents_chain

# create_retrieval_chain routes queries to retriever and passes matching docs to the LLM.
from langchain.chains import create_retrieval_chain

# --------------------------------------------------
# 1. Environment & Optional LangSmith Tracing
# --------------------------------------------------

load_dotenv()

# Optional: Enables tracing on LangSmith if API key is present.
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT", "ollama-rag-project"
    )

# --------------------------------------------------
# 2. Ingest Data from Target Website
# --------------------------------------------------

url = "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend"
print(f"Step 1: Ingesting web documentation from:\n  {url}")

loader = WebBaseLoader(url)
docs = loader.load()

if not docs:
    raise ValueError("No documents were loaded from the website.")

print(f"Loaded {len(docs)} document with {len(docs[0].page_content)} characters.")

# --------------------------------------------------
# 3. Chunk Documents into Search-Friendly Passages
# --------------------------------------------------

# Split long text into 1000-character chunks with 200-character overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(docs)
print(f"Step 2: Created {len(documents)} focused text chunks.")

# --------------------------------------------------
# 4. Generate Local Ollama Embeddings (nomic-embed-text)
# --------------------------------------------------

# nomic-embed-text is an open-source embedding model outperforming OpenAI text-embedding-ada-002
# with an 8192 token context window.
print("Step 3: Initializing local Ollama embedding model (nomic-embed-text)...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# --------------------------------------------------
# 5. Build FAISS Vector Index & Configure Retriever
# --------------------------------------------------

# FAISS indexes document vectors in RAM for instant similarity searches
print("Step 4: Indexing document chunks into FAISS vector store...")
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
)

# Configure retriever to fetch the top k=4 most relevant chunks
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
print("Retriever configured (k=4).")

# --------------------------------------------------
# 6. Initialize Local Ollama Chat Model (gemma:2b)
# --------------------------------------------------

# Gemma 2B is Google's lightweight, open-weights model designed for efficient local execution.
# temperature=0 ensures factual, deterministic reasoning strictly anchored in context.
print("Step 5: Initializing local ChatOllama model (gemma:2b)...")
llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
)

# --------------------------------------------------
# 7. Define Grounding Prompt Template
# --------------------------------------------------

# Strict system instructions prevent the model from making up answers when context is missing
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using ONLY the retrieved context below.

If the answer is not available in the context, say clearly that you do not have enough information.

Context:
{context}""",
        ),
        ("human", "{input}"),
    ]
)

# --------------------------------------------------
# 8. Assemble Document & Retrieval Chains
# --------------------------------------------------

# Document chain stuffs the retrieved documents into the {context} placeholder of prompt
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Retrieval chain connects retriever -> document_chain -> LLM response
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

# --------------------------------------------------
# 9. Ask a Question & Inspect Results
# --------------------------------------------------

query = "What are the two usage limits in LangSmith, and what do they measure?"
print(f"\nStep 6: Executing Local RAG Query:\n  \"{query}\"")

response = retrieval_chain.invoke({"input": query})

print("\n" + "=" * 50)
print("GENERATED LOCAL ANSWER (Gemma 2B):")
print("=" * 50)
print(response["answer"])

print("\n" + "=" * 50)
print(f"RETRIEVED CONTEXT EVIDENCE ({len(response['context'])} chunks):")
print("=" * 50)
for i, doc in enumerate(response["context"], start=1):
    print(f"\n[Document Chunk {i}]")
    print(doc.page_content[:250].strip() + "...\n")

# ============================================================
# REVISION NOTES — Intro.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is a complete, self-contained demonstration of a 100% local, private RAG pipeline.
# It downloads an online article, creates local vector representations via Ollama's nomic-embed-text,
# indexes them using FAISS, and generates answers using Google's local Gemma 2B model without
# sending any data to cloud providers.
#
# KEY CONCEPTS USED:
# • Local RAG: Running document ingestion, vector storage, and inference entirely on your personal workstation.
# • nomic-embed-text: High-performance open-source embedding model.
# • Gemma 2B: Efficient open-weights LLM by Google suitable for consumer hardware.
# • FAISS: High-performance local vector similarity engine.
# • create_retrieval_chain: Orchestrates the search -> context stuffing -> generation sequence.
#
# LOCAL RAG VS CLOUD RAG COMPARISON:
# • Cloud RAG (Simpleapp.py): Uses OpenAI API; higher model capacity (GPT-4o), but incurs costs and sends data to cloud.
# • Local RAG (Intro.py): Uses Ollama + FAISS; 100% free and confidential, runs offline, but depends on local RAM/GPU capacity.
#
# THINGS TO REMEMBER:
# • Before running, ensure both models are downloaded:
#   `ollama pull nomic-embed-text`
#   `ollama pull gemma:2b`
# • Ollama daemon must be running in the background.
#
# COMMON MISTAKES TO AVOID:
# • Asking questions not contained in the ingested URL; the model will (and should!) state it does not have enough information.
# • Expecting 2B parameter models to write long essays as fluently as 70B+ parameter models; smaller models excel when given tight, focused context.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the flow:
# WebBaseLoader -> RecursiveCharacterTextSplitter -> nomic-embed-text -> FAISS -> ChatOllama -> create_retrieval_chain.
# ============================================================
