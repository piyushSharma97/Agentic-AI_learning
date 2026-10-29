"""
Production RAG Application: OpenAI, WebBaseLoader, FAISS & LangChain Chains

What is Retrieval-Augmented Generation (RAG)?
Large Language Models have two major limitations:
1. Knowledge cutoff: They don't know about facts published after their training date.
2. Private data: They don't have access to your organization's private documents or live website updates.

Instead of retraining or fine-tuning models (which costs thousands of dollars and weeks of GPU compute),
RAG dynamically fetches relevant text chunks from an external knowledge base and injects ("stuffs") them
into the LLM's prompt at runtime.

The 8-Stage Production RAG Lifecycle:
1. Data Ingestion: Fetch website text with WebBaseLoader.
2. Chunking: Split text into overlapping segments with RecursiveCharacterTextSplitter.
3. Embedding Generation: Convert text segments into vectors with OpenAIEmbeddings.
4. Vector Storage: Index vectors in memory with FAISS.
5. Retriever Creation: Expose the vector store via LangChain's retriever interface.
6. Generation Model: Initialize ChatOpenAI (gpt-4o-mini).
7. Document Combination Chain: Format prompt with {context} using create_stuff_documents_chain.
8. Full Retrieval Chain: Orchestrate retriever + document chain with create_retrieval_chain.
"""

import os
from dotenv import load_dotenv

# WebBaseLoader scrapes and parses live HTML web pages.
from langchain_community.document_loaders import WebBaseLoader

# RecursiveCharacterTextSplitter chunks long documents at natural paragraph/sentence boundaries.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAIEmbeddings maps text chunks to dense vectors; ChatOpenAI generates the final synthesis.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# FAISS is Meta's high-speed in-memory vector indexing and search engine.
from langchain_community.vectorstores import FAISS

# ChatPromptTemplate builds structured prompts with role separation.
from langchain_core.prompts import ChatPromptTemplate

# create_stuff_documents_chain stuffs all retrieved context documents into a single prompt call.
from langchain.chains.combine_documents import create_stuff_documents_chain

# create_retrieval_chain coordinates the retriever search and routes matching docs to the LLM.
from langchain.chains import create_retrieval_chain

# --------------------------------------------------
# 1. Environment & API Configuration
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY is not set in environment or .env file.")

# --------------------------------------------------
# 2. Ingest Data from Live Web URL
# --------------------------------------------------

# Target documentation page to ingest
url = "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend"
print(f"Step 1: Ingesting web documentation from:\n  {url}")

loader = WebBaseLoader(url)
docs = loader.load()

if not docs:
    raise ValueError("No documents were loaded from the website.")

print(f"Successfully loaded {len(docs)} document with {len(docs[0].page_content)} characters.")

# --------------------------------------------------
# 3. Text Chunking & Splitting
# --------------------------------------------------

# Break the long article into 1000-character segments with 220-character overlap.
# Overlap preserves contextual continuity between adjacent segments.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=220,
)
documents = text_splitter.split_documents(docs)
print(f"Step 2: Created {len(documents)} focused text chunks.")

# --------------------------------------------------
# 4. Generate Embeddings & Index with FAISS
# --------------------------------------------------

# text-embedding-3-small is OpenAI's cost-effective, high-accuracy embedding model (1536 dims).
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

print("Step 3: Generating embeddings and indexing chunks into FAISS vector store...")
vectorStore = FAISS.from_documents(documents=documents, embedding=embedding)

# Convert vector store to a Retriever.
# search_kwargs={"k": 5} ensures the top 5 most relevant chunks are retrieved for each query.
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
print("Retriever configured (k=5).")

# --------------------------------------------------
# 5. Initialize LLM & Create Grounded Prompt Template
# --------------------------------------------------

# gpt-4o-mini is OpenAI's fast, affordable model ideal for RAG synthesis tasks.
# temperature=0 ensures grounded, deterministic answers strictly derived from context.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Grounding System Prompt:
# Instructs the model to ONLY use information in {context}.
# If the answer is not present, it admits lack of information rather than hallucinating!
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant answering questions about the provided documentation.

Use ONLY the information in the retrieved context below.
If the answer is not present in the context, clearly state that you do not have enough information.

Context:
{context}
""",
        ),
        ("human", "{input}"),
    ]
)

# --------------------------------------------------
# 6. Build the Document & Retrieval Chains
# --------------------------------------------------

# "Stuffing" Chain:
# Takes the list of retrieved Document objects, joins their page_content into a single string,
# and inserts it into the {context} variable of our prompt template.
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

# Complete RAG Chain:
# 1. Takes user input query {"input": "..."}.
# 2. Passes query to retriever -> fetches top 5 documents.
# 3. Hands retrieved documents + query to document_chain -> LLM answers using context.
# 4. Returns a dictionary containing both {"answer": "...", "context": [Document, ...]}.
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain,
)

# --------------------------------------------------
# 7. Ask Questions & Inspect Retrieved Evidence
# --------------------------------------------------

query = "What are the two usage limits in LangSmith, and what do they measure?"
print(f"\nStep 4: Executing RAG Pipeline for query:\n  \"{query}\"")

# invoke() runs both the retrieval search and the LLM synthesis
response = retrieval_chain.invoke({"input": query})

print("\n" + "=" * 50)
print("FINAL GENERATED ANSWER:")
print("=" * 50)
print(response["answer"])

print("\n" + "=" * 50)
print(f"RETRIEVED CONTEXT CHUNKS ({len(response['context'])} documents used):")
print("=" * 50)
for i, doc in enumerate(response["context"], start=1):
    print(f"\n[Source Chunk {i} - Characters: {len(doc.page_content)}]")
    print(doc.page_content[:250].strip() + "...\n")

# ============================================================
# REVISION NOTES — Simpleapp.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is a complete, production-grade implementation of Retrieval-Augmented Generation (RAG).
# It scrapes live technical documentation from LangSmith, segments it into overlapping chunks,
# creates vector embeddings using OpenAI, indexes them with FAISS, and uses an OpenAI chat model
# to answer specific questions with zero hallucinations.
#
# KEY CONCEPTS USED:
# • RAG (Retrieval-Augmented Generation): Providing relevant retrieved documents as prompt context to ground LLM outputs.
# • WebBaseLoader: Scrapes real-world technical documentation into Document objects.
# • FAISS (Facebook AI Similarity Search): In-memory vector database delivering ultra-fast nearest-neighbor lookups.
# • create_stuff_documents_chain: Consolidates retrieved documents into the `{context}` slot of a prompt.
# • create_retrieval_chain: End-to-end coordinator managing input routing, vector search, and answer generation.
# • Grounded Prompting: System prompt explicitly demanding the model admit ignorance if facts are not in context.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file represents the culmination of the OpenAI module (4_GenAI/01_openai),
# integrating loaders, text splitters, embeddings, vector stores, and chains into a single cohesive application.
#
# THINGS TO REMEMBER:
# • The response dictionary contains both `response["answer"]` (the synthesized text)
#   and `response["context"]` (the raw Document chunks used by the model).
# • Always inspect `response["context"]` to verify retrieval quality during debugging.
# • Set temperature=0 for RAG applications to minimize hallucinations.
#
# COMMON MISTAKES TO AVOID:
# • Not instructing the model to stick strictly to `{context}`, leading to hallucinations from pre-training memory.
# • Forgetting that web pages change; if the URL structure changes, the loader or content extraction may fail.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the two-chain setup:
# 1. `document_chain = create_stuff_documents_chain(llm, prompt)`
# 2. `retrieval_chain = create_retrieval_chain(retriever, document_chain)`
# ============================================================
