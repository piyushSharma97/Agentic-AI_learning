"""
Production RAG API: FastAPI, Server-Sent Events (SSE) & Async Streaming

What you will learn in this production backend:
1. Building Web APIs for Generative AI using FastAPI and Pydantic.
2. Startup Lifecycle Events: Indexing documents once at server startup rather than on every HTTP request.
3. Asynchronous Streaming: Streaming LLM token outputs in real-time as they are generated using `chain.astream()`.
4. Server-Sent Events (SSE): The industry standard protocol (`text/event-stream`) for streaming text to browsers.
5. Pydantic Request Validation: Enforcing schemas, min-lengths, and sanitization on incoming JSON payloads.
"""

import os
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

# FastAPI framework & HTTP response exceptions
from fastapi import FastAPI, HTTPException

# StreamingResponse allows sending incremental chunks over an open HTTP connection
from fastapi.responses import StreamingResponse

# Pydantic enforces strict type checking and validation on API request bodies
from pydantic import BaseModel, Field

# LangChain Document Loading, Chunking, Embedding & Vector Storage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Prompt Engineering & Output Parsing
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 1. Environment & FastAPI Application Initialization
# --------------------------------------------------

load_dotenv()

# Create the FastAPI application instance
app = FastAPI(
    title="Production RAG Backend",
    version="1.0.0",
    description="High-performance, async-streaming RAG API backend powered by LangChain and Chroma."
)

# Global variables to hold the vector database and retriever in RAM across API requests
VECTOR_DB = None
RETRIEVER = None

# --------------------------------------------------
# 2. Startup Lifecycle Event: Initialize Knowledge Base
# --------------------------------------------------

# Why use a startup event?
# Ingesting, chunking, and embedding documents is computationally expensive.
# By running this during server startup, the index is built once and kept in memory,
# allowing incoming HTTP queries to be answered with sub-second latency!
@app.on_event("startup")
def initialize_rag():
    global VECTOR_DB, RETRIEVER
    
    # Resolve speech.txt path robustly from the project directory
    project_root = Path(__file__).resolve().parent
    file_path = project_root / "extra" / "speech.txt"
    if not file_path.exists():
        file_path = project_root / "2_Langchain" / "04_Data Trnasformer" / "speech.txt"
    if not file_path.exists():
        file_path = Path("speech.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Could not locate document at {file_path}")

    print(f"[Startup] Ingesting knowledge base from: {file_path}")
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    # Split document into 150-character chunks with 30-character overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # Initialize OpenAI embeddings (using 1024 dimensions for storage efficiency)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

    # Index chunks in an in-memory Chroma collection
    VECTOR_DB = Chroma.from_documents(chunks, embeddings)

    # Configure the retriever to pull top k=3 chunks
    RETRIEVER = VECTOR_DB.as_retriever(search_kwargs={"k": 3})
    print(f"[Startup] RAG pipeline initialized successfully with {len(chunks)} chunks.")

# --------------------------------------------------
# 3. Request Schemas (Pydantic)
# --------------------------------------------------

# Pydantic validates incoming JSON bodies automatically.
# If a client sends {"query": "hi"}, it automatically rejects the request with HTTP 422
# because the min_length requirement is 3 characters!
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="The natural language question to ask the RAG knowledge base.",
        example="What did the speaker say regarding international peace?"
    )

# --------------------------------------------------
# 4. Asynchronous Streaming Engine (Server-Sent Events)
# --------------------------------------------------

# What are Server-Sent Events (SSE)?
# Rather than waiting 5-10 seconds for the entire answer to generate before sending an HTTP response,
# SSE keeps the HTTP connection open and pushes tokens to the client as they stream from the LLM.
# SSE Protocol format: "data: <payload>\n\n"
async def generate_rag_stream(query: str) -> AsyncGenerator[str, None]:
    """Retrieves relevant passages, formats the prompt, and yields SSE chunks."""
    # 1. Retrieve relevant chunks
    docs = RETRIEVER.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Strict grounding prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    )

    # 3. Initialize streaming LLM
    # streaming=True enables token-by-token emission from OpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

    # 4. Assemble the LCEL chain
    chain = prompt | llm | StrOutputParser()

    # 5. astream() asynchronously iterates over generated tokens in real time
    async for chunk in chain.astream({"context": context_text, "question": query}):
        # Format chunk according to Server-Sent Events specification:
        yield f"data: {chunk}\n\n"
    
    # Signal to the frontend client that the generation stream has finished
    yield "data: [DONE]\n\n"

# --------------------------------------------------
# 5. FastAPI HTTP Endpoints
# --------------------------------------------------

@app.post("/api/query")
async def rag_query_endpoint(payload: QueryRequest):
    """
    POST endpoint accepting a validated query and returning a live token stream.
    Media type 'text/event-stream' instructs browser clients (EventSource) to handle as an SSE stream.
    """
    if not RETRIEVER:
        raise HTTPException(
            status_code=500,
            detail="Retriever not initialized. Ensure startup event completed successfully."
        )
    
    return StreamingResponse(
        generate_rag_stream(payload.query),
        media_type="text/event-stream"
    )

# Direct execution entry point
if __name__ == "__main__":
    import uvicorn
    # uvicorn runs the ASGI web server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ============================================================
# REVISION NOTES — dummy.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file provides a production-grade asynchronous web API for Retrieval-Augmented Generation using FastAPI.
# It indexes documents once during server startup and streams real-time generated responses to client applications
# over HTTP using Server-Sent Events (SSE).
#
# KEY CONCEPTS USED:
# • FastAPI: Modern asynchronous web framework for Python built on ASGI and Starlette.
# • Startup Event (@app.on_event("startup")): One-time initialization logic executed when the server boots up.
# • Pydantic Schema: Automatic runtime validation and documentation of API request payloads.
# • StreamingResponse & SSE (`text/event-stream`): Streaming protocol delivering tokens one by one as they generate.
# • chain.astream(): Asynchronous generator protocol in LangChain that yields tokens incrementally.
#
# WHY SSE OVER WEBSOCKETS FOR RAG?
# • WebSockets are bidirectional and complex to maintain behind corporate firewalls and load balancers.
# • Server-Sent Events (SSE) use standard HTTP requests, are unidirectional (server -> client),
#   reconnect automatically, and are natively supported by browsers via EventSource or fetch().
#
# THINGS TO REMEMBER:
# • The endpoint returns `text/event-stream`; standard JSON fetch will see an open stream of `data: ...` blocks.
# • The global RETRIEVER is populated on startup; if startup fails, queries return HTTP 500.
# • Run with uvicorn: `uvicorn dummy:app --reload`
#
# COMMON MISTAKES TO AVOID:
# • Re-indexing documents inside the `/api/query` route, which would cause each query to take 10+ seconds.
# • Forgetting the trailing `\n\n` in SSE event formatting, which prevents browsers from flushing incoming events.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow `generate_rag_stream(query)`:
# 1. RETRIEVER.invoke(query) retrieves matching context.
# 2. chain.astream(...) streams the LLM response tokens.
# 3. StreamingResponse returns the async generator over HTTP.
# ============================================================
