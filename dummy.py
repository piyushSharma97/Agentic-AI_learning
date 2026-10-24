# os reads and sets environment variables so API credentials can be loaded without hard-coding them.
import os
# Path resolves document locations relative to this file rather than the caller's working directory.
from pathlib import Path
# AsyncGenerator describes the type of asynchronous text stream returned by the RAG generator.
from typing import AsyncGenerator
# python-dotenv loads local settings, such as API keys, from a .env file.
from dotenv import load_dotenv

# FastAPI defines HTTP endpoints; HTTPException returns a structured HTTP error response.
from fastapi import FastAPI, HTTPException
# StreamingResponse sends generated text incrementally instead of waiting for the full answer.
from fastapi.responses import StreamingResponse
# Pydantic validates incoming request data and Field adds a minimum-length rule and description.
from pydantic import BaseModel, Field

# TextLoader reads the local source document into LangChain Document objects.
from langchain_community.document_loaders import TextLoader
# RecursiveCharacterTextSplitter divides long text into manageable, overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAIEmbeddings maps text to vectors; ChatOpenAI generates the response from retrieved context.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Chroma stores document vectors and supports similarity-based retrieval.
from langchain_community.vectorstores import Chroma
# ChatPromptTemplate creates a reusable prompt with slots for context and the question.
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser turns the model's message output into plain text chunks.
from langchain_core.output_parsers import StrOutputParser

# Load .env settings before initializing any service that may need credentials.
load_dotenv()

# Create the FastAPI application object used to register endpoints and startup behavior.
app = FastAPI(title="Production RAG Backend", version="1.0.0")

# --- 1. Global Instances & Startup Pipeline ---
VECTOR_DB = None
RETRIEVER = None

# Build the document index once when the API starts; this takes no input and populates the global DB/retriever.
@app.on_event("startup")
def initialize_rag():
    global VECTOR_DB, RETRIEVER
    
    # Resolve speech.txt path cleanly
    file_path = Path(__file__).resolve().parent.parent.parent / "extra" / "speech.txt"
    if not file_path.exists():
        # Fallback to local directory if relative path differs
        file_path = Path("speech.txt")
        if not file_path.exists():
            raise FileNotFoundError(f"Could not locate document at {file_path}")

    # Load the source into document objects, then split it so retrieval can return focused passages.
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # Embeddings are meaning-oriented number vectors; Chroma indexes them for similarity search.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    VECTOR_DB = Chroma.from_documents(chunks, embeddings)
    RETRIEVER = VECTOR_DB.as_retriever(search_kwargs={"k": 3})
    print(f"RAG initialized with {len(chunks)} chunks.")

# --- 2. Schemas ---
# QueryRequest validates the incoming JSON body; it receives a query string and produces a validated model.
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")

# --- 3. Streaming Engine ---
# Retrieve relevant chunks and asynchronously yield answer text in Server-Sent Event format.
# The function receives the user's query and yields a sequence of encoded text events.
async def generate_rag_stream(query: str) -> AsyncGenerator[str, None]:
    # Retrieval compares the question with stored embedding vectors to select relevant source chunks.
    docs = RETRIEVER.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # The prompt supplies retrieved context; the chain pipes that prompt through the model and text parser.
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using ONLY the provided context.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
    chain = prompt | llm | StrOutputParser()

    # astream produces answer pieces incrementally; SSE wraps each piece for browser/client streaming.
    async for chunk in chain.astream({"context": context_text, "question": query}):
        # Format as Server-Sent Event data block
        yield f"data: {chunk}\n\n"
    
    yield "data: [DONE]\n\n"

# --- 4. Endpoints ---
# Accept a validated query and return a streaming HTTP response; the response body is the generated SSE stream.
@app.post("/api/query")
async def rag_query_endpoint(payload: QueryRequest):
    if not RETRIEVER:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    return StreamingResponse(
        generate_rag_stream(payload.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    # Uvicorn serves this FastAPI app when the file is run directly.
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

# ============================================================
# REVISION NOTES — dummy.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file defines a FastAPI backend that loads a local speech document, indexes it, and answers questions using retrieval-augmented generation.
# Its query endpoint streams the generated answer to the client using Server-Sent Events.
#
# KEY CONCEPTS USED:
# • Retrieval-augmented generation (RAG): find relevant source text and provide it to a language model as answer context.
# • Embeddings and vector store: represent text as vectors and search for passages with meaning similar to a query.
# • Server-Sent Events (SSE): stream text updates to a client over an HTTP response.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file combines the project's LangChain document loading, splitting, embedding, retrieval, and model examples in a web API.
#
# THINGS TO REMEMBER:
# • The startup event must initialize the global retriever before questions can be answered.
# • The source speech.txt path is resolved relative to this file first, with a local-path fallback.
# • OpenAI credentials must be available in the environment for embeddings and chat generation.
#
# COMMON MISTAKES TO AVOID:
# • Calling the query endpoint before the startup pipeline has created the retriever.
# • Treating Server-Sent Event data as ordinary unframed text; each event uses the data: prefix and blank-line separator.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow generate_rag_stream: it retrieves context, builds the prompt chain, and yields the answer a piece at a time.
# ============================================================