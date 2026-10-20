import os
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Production RAG Backend", version="1.0.0")

# --- 1. Global Instances & Startup Pipeline ---
VECTOR_DB = None
RETRIEVER = None

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

    # Load and split
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # Embed and initialize in-memory Chroma
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    VECTOR_DB = Chroma.from_documents(chunks, embeddings)
    RETRIEVER = VECTOR_DB.as_retriever(search_kwargs={"k": 3})
    print(f"RAG initialized with {len(chunks)} chunks.")

# --- 2. Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")

# --- 3. Streaming Engine ---
async def generate_rag_stream(query: str) -> AsyncGenerator[str, None]:
    # 1. Retrieve relevant chunks
    docs = RETRIEVER.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Build Prompt & LCEL Chain
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

    # 3. Stream tokens via SSE
    async for chunk in chain.astream({"context": context_text, "question": query}):
        # Format as Server-Sent Event data block
        yield f"data: {chunk}\n\n"
    
    yield "data: [DONE]\n\n"

# --- 4. Endpoints ---
@app.post("/api/query")
async def rag_query_endpoint(payload: QueryRequest):
    if not RETRIEVER:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    return StreamingResponse(
        generate_rag_stream(payload.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)