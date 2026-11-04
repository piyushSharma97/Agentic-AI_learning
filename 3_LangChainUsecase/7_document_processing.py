"""
=============================================================================
USE CASE 7: DOCUMENT PROCESSING & RAG IN LANGCHAIN
=============================================================================
Document Processing is the foundation of Retrieval-Augmented Generation (RAG).
It turns unstructured company knowledge (PDFs, docs, wiki pages) into searchable,
semantically indexed vectors that ground LLM responses with factual data.

Key Concepts Covered:
1. Document Objects: Structure with `page_content` and `metadata`
2. Text Splitting: RecursiveCharacterTextSplitter with chunk_size & chunk_overlap
3. Vector Stores & Embeddings: Indexing knowledge into InMemoryVectorStore
4. Similarity Search & Relevance Filtering
5. End-to-End RAG Chain with Source Citations
=============================================================================
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import shared LLM config
from config import get_llm, safe_execute


def get_embeddings():
    """
    Returns an embedding model:
    - Uses OpenAIEmbeddings if available and network permits.
    - Otherwise falls back to deterministic FakeEmbeddings so code runs locally anytime.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            # Quick check if reachable
            embeddings.embed_query("test")
            return embeddings
        except Exception:
            pass
    from langchain_core.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)


# ---------------------------------------------------------------------------
# 1. Document Ingestion & Chunking
# ---------------------------------------------------------------------------
def demo_chunking_and_splitting():
    print("--- 1. Document Chunking & Text Splitting ---")
    
    # Raw company policy text
    raw_policy_text = """
    ACME CORP REMOTE WORK & HARDWARE POLICY (2026 EDITION)
    
    Section 1: Work Hours & Availability
    All full-time engineers are expected to maintain core availability between 10:00 AM
    and 3:00 PM EST. Asynchronous communication via Slack and Jira is encouraged outside
    these core hours. All client-facing meetings must be scheduled at least 24 hours in advance.
    
    Section 2: Equipment Stipend
    Every new hire receives a one-time $1,500 equipment stipend for home office setup.
    Eligible purchases include ergonomic chairs, sit-stand desks, monitors, and noise-cancelling
    headphones. Expense claims must be submitted via Concur within 45 days of hire.
    
    Section 3: Annual Learning & Conference Budget
    Employees are allotted an annual budget of $2,000 for conferences, professional certifications,
    and technical books. Prior manager approval is required for all international events.
    """
    
    # Text splitter configured with chunk_size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.create_documents(
        texts=[raw_policy_text],
        metadatas=[{"source": "acme_hr_policy_2026.pdf", "version": "2.4"}]
    )
    
    print(f"Total chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[Chunk {i}] ({len(chunk.page_content)} chars | Metadata: {chunk.metadata}):")
        print(f"\"{chunk.page_content.strip()}\"")
    print()
    return chunks


# ---------------------------------------------------------------------------
# 2. Vector Store Indexing and Semantic Retrieval
# ---------------------------------------------------------------------------
def demo_vector_store_retrieval(chunks: List[Document]):
    print("--- 2. Vector Store Indexing & Similarity Search ---")
    
    embeddings = get_embeddings()
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    query = "What is the budget for home office equipment?"
    print(f"Search Query: '{query}'")
    
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    print(f"\nRetrieved Top {len(retrieved_docs)} Documents:")
    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"Result {idx} from [{doc.metadata.get('source')}]:\n{doc.page_content.strip()}\n")
        
    return vectorstore


# ---------------------------------------------------------------------------
# 3. End-to-End RAG Pipeline (Retriever + Prompt + LLM)
# ---------------------------------------------------------------------------
def demo_end_to_end_rag(vectorstore: InMemoryVectorStore):
    print("--- 3. End-to-End RAG Pipeline with Source Citations ---")
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for d in docs:
            formatted.append(f"[Source: {d.metadata.get('source')}]:\n{d.page_content}")
        return "\n\n".join(formatted)
        
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an internal corporate policy advisor. "
            "Answer the user question using ONLY the provided context snippets. "
            "If the answer cannot be found in the context, state clearly that policy does not mention it. "
            "Always cite the source document name."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # Modern LCEL RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    test_question = "How much can I spend on a home office desk, and how do I submit the expense?"
    print(f"User Question: {test_question}\n")
    
    answer = rag_chain.invoke(test_question)
    print(f"RAG Grounded Response:\n{answer}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Document Processing & RAG Demonstrations", lambda: [
        chunks := demo_chunking_and_splitting(),
        vs := demo_vector_store_retrieval(chunks),
        demo_end_to_end_rag(vs),
    ])


if __name__ == "__main__":
    main()
