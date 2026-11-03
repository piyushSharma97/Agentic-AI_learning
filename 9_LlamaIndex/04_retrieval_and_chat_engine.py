# ============================================================
# LLAMAINDEX 104: RETRIEVAL VS. QUERY ENGINE VS. CHAT ENGINE
# ============================================================
#
# Three Interaction Layers in LlamaIndex:
# ---------------------------------------
# 1. Retriever (`index.as_retriever`):
#    - Pure semantic search.
#    - Input: Query string -> Output: List of NodeWithScore (chunks).
#    - DOES NOT CALL AN LLM. Zero LLM cost!
#    - Essential for evaluating retrieval quality, reranking, or custom pipelines.
#
# 2. Query Engine (`index.as_query_engine`):
#    - One-shot question answering over your data.
#    - Pipeline: Query -> Retriever -> LLM Synthesis -> Answer.
#    - Stateless: Does NOT remember previous questions.
#
# 3. Chat Engine (`index.as_chat_engine`):
#    - Multi-turn conversation over your data with conversational memory.
#    - Understands follow-up questions (e.g. "Can you elaborate on that budget?").
#    - Automatically manages message history and context condensation.
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


def load_sample_index():
    """Helper to load and index the sample company handbook."""
    data_dir = Path(__file__).parent / "data"
    reader = SimpleDirectoryReader(input_dir=str(data_dir))
    documents = reader.load_data()
    return VectorStoreIndex.from_documents(documents)


def example_1_pure_retriever(index):
    """
    EXAMPLE 1: Pure Retriever (No LLM generation).
    Inspect raw semantic matching and similarity scores.
    """
    print("\n" + "=" * 65)
    print("1. PURE RETRIEVER (Zero LLM calls, Semantic Search Only)")
    print("=" * 65)

    # similarity_top_k specifies how many top relevant chunks to retrieve
    retriever = index.as_retriever(similarity_top_k=2)

    query = "What happens with sick leave?"
    print(f"Retrieving chunks for query: '{query}'\n")

    # retrieve() returns a list of NodeWithScore
    matched_nodes = retriever.retrieve(query)

    print(f"Retrieved {len(matched_nodes)} relevant node(s):")
    for i, node_with_score in enumerate(matched_nodes, start=1):
        score = node_with_score.score
        score_display = f"{score:.4f}" if score is not None else "N/A"
        print(f"\n[Result {i}] Score: {score_display}")
        print(f"Content:\n{node_with_score.node.get_text().strip()}")
        print("-" * 40)


def example_2_query_engine(index):
    """
    EXAMPLE 2: Query Engine (Single-turn Q&A).
    Stateless: Each call is independent.
    """
    print("\n" + "=" * 65)
    print("2. QUERY ENGINE (One-shot Q&A)")
    print("=" * 65)

    query_engine = index.as_query_engine(similarity_top_k=2)

    # Question 1
    q1 = "Who founded TechFlow and when?"
    print(f"Question 1: {q1}")
    res1 = query_engine.query(q1)
    print("Answer 1:", res1.response)

    # Question 2 (Notice: Query engine has NO memory of Question 1)
    q2 = "Where was she located?"  # 'she' refers to Sarah Jenkins, but query engine doesn't know!
    print(f"\nQuestion 2: {q2}")
    res2 = query_engine.query(q2)
    print("Answer 2:", res2.response)


def example_3_chat_engine(index):
    """
    EXAMPLE 3: Chat Engine (Multi-turn conversational RAG).
    Maintains conversational state across turns to answer follow-up queries.
    """
    print("\n" + "=" * 65)
    print("3. CHAT ENGINE (Multi-turn Conversational RAG with Memory)")
    print("=" * 65)

    # chat_mode options:
    # - "condense_plus_context": (Recommended default) Rewrites user query
    #   using chat history, retrieves context, and responds with full context.
    # - "context": Always retrieves context and appends history.
    # - "simple": Standard chatbot without document retrieval.
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=2,
        system_prompt=(
            "You are an HR assistant for TechFlow Innovations. "
            "Always answer based strictly on the provided company handbook."
        )
    )

    # Turn 1:
    print("User: Who is the founder of TechFlow?")
    response_1 = chat_engine.chat("Who is the founder of TechFlow?")
    print(f"Bot:  {response_1.response}\n")

    # Turn 2: Follow-up relying on context from Turn 1!
    print("User: Where did she establish the headquarters?")
    response_2 = chat_engine.chat("Where did she establish the headquarters?")
    print(f"Bot:  {response_2.response}\n")

    # Turn 3: Follow-up about policy
    print("User: What is their parental leave benefit?")
    response_3 = chat_engine.chat("What is their parental leave benefit?")
    print(f"Bot:  {response_3.response}\n")


def main():
    print("=" * 65)
    print("LLAMAINDEX INTRO PRACTICE: RETRIEVAL VS CHAT ENGINE")
    print("=" * 65)

    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] Set OPENAI_API_KEY in .env to run query and chat engines.")
        return

    # Load data once
    index = load_sample_index()

    # 1. Test pure retriever (no LLM generation)
    example_1_pure_retriever(index)

    # 2. Test one-shot query engine
    # example_2_query_engine(index)

    # 3. Test multi-turn conversational chat engine
    # example_3_chat_engine(index)

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

