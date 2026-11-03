# ============================================================
# LLAMAINDEX 102: DOCUMENTS, NODES, AND CHUNKING
# ============================================================
#
# Core Concept: Document vs. Node
# -------------------------------
# - Document: A container around a full data source (e.g., an entire PDF,
#             a 20-page document, a database row, or a web article).
# - Node: The atomic, first-class unit of data in LlamaIndex.
#         It is a "chunk" of a Document that gets converted into an embedding
#         and retrieved by the query engine.
#
# Why Nodes Matter:
# 1. LLMs have context window limits; you cannot pass entire books.
# 2. Smaller, focused chunks improve search precision (better similarity).
# 3. Nodes preserve relationships: each Node knows its Parent Document,
#    Previous Node, and Next Node.
#
# Topics covered:
# 1. Creating Documents with custom metadata & doc_ids
# 2. Splitting Documents into Nodes using SentenceSplitter
# 3. Controlling chunk_size and chunk_overlap
# 4. Inspecting Node relationships and metadata inheritance
# 5. Building an index directly from Nodes (instead of raw documents)
# ============================================================

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship


def example_1_documents_and_metadata():
    """
    Documents hold raw text and arbitrary metadata (key-value pairs).
    Metadata can be used for filtering during retrieval (e.g. only search HR docs).
    """
    print("\n" + "=" * 65)
    print("1. CREATING DOCUMENTS WITH METADATA")
    print("=" * 65)

    doc = Document(
        text="TechFlow provides 16 weeks of 100% paid parental leave for caregivers.",
        metadata={
            "document_type": "handbook",
            "department": "Human Resources",
            "year": 2026,
            "version": "v2.1"
        },
        id_="doc_hr_parental_leave_001"  # Custom unique ID
    )

    print(f"Document ID: {doc.doc_id}")
    print(f"Document Text: {doc.text}")
    print(f"Metadata: {doc.metadata}")
    print(f"Metadata excluded from LLM: {doc.excluded_llm_metadata_keys}")


def example_2_chunking_with_sentence_splitter():
    """
    SentenceSplitter breaks text along natural sentence boundaries
    while respecting character/token limits.
    """
    print("\n" + "=" * 65)
    print("2. CHUNKING DOCUMENTS INTO NODES")
    print("=" * 65)

    sample_long_text = (
        "Artificial Intelligence has rapidly evolved from rule-based systems to modern generative models. "
        "Large Language Models are trained on massive datasets using transformer architectures. "
        "Retrieval-Augmented Generation (RAG) combines pre-trained language models with external retrieval. "
        "This allows models to cite sources and access proprietary or up-to-date business documents. "
        "LlamaIndex simplifies data ingestion, node parsing, embedding creation, and semantic search. "
        "By structuring documents into clean nodes, search engines achieve much higher retrieval precision."
    )

    doc = Document(text=sample_long_text, metadata={"source": "ai_overview.txt"})

    # SentenceSplitter parameters:
    # - chunk_size: Approximate token count per chunk.
    # - chunk_overlap: Number of tokens shared between consecutive chunks
    #   (ensures ideas crossing chunk boundaries aren't severed).
    splitter = SentenceSplitter(chunk_size=35, chunk_overlap=10)

    # get_nodes_from_documents parses a list of Documents into Nodes
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"Original text length: {len(sample_long_text)} characters")
    print(f"Created {len(nodes)} nodes (chunks):\n")

    for idx, node in enumerate(nodes, start=1):
        print(f"--- Node {idx} [ID: {node.node_id[:8]}...] ---")
        print(f"Content: {node.get_text()}")
        print(f"Inherited Metadata: {node.metadata}")
        
        # Check node relationships
        if NodeRelationship.SOURCE in node.relationships:
            print(f"Parent Doc ID: {node.relationships[NodeRelationship.SOURCE].node_id}")
        if NodeRelationship.PREVIOUS in node.relationships:
            print(f"Prev Node ID:   {node.relationships[NodeRelationship.PREVIOUS].node_id[:8]}...")
        if NodeRelationship.NEXT in node.relationships:
            print(f"Next Node ID:   {node.relationships[NodeRelationship.NEXT].node_id[:8]}...")
        print()


def example_3_build_index_from_nodes():
    """
    You can build a VectorStoreIndex directly from Nodes.
    This gives you complete control over how text was split, cleaned, and enriched.
    """
    print("\n" + "=" * 65)
    print("3. BUILDING INDEX DIRECTLY FROM NODES")
    print("=" * 65)

    # Prepare document
    doc = Document(
        text=(
            "TechFlow Seattle office: 500 Pine Street, 4th Floor. "
            "Office hours: 8:00 AM - 6:00 PM PST. "
            "Visitors must register with reception on the ground floor."
        ),
        metadata={"facility": "Seattle HQ"}
    )

    # 1. Parse into nodes
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"Built {len(nodes)} nodes from document.")

    # 2. Build index from nodes (not documents!)
    index = VectorStoreIndex(nodes)

    # 3. Query
    query_engine = index.as_query_engine()
    response = query_engine.query("Where is the Seattle office located?")

    print("\nQuery: Where is the Seattle office located?")
    print("Answer:", response.response)


def main():
    print("=" * 65)
    print("LLAMAINDEX INTRO PRACTICE: DOCUMENTS & NODES")
    print("=" * 65)

    example_1_documents_and_metadata()
    example_2_chunking_with_sentence_splitter()
    # example_3_build_index_from_nodes()

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

