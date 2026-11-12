"""
=============================================================================
CORRECTIVE RAG
=============================================================================

Agentic RAG lets the model choose a tool. Corrective RAG always retrieves
first, then checks the chunks. A bad retrieval is corrected with a web search
before the answer is written.

    START
      |
    retrieve          vector search over the local agent notes
      |
    grade             keep the chunks that can answer the question
      |
      +-- all kept chunks are enough --> generate --> END
      |
      +-- a chunk was irrelevant -----> web_search --> generate --> END

The web branch runs when any retrieved chunk fails the grade. Relevant
chunks are kept. The search result is appended as one more document.
Generate then sees both.

State here is a question plus documents, not a chat history. There is no
tool-calling agent in this graph. The edges are fixed.

Run:
    python "14_RAG With LangGraph/2-CorrectiveRAG.py"
=============================================================================
"""

import os
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm, print_banner, print_mermaid, safe_run
from corpus import AGENT_NOTES, build_retriever, preview_retrieval
from grading import is_relevant


class CRAGState(TypedDict):
    question: str
    documents: list[Document]
    needs_web_search: bool
    generation: str


def format_search(payload) -> str:
    """Tavily returns a dict. The generator needs a single string."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return str(payload)
    blocks = []
    for item in payload.get("results") or []:
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        blocks.append(f"{title}\n{content}\n{url}")
    return "\n\n".join(blocks) if blocks else str(payload)


def build_graph():
    llm = get_llm(temperature=0)
    # k=1 keeps the demo on one chunk. The agent question then stays on the
    # notes. The population question fails that chunk and opens web search.
    # The lecture turns web search on if any chunk in a larger k fails.
    retriever = build_retriever(AGENT_NOTES, "agents", k=1)

    def retrieve(state: CRAGState) -> dict:
        print("[node] retrieve")
        docs = retriever.invoke(state["question"])
        for doc in docs:
            print(f"  hit: {doc.metadata.get('source')}")
        return {"documents": docs}

    def grade(state: CRAGState) -> dict:
        """
        Drop chunks that cannot answer the question.
        One rejected chunk is enough to turn web search on. The useful
        chunks stay in the list and are still passed to generate.
        """
        print("[node] grade")
        kept: list[Document] = []
        rejected = False
        for doc in state["documents"]:
            source = doc.metadata.get("source", "notes")
            print(f"  checking {source}")
            if is_relevant(llm, state["question"], doc.page_content):
                kept.append(doc)
            else:
                rejected = True
        print(f"[node] kept {len(kept)} of {len(state['documents'])} chunks")
        return {"documents": kept, "needs_web_search": rejected or not kept}

    def route_after_grade(state: CRAGState) -> Literal["web_search", "generate"]:
        if state["needs_web_search"]:
            print("[route] retrieval is incomplete -> web_search")
            return "web_search"
        print("[route] local notes are enough -> generate")
        return "generate"

    def web_search(state: CRAGState) -> dict:
        print("[node] web_search")
        if not os.getenv("TAVILY_API_KEY"):
            print("[node] TAVILY_API_KEY is missing. Continuing with the local notes only.")
            return {"documents": state["documents"]}

        from langchain_tavily import TavilySearch

        payload = TavilySearch(max_results=3).invoke({"query": state["question"]})
        web_doc = Document(
            page_content=format_search(payload)[:2000],
            metadata={"source": "tavily"},
        )
        # This field has no reducer, so return the whole list generate should see.
        return {"documents": [*state["documents"], web_doc]}

    def generate(state: CRAGState) -> dict:
        print("[node] generate")
        context = "\n\n".join(
            f"[{doc.metadata.get('source', 'notes')}]\n{doc.page_content}"
            for doc in state["documents"]
        ) or "No documents were retrieved."
        response = llm.invoke([
            SystemMessage(content=(
                "Answer in a short paragraph from the context only. "
                "Mention when a point came from the web search."
            )),
            HumanMessage(content=f"Question:\n{state['question']}\n\nContext:\n{context}"),
        ])
        return {"generation": response.content.strip()}

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"web_search": "web_search", "generate": "generate"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    preview_retrieval(retriever, "What is the planner in an LLM agent?")
    return graph.compile()


def run_demo() -> None:
    print_banner(
        "CORRECTIVE RAG",
        "Retrieve, grade each chunk, and search the web when the local notes are not enough.",
    )
    graph = build_graph()
    print_mermaid(graph, "Corrective RAG")

    questions = [
        "What is the planner in an LLM agent, and when does the loop stop?",
        "What is the population of Japan right now?",
    ]
    for question in questions:
        print("\n" + "=" * 72)
        print("QUESTION:", question)
        result = graph.invoke({
            "question": question,
            "documents": [],
            "needs_web_search": False,
            "generation": "",
        })
        print(f"web search used: {result['needs_web_search']}")
        print("\nANSWER:")
        print(result["generation"])


if __name__ == "__main__":
    safe_run("Corrective RAG", run_demo)
