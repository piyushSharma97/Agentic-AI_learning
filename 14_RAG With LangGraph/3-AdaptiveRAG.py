r"""
=============================================================================
ADAPTIVE RAG
=============================================================================

Adaptive RAG chooses a path for each question.

    START
      |
    route_question
      |                         \
      | vectorstore              \ web_search
      v                           v
    retrieve                   web_search
      |                           |
    grade_documents               |
      |                           |
      +-- no relevant notes --> transform_query --> retrieve
      |
      +-- relevant notes -----> generate <----------+
                                  |                  |
                                  +-- not grounded --+
                                  |
                                  +-- does not answer the question --> transform_query
                                  |
                                  +-- grounded and useful --> END

The notebook sections below are separate examples. Each one can run on
its own. main() calls all of them. Comment out a call to skip that example.

The lecture downloads three live blog posts into FAISS. faiss is not
installed here, so the notes are local and the store is InMemoryVectorStore.
The router still has the same job: agent, prompt, and attack questions
stay in the notes. Other questions go to the web.

Run:
    python "14_RAG With LangGraph/3-AdaptiveRAG.py"
=============================================================================
"""

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_embeddings, get_llm, print_banner, safe_run


# Topics the router is told live in the vector store.
# "What is machine learning?" and a sports question are outside these notes,
# so those examples should be sent to web search.
NOTES = [
    (
        "agent-memory",
        "Agent memory is described in three types. Sensory memory is the immediate input, "
        "such as the current user message. Short-term memory is the running conversation, "
        "usually the message list sent back to the model on the next call. Long-term memory "
        "is an external store, such as a vector index of past tasks, retrieved when a new "
        "task looks related. Writing every thought into long-term memory makes later retrieval noisy.",
    ),
    (
        "prompt-engineering",
        "Prompt engineering is how the instruction is written. Zero-shot asks with no examples. "
        "Few-shot adds examples of the desired output. Chain-of-thought asks the model to write "
        "intermediate steps before the answer. A system prompt holds standing rules. The human "
        "prompt carries the task for this turn.",
    ),
    (
        "adversarial-attacks",
        "Adversarial attacks try to make a model ignore its instructions. Prompt injection hides "
        "a new instruction inside retrieved text or a user message. A jailbreak coaxes the model "
        "past a safety rule. One defense is to keep untrusted text separate from the system "
        "instruction and to grade retrieved text before treating it as fact.",
    ),
]


class RouteQuery(BaseModel):
    """The router must pick one data source and nothing else."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description="vectorstore for agents, prompts, and attacks; web_search otherwise"
    )


class GradeDocuments(BaseModel):
    """yes means this chunk is worth passing to the answer step."""

    binary_score: Literal["yes", "no"]


class GradeHallucinations(BaseModel):
    """yes means the answer stays inside the retrieved facts."""

    binary_score: Literal["yes", "no"]


class GradeAnswer(BaseModel):
    """yes means the answer actually responds to the question."""

    binary_score: Literal["yes", "no"]


class GraphState(TypedDict):
    """Shared state. A node returns only the keys it changes."""

    question: str
    generation: str
    documents: list
    rewrites: int
    attempts: int


_retriever = None
_chains = None


def score_of(result) -> str:
    """Structured output is a pydantic model. Some providers return a dict."""
    if isinstance(result, dict):
        return str(result.get("binary_score", "no")).lower()
    return str(result.binary_score).lower()


def route_of(result) -> str:
    if isinstance(result, dict):
        return result.get("datasource", "vectorstore")
    return result.datasource


def format_docs(documents) -> str:
    """The answer prompt needs text. Documents may arrive as one item or a list."""
    if documents is None:
        return ""
    if isinstance(documents, Document):
        documents = [documents]
    return "\n\n".join(doc.page_content for doc in documents)


def format_search(payload) -> str:
    """Tavily returns a dict with a results list. The generator needs one string."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(item.get("content", str(item)) for item in payload)
    if isinstance(payload, dict):
        blocks = []
        for item in payload.get("results") or []:
            blocks.append(f"{item.get('title', '')}\n{item.get('content', '')}")
        return "\n\n".join(blocks) if blocks else str(payload)
    return str(payload)


def get_retriever():
    """
    Build the index once, then reuse it.

    The first example that retrieves pays for the embeddings.
    The router example does not call this, so it does not embed anything.
    """
    global _retriever
    if _retriever is not None:
        return _retriever

    documents = [
        Document(page_content=text, metadata={"source": source})
        for source, text in NOTES
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=40).split_documents(documents)
    print(f"[index] {len(documents)} notes -> {len(chunks)} chunks")
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    _retriever = store.as_retriever(search_kwargs={"k": 2})
    return _retriever


def get_chains():
    """
    Build each prompt chain once.

    The lecture creates a new ChatOpenAI for every chain. One chat model
    is enough. Each chain still has its own prompt and its own schema.
    """
    global _chains
    if _chains is not None:
        return _chains

    llm = get_llm(temperature=0)

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You route a user question to a vectorstore or web search. "
         "The vectorstore contains notes on agents, prompt engineering, and adversarial attacks. "
         "Use the vectorstore for those topics. Otherwise use web_search."),
        ("human", "{question}"),
    ])
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You grade one retrieved document. "
         "Score yes when it contains keywords or meaning related to the question. "
         "This filter removes the wrong chunk. It is not a strict exam. "
         "Reply with yes or no."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    # The lecture loads this text with hub.pull("rlm/rag-prompt").
    # The instruction is stored here so the example runs without LangChain Hub.
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer from the context only. Use at most three sentences. "
         "If the context does not contain the answer, say you do not know."),
        ("human", "Question: {question}\n\nContext: {context}"),
    ])
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You grade whether an answer is grounded in the retrieved facts. "
         "yes means every claim is supported by those facts."),
        ("human", "Facts:\n\n{documents}\n\nAnswer: {generation}"),
    ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You grade whether an answer resolves the question. "
         "yes means the question is answered. no means it missed the question."),
        ("human", "Question:\n\n{question}\n\nAnswer: {generation}"),
    ])
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You rewrite a question so a vector search can find the right notes. "
         "Keep the same intent. Return only the rewritten question."),
        ("human", "Initial question:\n\n{question}"),
    ])

    _chains = {
        "router": router_prompt | llm.with_structured_output(RouteQuery),
        "retrieval_grader": grade_prompt | llm.with_structured_output(GradeDocuments),
        "rag": rag_prompt | llm | StrOutputParser(),
        "hallucination_grader": hallucination_prompt | llm.with_structured_output(GradeHallucinations),
        "answer_grader": answer_prompt | llm.with_structured_output(GradeAnswer),
        "rewriter": rewrite_prompt | llm | StrOutputParser(),
    }
    return _chains


# ---------------------------------------------------------------------------
# Graph nodes. The example functions further down call these through the app.
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """Vector search. This node runs only after the router picks the notes."""
    print("---RETRIEVE---")
    documents = get_retriever().invoke(state["question"])
    for doc in documents:
        print(f"  hit: {doc.metadata.get('source')}")
    return {"documents": documents}


def generate(state: GraphState) -> dict:
    """Write an answer from whatever documents are in state: notes or web text."""
    print("---GENERATE---")
    generation = get_chains()["rag"].invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"],
    })
    return {"generation": generation, "attempts": state.get("attempts", 0) + 1}


def grade_documents(state: GraphState) -> dict:
    """Drop chunks the grader marks as unrelated. An empty list changes the route."""
    print("---CHECK DOCUMENT RELEVANCE---")
    kept = []
    for doc in state["documents"]:
        score = get_chains()["retrieval_grader"].invoke({
            "question": state["question"],
            "document": doc.page_content,
        })
        if score_of(score) == "yes":
            print(f"---GRADE: RELEVANT ({doc.metadata.get('source')})---")
            kept.append(doc)
        else:
            print(f"---GRADE: NOT RELEVANT ({doc.metadata.get('source')})---")
    return {"documents": kept}


def transform_query(state: GraphState) -> dict:
    """Rewrite the question once, then retrieve again with the new wording."""
    print("---TRANSFORM QUERY---")
    better_question = get_chains()["rewriter"].invoke({"question": state["question"]})
    print(f"  rewritten: {better_question}")
    return {"question": better_question.strip(), "rewrites": state.get("rewrites", 0) + 1}


def web_search(state: GraphState) -> dict:
    """
    Replace the document list with web results.

    This path did not retrieve notes, so there is nothing to append to.
    The current Tavily tool returns a dict. The lecture's TavilySearchResults
    returned a list of dicts. format_search accepts both.
    """
    print("---WEB SEARCH---")
    import os

    if not os.getenv("TAVILY_API_KEY"):
        print("  TAVILY_API_KEY is missing.")
        text = "Web search is unavailable."
    else:
        from langchain_tavily import TavilySearch

        payload = TavilySearch(max_results=3).invoke({"query": state["question"]})
        text = format_search(payload)[:2000]
    return {"documents": [Document(page_content=text, metadata={"source": "tavily"})]}


def route_question(state: GraphState) -> str:
    """First decision: notes or the web. This runs from START."""
    print("---ROUTE QUESTION---")
    source = route_of(get_chains()["router"].invoke({"question": state["question"]}))
    if source == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"


def decide_to_generate(state: GraphState) -> str:
    """
    Second decision, after grading.

    Relevant chunks go to generate. An empty list rewrites the question
    one time. A second miss still generates, so retrieve and rewrite
    cannot cycle forever.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        if state.get("rewrites", 0) >= 1:
            print("---DECISION: REWRITE ALREADY USED, GENERATE ANYWAY---")
            return "generate"
        print("---DECISION: NO RELEVANT DOCUMENTS, TRANSFORM QUERY---")
        return "transform_query"
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation(state: GraphState) -> str:
    """
    Third decision, after the answer is written.

    Grounded and on-topic ends the graph. A grounded answer that misses
    the question is rewritten. An ungrounded answer is generated once more,
    then the graph stops.
    """
    print("---CHECK HALLUCINATIONS---")
    grounded = score_of(get_chains()["hallucination_grader"].invoke({
        "documents": format_docs(state["documents"]),
        "generation": state["generation"],
    }))
    if grounded != "yes":
        if state.get("attempts", 1) >= 2:
            print("---DECISION: NOT GROUNDED, STOP AFTER RETRY---")
            return "useful"
        print("---DECISION: NOT GROUNDED, RE-TRY---")
        return "not supported"

    print("---DECISION: GROUNDED IN DOCUMENTS---")
    print("---GRADE GENERATION vs QUESTION---")
    answers = score_of(get_chains()["answer_grader"].invoke({
        "question": state["question"],
        "generation": state["generation"],
    }))
    if answers == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
    return "not useful"


def build_app():
    """Connect the nodes. Call this from the graph examples, not at import."""
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START,
        route_question,
        {"web_search": "web_search", "vectorstore": "retrieve"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"},
    )
    return workflow.compile()


def ask_graph(question: str) -> None:
    """Run one question through the full adaptive graph and print the answer."""
    print("\n" + "=" * 72)
    print("QUESTION:", question)
    result = build_app().invoke(
        {
            "question": question,
            "generation": "",
            "documents": [],
            "rewrites": 0,
            "attempts": 0,
        },
        {"recursion_limit": 12},
    )
    print("\nANSWER:")
    print(result["generation"])


# ---------------------------------------------------------------------------
# Examples. Each function is one notebook section.
# Comment out its call in main() to skip it.
# ---------------------------------------------------------------------------

def example_route_to_web_or_notes() -> None:
    """
    Example 1 — router only.

    The router reads the question and returns vectorstore or web_search.
    It does not retrieve and it does not answer.
    Agent memory is in the notes. A cricket result is not.
    """
    print("\n### Example 1: route the question")
    router = get_chains()["router"]
    for question in (
        "Who won the Cricket World Cup 2023?",
        "What are the types of agent memory?",
    ):
        decision = route_of(router.invoke({"question": question}))
        print(f"  {question}\n  -> {decision}")


def example_grade_a_retrieved_chunk() -> None:
    """
    Example 2 — retrieval grader.

    Retrieve "agent memory" and grade the top chunk.
    yes means the chunk may be shown to the answer model.
    """
    print("\n### Example 2: grade one retrieved chunk")
    docs = get_retriever().invoke("agent memory")
    top = docs[0]
    print(f"  source: {top.metadata.get('source')}")
    grade = get_chains()["retrieval_grader"].invoke({
        "question": "agent memory",
        "document": top.page_content,
    })
    print(f"  grade: {score_of(grade)}")


def example_generate_from_notes() -> None:
    """
    Example 3 — answer from retrieved notes.

    This is the generate step by itself, before it is placed in the graph.
    """
    print("\n### Example 3: generate an answer from the notes")
    question = "What are the types of agent memory?"
    docs = get_retriever().invoke(question)
    answer = get_chains()["rag"].invoke({
        "context": format_docs(docs),
        "question": question,
    })
    print(answer)


def example_grade_hallucination() -> None:
    """
    Example 4 — hallucination grader.

    yes means the answer is supported by the notes.
    An answer that adds facts the notes never stated should score no.
    """
    print("\n### Example 4: is the answer grounded in the notes?")
    question = "What are the types of agent memory?"
    docs = get_retriever().invoke(question)
    answer = get_chains()["rag"].invoke({
        "context": format_docs(docs),
        "question": question,
    })
    grade = get_chains()["hallucination_grader"].invoke({
        "documents": format_docs(docs),
        "generation": answer,
    })
    print(answer)
    print(f"  grounded: {score_of(grade)}")


def example_grade_answer() -> None:
    """
    Example 5 — answer grader.

    This check is different from the hallucination check.
    A grounded paragraph can still fail here if it never answers the question.
    """
    print("\n### Example 5: does the answer address the question?")
    question = "What are the types of agent memory?"
    docs = get_retriever().invoke(question)
    answer = get_chains()["rag"].invoke({
        "context": format_docs(docs),
        "question": question,
    })
    grade = get_chains()["answer_grader"].invoke({
        "question": question,
        "generation": answer,
    })
    print(answer)
    print(f"  addresses question: {score_of(grade)}")


def example_rewrite_question() -> None:
    """
    Example 6 — question rewriter.

    The graph uses this when the retrieved notes are all graded irrelevant.
    A clearer query is sent back to retrieve.
    """
    print("\n### Example 6: rewrite a question for vector search")
    question = "agent memory"
    rewritten = get_chains()["rewriter"].invoke({"question": question})
    print(f"  original:  {question}")
    print(f"  rewritten: {rewritten.strip()}")


def example_graph_question_outside_the_notes() -> None:
    """
    Example 7 — full graph, web path.

    Machine learning is not one of the three note topics.
    The router should choose web_search, then generate from those results.
    """
    print("\n### Example 7: full graph, question outside the notes")
    ask_graph("What is machine learning?")


def example_graph_question_inside_the_notes() -> None:
    """
    Example 8 — full graph, notes path.

    Agent memory is in the index. The router should choose the vector store,
    grade the chunks, then generate.
    """
    print("\n### Example 8: full graph, question inside the notes")
    ask_graph("What are the types of agent memory?")


def main() -> None:
    """
    Run every example from top to bottom.

    Put a # in front of a call to skip that example.
    Example 1 does not build the index. Examples 2 to 6 do.
    Examples 7 and 8 run the whole graph, including web search in example 7.
    """
    print_banner(
        "ADAPTIVE RAG",
        "Eight examples. Comment out a call in main() to skip it.",
    )

    example_route_to_web_or_notes()
    example_grade_a_retrieved_chunk()
    example_generate_from_notes()
    example_grade_hallucination()
    example_grade_answer()
    example_rewrite_question()
    example_graph_question_outside_the_notes()
    example_graph_question_inside_the_notes()


if __name__ == "__main__":
    safe_run("Adaptive RAG", main)
