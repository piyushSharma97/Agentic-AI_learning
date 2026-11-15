"""
=============================================================================
LANGGRAPH ROUTER DEMO
=============================================================================

What this file is
-----------------
A graph that reads one user query and runs exactly one specialist path.
The other specialists stay idle for that query.

How one call works
------------------
graph.invoke({"query": "..."}) does this:

    1. LangGraph creates a state dict. At the start it only contains query.
    2. The START edge always enters the router node. START is not a function.
    3. router_node reads query, picks a label, and returns
       {"route": "...", "routing_reason": "..."}.
       LangGraph merges that dict into state. The old query stays.
    4. The conditional edge calls choose_destination(state).
       That function returns a label such as "rag". It does not run a node.
    5. PATH_MAP turns the label into a node name. For "workflow" the node
       is "validate". For every other label the node name matches the label.
    6. That one node runs and returns its own partial dict, usually
       {"result": "..."}.
    7. A normal edge then goes to END, or further along that path.
       END stops the graph. invoke() returns the whole state.

Concrete walk-through. Query: "What is our company leave policy?"

    START
      |  normal edge, always
      v
    router          route = "rag"
                    routing_reason = "answered from internal documents"
      |  conditional edge. choose_destination returns "rag"
      v
    rag             reads query + MOCK_DOCUMENTS, writes result
      |  normal edge, always
      v
    END             chat, web, database, support did not run

Picture of every path
---------------------

    START
      |
    router          writes route + routing_reason
      |
      +-- chat --------+
      +-- web ---------+
      +-- rag ---------+--> END
      +-- database ----+
      +-- support -----+
      |
      +-- validate --+-- execute --> summarize --> END
      |              |
      |              +--> END          query failed the check
      |
      +-- multi_agent --+--> researcher --+
                        +--> analyst -----+--> synthesizer --> END

Two kinds of edge
-----------------
Normal edge:       A --> B means "when A finishes, always run B".
Conditional edge:  after A, call a function. The string it returns selects
                   one next node from a path map. The other nodes do not run.

A node never chooses the next node by returning it. A node only returns
state fields. The edges, registered in build_graph(), choose what runs next.

Two routers
-----------
ROUTING_MODE in this folder's .env selects who writes the label.

- rule   Scan RULES from top to bottom. First keyword match wins.
         "charge me back" does not contain "refund", so this router misses it.
- llm    The model reads ROUTE_GUIDE and returns a RouteDecision.
         It can handle paraphrases. It costs one extra model call.

Run from the repo root:
    python langgraph-router-demo/main.py
    python langgraph-router-demo/main.py "What is our leave policy?"
=============================================================================
"""

import os
import operator
from pathlib import Path
from typing import Annotated, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict


# This folder's .env holds ROUTING_MODE.
# The repo-root .env holds GROQ_API_KEY.
# load_dotenv leaves a value alone when it is already set, so a variable you
# export in the shell beats both files, and this folder beats the repo root.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to the repo-root .env before running this demo."
    )

# temperature 0 asks the model to be stable. The same query should get the
# same route label on the next run.
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0.0,
    groq_api_key=_api_key,
)


# ---------------------------------------------------------------------------
# Labels the router is allowed to write
# ---------------------------------------------------------------------------
# Route is a typing allow-list, not a node. Structured output must pick one
# of these strings. Adding a specialist means touching all of these:
#   1. this Literal
#   2. ROUTE_GUIDE
#   3. RULES, if the keyword router should know it
#   4. a node function
#   5. builder.add_node(...)
#   6. PATH_MAP
# Missing step 6 is the usual bug: the model picks a label the graph cannot follow.

Route = Literal[
    "chat",
    "web",
    "rag",
    "database",
    "support",
    "workflow",
    "multi_agent",
]

# One description per label. The keyword router treats this as the meaning of
# each rule. The LLM router pastes this text into the prompt. One dict keeps
# the two routers from describing the same label differently.
ROUTE_GUIDE: dict[Route, str] = {
    "web": "Needs live or recent information: news, weather, today's price, or anything that goes stale.",
    "rag": "The answer should come from internal documents, a PDF, a handbook, or company policy.",
    "database": "Asks for stored records, counts, or other facts a database already holds.",
    "support": "A customer needs help with a refund, billing, an account, or a complaint.",
    "workflow": "The user wants a predefined process to be checked and then run, such as onboarding.",
    "multi_agent": "The question needs two perspectives, such as research plus a comparison.",
    "chat": "Ordinary conversation. Use this only when no specialist above fits.",
}

# Keyword router table. Each row is (label, words, reason).
# Matching is a lowercase substring test, and the first hit wins.
# "latest leave policy" matches web, because "latest" is in the first row,
# even though "policy" would also match rag.
# "vector databases" matches database, because "database" is inside the word,
# and that row sits above multi_agent.
RULES: list[tuple[Route, tuple[str, ...], str]] = [
    (
        "web",
        ("latest", "today", "current", "news", "weather"),
        "Query needs current information",
    ),
    (
        "rag",
        ("pdf", "document", "policy", "handbook"),
        "Query should be answered from internal documents",
    ),
    (
        "database",
        ("sql", "database", "records", "how many"),
        "Query asks for stored records",
    ),
    (
        "support",
        ("refund", "complaint", "support", "billing", "invoice"),
        "Customer support request",
    ),
    (
        "workflow",
        ("workflow", "onboarding"),
        "A predefined workflow was requested",
    ),
    (
        "multi_agent",
        ("research", "compare", "pros and cons"),
        "More than one perspective is useful",
    ),
]


class RouteDecision(BaseModel):
    """
    The object both routers return.

    This is not graph state. router_node copies route and reason into
    AgentState. Field descriptions are sent to the model with the schema,
    so they tell the model what to put in each field.
    """

    # The label. Must be one value from Route.
    route: Route = Field(description="Exactly one destination from the route guide.")
    # Shown in the terminal so you can see why that label was chosen.
    reason: str = Field(description="One sentence explaining why this destination fits.")


class SupportRequest(BaseModel):
    """
    The object support_node asks the model to fill.

    Same idea as RouteDecision: the model returns fields, and the node
    turns those fields into state. This class is not the graph state.
    """

    # Which human queue would own the ticket.
    category: Literal["technical", "billing", "account", "other"] = Field(
        description="Which queue should own this ticket."
    )
    # True means a person must take over. The node then writes an escalation message.
    escalate: bool = Field(
        description="True only for a legal threat, a safety issue, or a request to talk to a human."
    )
    reason: str = Field(description="One sentence for the category and the escalation choice.")


class AgentState(TypedDict, total=False):
    """
    The one dict that travels through the graph.

    total=False means invoke() may pass only query. The other fields appear
    when a node writes them. A node returns the fields it changed, not a
    full copy of this class.

    Who writes each field
    ---------------------
    query            the caller, in invoke()
    route            router_node
    routing_reason   router_node
    result           the specialist that actually ran
    category         support_node
    escalate         support_node
    validated        validate_node
    worker_results   researcher_node and analyst_node together
    """

    # The user text. Every node is allowed to read it. Nothing in the graph overwrites it.
    query: str
    # Label chosen for this query: "chat", "rag", "workflow", and so on.
    route: str
    # One sentence from the router. Useful when the label looks wrong.
    routing_reason: str
    # The answer the runner prints. Each specialist writes its own version.
    result: str
    # Support queue. Stays empty on every path except support.
    category: str
    # Support flag. True only when a human should take the ticket.
    escalate: bool
    # Workflow gate. route_after_validation reads this and nothing else.
    validated: bool
    # Notes from the two multi_agent workers.
    # operator.add is the reducer: if both workers return a list in the same
    # step, LangGraph concatenates the lists. A plain list would keep one
    # worker and drop the other. Those workers must not write `result`.
    worker_results: Annotated[list[str], operator.add]


def ask(system: str, human: str) -> str:
    """
    HELPER, not a node. Any node can call this. The graph does not know it exists.

    What: one model call that returns plain text.
    system: the role and the rules ("answer only from these documents").
    human:  the case ("Question: ... Documents: ...").
    Why split the prompt: the rules stay separate from the user text, so a
    question cannot as easily rewrite the rules.
    """
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])
    content = response.content
    return content.strip() if isinstance(content, str) else str(content).strip()


def routing_mode() -> Literal["rule", "llm"]:
    """
    HELPER, not a node. router_node calls this before it picks a router.

    Reads ROUTING_MODE, then ROUTER_MODE, then defaults to "llm".
    "rule" / "rules" / "rule_based" all mean the keyword router.
    "llm" / "model" both mean the model router.
    Anything else raises, so a typo does not silently pick the wrong router.
    """
    raw = (os.getenv("ROUTING_MODE") or os.getenv("ROUTER_MODE") or "llm").strip().lower()
    if raw in {"rule", "rules", "rule_based"}:
        return "rule"
    if raw in {"llm", "model"}:
        return "llm"
    raise ValueError(f"Invalid ROUTING_MODE={raw!r}. Use 'rule' or 'llm'.")


def decide_by_rules(query: str) -> RouteDecision:
    """
    HELPER used by router_node when ROUTING_MODE=rule. Not a graph node.

    Reads the query. Does not read or write AgentState.
    Returns a RouteDecision. router_node copies that into state.

    How: lowercase the query, walk RULES from top to bottom, return on the
    first word that appears anywhere in the text. No match returns chat.
    """
    text = query.lower()
    for route, words, reason in RULES:
        if any(word in text for word in words):
            return RouteDecision(route=route, reason=reason)
    return RouteDecision(route="chat", reason="No specialist keyword matched")


def decide_with_llm(query: str) -> RouteDecision:
    """
    HELPER used by router_node when ROUTING_MODE=llm. Not a graph node.

    with_structured_output(RouteDecision) tells the model to return that
    schema instead of a free-form paragraph. The ROUTE_GUIDE text is the
    instruction for which label fits which question.

    On a schema failure, return chat. Chat can still answer, and the
    conditional edge still has a real node to enter.
    """
    guide = "\n".join(f"- {name}: {when}" for name, when in ROUTE_GUIDE.items())
    structured = llm.with_structured_output(RouteDecision)
    try:
        decision = structured.invoke([
            SystemMessage(content=(
                "Route the user query to exactly one specialist.\n"
                "Prefer a specialist over chat.\n\n"
                f"{guide}"
            )),
            HumanMessage(content=query),
        ])
    except (ValidationError, ValueError) as exc:
        return RouteDecision(route="chat", reason=f"Router fell back to chat ({exc}).")

    if isinstance(decision, RouteDecision):
        return decision
    # Some versions return a plain dict. model_validate rebuilds the object.
    return RouteDecision.model_validate(decision)


def router_node(state: AgentState) -> dict:
    """
    NODE router
    Reads:  query
    Writes: route, routing_reason
    Next:   chosen by choose_destination + PATH_MAP, not by this return value

    This is the first node of every query. It only classifies. The answer
    is written later by whichever specialist the edge selects.
    """
    mode = routing_mode()
    if mode == "rule":
        decision = decide_by_rules(state["query"])
    else:
        decision = decide_with_llm(state["query"])

    print(f"\n[Router:{mode}] {decision.route}")
    print(f"[Router] {decision.reason}")
    return {"route": decision.route, "routing_reason": decision.reason}


def choose_destination(state: AgentState) -> Route:
    """
    CONDITIONAL EDGE FUNCTION after router. Not a node.

    LangGraph calls this after router_node has already merged its update.
    So state["route"] is visible here.

    Returns a key of PATH_MAP, such as "rag" or "workflow".
    It must not call the model and must not return a dict. A dict would be
    a state update, and this function's job is only to name the next step.

    An unknown label becomes "chat", which PATH_MAP sends to the chat node.
    """
    route = state.get("route", "chat")
    if route not in ROUTE_GUIDE:
        return "chat"
    return cast(Route, route)


# Label -> node name.
# chat/web/rag/database/support/multi_agent use the same string for both.
# workflow is different on purpose: the business label is "workflow", and
# the first node on that path is "validate". Execution comes after the check.
PATH_MAP = {
    "chat": "chat",
    "web": "web",
    "rag": "rag",
    "database": "database",
    "support": "support",
    "workflow": "validate",
    "multi_agent": "multi_agent",
}


# ---------------------------------------------------------------------------
# Specialist nodes.
# Each one reads query, writes result, then a normal edge goes to END.
# The lists below are stand-ins. A real web node would call a search API,
# a real rag node would query a vector store, and a real database node would
# run a read-only SQL query. The edges around them would stay the same.
# ---------------------------------------------------------------------------

def chat_node(state: AgentState) -> dict:
    """
    NODE chat
    Reads:  query
    Writes: result
    Next:   END, always

    General questions. This node has no documents, no database, and no ticket fields.
    """
    print("[Chat]")
    answer = ask(
        "You are a concise assistant. Answer in under 5 lines.",
        state["query"],
    )
    return {"result": answer}


# Fake search hits. web_node passes these to the model and tells it not to
# invent sources beyond the list.
MOCK_SEARCH = [
    "LangGraph routes one query to one specialist. Parallel workers are a separate pattern.",
    "Conditional edges read a label from state and then run a single next node.",
]


def web_node(state: AgentState) -> dict:
    """
    NODE web
    Reads:  query, and the MOCK_SEARCH list in this file
    Writes: result
    Next:   END, always

    Used for questions that go stale: news, weather, "latest", "today".
    """
    print("[Web] using mock search results")
    hits = "\n".join(f"- {hit}" for hit in MOCK_SEARCH)
    answer = ask(
        "Answer using only the search results. If they do not contain the answer, say so.",
        f"Question: {state['query']}\n\nSearch results:\n{hits}",
    )
    return {"result": answer}


# Fake company documents. rag_node is allowed to use these and nothing else.
MOCK_DOCUMENTS = [
    "Company leave policy allows 20 annual leaves.",
    "Employees should apply through the HR portal.",
    "Manager approval is required.",
]


def rag_node(state: AgentState) -> dict:
    """
    NODE rag
    Reads:  query, and MOCK_DOCUMENTS
    Writes: result
    Next:   END, always

    RAG means: retrieve text first, then let the model answer from that text.
    Here the "retrieval" is a fixed list, so you can see the pattern without
    a vector database.
    """
    print("[RAG] using mock documents")
    context = "\n".join(MOCK_DOCUMENTS)
    answer = ask(
        "Answer using only the documents. If the answer is not in them, say you cannot find it.",
        f"Question: {state['query']}\n\nDocuments:\n{context}",
    )
    return {"result": answer}


# Fake table snapshot. database_node must not invent extra rows.
MOCK_DATABASE = {
    "total_users": 1500,
    "active_users": 1200,
    "inactive_users": 300,
}


def database_node(state: AgentState) -> dict:
    """
    NODE database
    Reads:  query, and MOCK_DATABASE
    Writes: result
    Next:   END, always

    Used for counts and stored records. The snapshot stands in for SQL.
    """
    print("[Database] using mock snapshot")
    answer = ask(
        "Answer using only the database snapshot. Do not invent extra tables or rows.",
        f"Question: {state['query']}\n\nSnapshot: {MOCK_DATABASE}",
    )
    return {"result": answer}


def support_node(state: AgentState) -> dict:
    """
    NODE support
    Reads:  query
    Writes: category, escalate, result
    Next:   END, always

    The model fills a SupportRequest. This function then turns that object
    into state fields. escalate=True produces a hand-off message. Otherwise
    the result names the queue and the reason.
    """
    print("[Support]")
    structured = llm.with_structured_output(SupportRequest)
    decision = structured.invoke([
        SystemMessage(content=(
            "Classify this support request.\n"
            "Categories: technical, billing, account, other.\n"
            "Escalate only for a legal threat, a safety issue, or a request to talk to a human."
        )),
        HumanMessage(content=state["query"]),
    ])
    if not isinstance(decision, SupportRequest):
        decision = SupportRequest.model_validate(decision)

    if decision.escalate:
        result = f"Escalated to human support. Reason: {decision.reason}"
    else:
        result = f"Support category: {decision.category}. Reason: {decision.reason}"

    print(f"[Support] {decision.category}, escalate={decision.escalate}")
    return {
        "category": decision.category,
        "escalate": decision.escalate,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Workflow path. Reached only when the route label is "workflow".
# PATH_MAP sends that label to validate, not to execute.
#
#   validate --> route_after_validation
#                  |-- "execute" --> execute --> summarize --> END
#                  |-- "finish"  --> END
#
# route_after_validation reads validated. It does not read the route label.
# ---------------------------------------------------------------------------

def validate_node(state: AgentState) -> dict:
    """
    NODE validate
    Reads:  query
    Writes: validated, result
    Next:   chosen by route_after_validation

    The check is deliberately small: more than 5 characters after stripping
    spaces. A real workflow would check required fields and permissions here,
    before execute_node is allowed to run.
    """
    is_valid = len(state["query"].strip()) > 5
    print(f"[Validate] valid={is_valid}")
    if is_valid:
        return {"validated": True, "result": "Task validated"}
    return {
        "validated": False,
        "result": "Invalid task: the request is too short to run a workflow.",
    }


def route_after_validation(state: AgentState) -> Literal["execute", "finish"]:
    """
    CONDITIONAL EDGE FUNCTION after validate. Not a node.

    "execute" is a PATH key. The map below sends it to the execute node.
    "finish" is also a key. The map sends it to END, so a failed check
    stops without calling execute or summarize.
    """
    if state.get("validated"):
        return "execute"
    return "finish"


def execute_node(state: AgentState) -> dict:
    """
    NODE execute
    Reads:  query
    Writes: result   (replaces the "Task validated" text)
    Next:   summarize, always

    This is the side effect of the workflow. The demo only records a sentence.
    It runs only after validate wrote validated=True.
    """
    print("[Execute]")
    return {"result": f"Workflow executed for: {state['query']}"}


def summarize_node(state: AgentState) -> dict:
    """
    NODE summarize
    Reads:  result   (the sentence execute_node just wrote)
    Writes: result   (a one-sentence confirmation, replacing that sentence)
    Next:   END, always
    """
    print("[Summarize]")
    summary = ask(
        "Summarize this workflow result in one sentence.",
        state["result"],
    )
    return {"result": summary}


# ---------------------------------------------------------------------------
# Multi-agent path. Reached only when the route label is "multi_agent".
#
#   multi_agent --> researcher --+
#               --> analyst -----+--> synthesizer --> END
#
# Two normal edges leave multi_agent, so both workers run in the same step.
# That is fan-out. synthesizer has two incoming edges, so it waits for both.
# That is fan-in.
#
# The worker count here is fixed at two, which is why normal edges are enough.
# Send() is for a count that is only known while the graph is running.
# ---------------------------------------------------------------------------

def multi_agent_node(state: AgentState) -> dict:
    """
    NODE multi_agent
    Reads:  nothing it needs to change
    Writes: nothing (empty dict leaves state as it is)
    Next:   researcher AND analyst, because build_graph adds two edges

    This node is a marker. The interesting work happens in the workers.
    An empty return is valid: "I ran, I did not change state."
    """
    print("[Multi-agent] researcher and analyst")
    return {}


def researcher_node(state: AgentState) -> dict:
    """
    NODE researcher
    Reads:  query
    Writes: worker_results   one note, wrapped in a list
    Next:   synthesizer

    Runs beside analyst, not after it. The reducer appends this list to
    whatever analyst wrote in the same step.
    """
    print("[Researcher]")
    note = ask(
        "You research. Give three short facts that help answer the question.",
        state["query"],
    )
    return {"worker_results": [f"Researcher: {note}"]}


def analyst_node(state: AgentState) -> dict:
    """
    NODE analyst
    Reads:  query
    Writes: worker_results   one note, wrapped in a list
    Next:   synthesizer

    Same step as researcher. It writes the shared list, not result.
    """
    print("[Analyst]")
    note = ask(
        "You compare options. Give a short recommendation with one tradeoff.",
        state["query"],
    )
    return {"worker_results": [f"Analyst: {note}"]}


def synthesizer_node(state: AgentState) -> dict:
    """
    NODE synthesizer
    Reads:  query, worker_results
    Writes: result
    Next:   END, always

    LangGraph starts this node after both incoming edges have fired, so both
    notes are already in worker_results.
    """
    print("[Synthesizer]")
    notes = "\n\n".join(state.get("worker_results", []))
    summary = ask(
        "Combine these notes into one short answer. Keep both perspectives visible.",
        f"Question: {state['query']}\n\nNotes:\n{notes}",
    )
    return {"result": summary}


def build_graph():
    """
    Build the graph, then compile it.

    Order inside this function:
      1. Create an empty StateGraph bound to AgentState.
      2. Register every node. The string is the name edges will use.
         The function is what runs when that name is selected.
      3. Register edges.
      4. compile() checks that edge targets name real nodes and returns
         the object invoke() can run.

    Nothing here calls the model. Building the graph only records the shape.
    """
    builder = StateGraph(AgentState)

    # -- nodes --------------------------------------------------------------
    # Name "router" runs router_node. The name is what edges point at.
    builder.add_node("router", router_node)
    builder.add_node("chat", chat_node)            # general answer, then END
    builder.add_node("web", web_node)              # mock search, then END
    builder.add_node("rag", rag_node)              # mock documents, then END
    builder.add_node("database", database_node)    # mock snapshot, then END
    builder.add_node("support", support_node)      # ticket classification, then END
    builder.add_node("validate", validate_node)    # workflow gate
    builder.add_node("execute", execute_node)      # workflow side effect
    builder.add_node("summarize", summarize_node)  # workflow confirmation
    builder.add_node("multi_agent", multi_agent_node)  # fan-out marker
    builder.add_node("researcher", researcher_node)    # worker A
    builder.add_node("analyst", analyst_node)          # worker B
    builder.add_node("synthesizer", synthesizer_node)  # fan-in

    # -- edges: entry -------------------------------------------------------
    # START --> router
    # Normal edge. Every invoke begins at START, so every query hits router.
    builder.add_edge(START, "router")

    # router --> one of PATH_MAP
    # Conditional edge. After router_node returns, LangGraph calls
    # choose_destination(state). The returned label is looked up in PATH_MAP.
    # One query enters one of those nodes.
    builder.add_conditional_edges("router", choose_destination, PATH_MAP)

    # -- edges: single-step specialists -------------------------------------
    # chat --> END, web --> END, rag --> END, database --> END, support --> END
    # Normal edges. These nodes are the end of their path.
    for node_name in ("chat", "web", "rag", "database", "support"):
        builder.add_edge(node_name, END)

    # -- edges: workflow ----------------------------------------------------
    # validate --> execute    when route_after_validation returns "execute"
    # validate --> END        when it returns "finish"
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {"execute": "execute", "finish": END},
    )
    # execute --> summarize --> END 
    # Two normal edges. A valid workflow always summarizes, then stops.
    builder.add_edge("execute", "summarize")
    builder.add_edge("summarize", END)

    # -- edges: multi_agent fan-out and fan-in -----------------------------
    # multi_agent --> researcher
    # multi_agent --> analyst
    # Two normal edges from the same node. Both workers run.
    builder.add_edge("multi_agent", "researcher")
    builder.add_edge("multi_agent", "analyst")
    # researcher --> synthesizer
    # analyst --> synthesizer
    # Two edges into one node. synthesizer waits for both.
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("analyst", "synthesizer")
    # synthesizer --> END
    builder.add_edge("synthesizer", END)

    return builder.compile()


# Default queries used when you run the file with no arguments.
# Each one is written to hit a different label under the keyword router.
EXAMPLES = [
    "Explain what a router does in one short paragraph.",          # chat
    "What is the latest news about LangGraph?",                    # web ("latest")
    "What is our company leave policy?",                           # rag ("policy")
    "How many active users are in the database?",                  # database
    "I want a refund for this charge.",                            # support ("refund")
    "Please run the onboarding workflow for a new hire.",          # workflow
    # "vector databases" would match the database rule first.
    # "vector stores" does not contain that keyword, so multi_agent wins.
    "Research and compare vector stores for a small team.",        # multi_agent
]


def print_mermaid(compiled) -> None:
    """
    HELPER for main. Prints the graph as Mermaid text.
    Paste the text into a Mermaid viewer if you want a picture.
    Drawing the graph does not run any node.
    """
    print("\n--- Graph ---")
    print(compiled.get_graph().draw_mermaid())


def run_query(compiled, query: str) -> None:
    """
    HELPER for main. One full trip through the graph.

    invoke({"query": query}) starts at START, follows the edges, and returns
    the final AgentState. This function then prints the fields that matter.
    category is printed only after the support node wrote it.
    worker_results is printed only after the multi_agent workers wrote them.
    """
    print("\n" + "=" * 72)
    print("QUERY:", query)
    result = compiled.invoke({"query": query})
    print("ROUTE:", result.get("route"))
    print("WHY:  ", result.get("routing_reason"))
    if result.get("category"):
        print("SUPPORT CATEGORY:", result["category"])
    notes = result.get("worker_results") or []
    if notes:
        print("WORKER NOTES:")
        for note in notes:
            print("-", note)
    print("RESULT:")
    print(result.get("result", ""))


def main() -> None:
    """
    Build the graph once. Run every query passed on the command line.
    With no arguments, run EXAMPLES so each specialist path is exercised.
    """
    import sys

    compiled = build_graph()
    print(f"ROUTING_MODE={routing_mode()}")
    print("One query takes one path. The other specialist nodes do not run.")
    print_mermaid(compiled)

    queries = sys.argv[1:] or EXAMPLES
    for query in queries:
        run_query(compiled, query)


if __name__ == "__main__":
    main()


# =============================================================================
# REVISION NOTES
# =============================================================================
#
# THE FLOW IN ONE LINE
# invoke(query) -> START -> router writes a label -> one edge function picks
# one node -> that path writes result -> END -> invoke returns the state.
#
# STATE
# query is the input. route and routing_reason come from the router.
# result comes from the specialist that ran. category and escalate come from
# support. validated comes from the workflow gate. worker_results comes from
# the two parallel workers and is concatenated by operator.add.
#
# NODE vs EDGE
# A node returns fields. An edge decides the next node.
# choose_destination and route_after_validation are edge functions.
# They return a string key. They are not registered with add_node.
#
# EDGES IN THIS FILE
# START -> router                         always
# router -> PATH_MAP[label]               one branch
# chat, web, rag, database, support -> END
# validate -> execute | END
# execute -> summarize -> END
# multi_agent -> researcher and analyst   both
# researcher and analyst -> synthesizer   waits for both
# synthesizer -> END
#
# IF YOU FORGET HOW THIS WORKS, START HERE
# router_node writes state["route"].
# choose_destination reads it.
# PATH_MAP turns the label into a node name.
# =============================================================================
