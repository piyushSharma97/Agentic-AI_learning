"""
=============================================================================
AGENTIC RAG
=============================================================================

Retrieval is a tool. The model decides whether to call it, and which
knowledge base to call.

    START
      |
    agent ---- no tool call ----> END
      |
      | tool call
      v
    retrieve
      |
    grade
      |
      +-- relevant --------> generate --> END
      |
      +-- not relevant ----> rewrite --> agent

Two retrievers are two tools:

    search_langgraph   notes on graphs, nodes, edges, checkpointers
    search_langchain   notes on prompts, models, tools, chat history

tools_condition reads the agent's last message. If that message contains
tool calls, the graph goes to retrieve. ToolNode runs the tool the model
named and appends a ToolMessage.

The grader is its own node. It writes relevant=True or False into state.
The router after it only reads that flag. A rewrite is allowed once.
After that the graph answers with whatever was retrieved, so a bad grade
cannot loop forever.

Run:
    python "14_RAG With LangGraph/1-AgenticRAG.py"
=============================================================================
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import get_llm, print_banner, print_mermaid, safe_run
from corpus import (
    LANGCHAIN_NOTES,
    LANGGRAPH_NOTES,
    build_retriever,
    make_retriever_tool,
    preview_retrieval,
)
from grading import is_relevant


MAX_REWRITES = 1

SYSTEM = """You answer from the project notes by calling a tool first.
Use search_langgraph for LangGraph: nodes, edges, state, checkpointers, interrupts.
Use search_langchain for LangChain: prompts, chat models, tools, message history.
Call one tool. After a tool result is already in the conversation, write the answer from that result.
"""


class AgentState(TypedDict):
    """
    messages uses add_messages, so a node returns new messages and LangGraph
    appends them. The other fields are replaced by whatever the node returns.
    """

    messages: Annotated[list, add_messages]
    relevant: bool
    rewrites: int


def latest_human(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def latest_tool_context(messages) -> str:
    """Join the tool results that belong to the latest retrieval."""
    chunks: list[str] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            chunks.append(message.content)
        elif chunks:
            break
    return "\n\n".join(reversed(chunks))


def build_graph():
    llm = get_llm(temperature=0)
    langgraph_retriever = build_retriever(LANGGRAPH_NOTES, "langgraph")
    langchain_retriever = build_retriever(LANGCHAIN_NOTES, "langchain")
    tools = [
        make_retriever_tool(
            langgraph_retriever,
            "search_langgraph",
            "Search the LangGraph notes about graphs, nodes, edges, and checkpointers.",
        ),
        make_retriever_tool(
            langchain_retriever,
            "search_langchain",
            "Search the LangChain notes about prompts, models, tools, and chat history.",
        ),
    ]
    agent_model = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        print("[node] agent")
        response = agent_model.invoke(state["messages"])
        calls = getattr(response, "tool_calls", None) or []
        if calls:
            names = ", ".join(call["name"] for call in calls)
            print(f"[node] agent requested: {names}")
        else:
            print("[node] agent answered without a tool")
        return {"messages": [response]}

    def grade(state: AgentState) -> dict:
        """Judge the latest tool result. The router will read `relevant`."""
        print("[node] grade")
        question = latest_human(state["messages"])
        context = latest_tool_context(state["messages"])
        relevant = bool(context) and is_relevant(llm, question, context)
        return {"relevant": relevant}

    def route_after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
        if state["relevant"]:
            print("[route] notes can answer the question -> generate")
            return "generate"
        if state["rewrites"] >= MAX_REWRITES:
            print("[route] rewrite limit reached -> generate with the current notes")
            return "generate"
        print("[route] notes missed the question -> rewrite")
        return "rewrite"

    def rewrite(state: AgentState) -> dict:
        """
        Append a new human question. The original question stays in history.
        The agent runs again and can pick a tool with the clearer query.
        """
        print("[node] rewrite")
        question = latest_human(state["messages"])
        response = llm.invoke([
            SystemMessage(content="Turn the question into a short search query. Return only the query."),
            HumanMessage(content=question),
        ])
        query = response.content.strip()
        print(f"[node] rewrite query: {query}")
        return {
            "messages": [HumanMessage(content=query)],
            "rewrites": state["rewrites"] + 1,
        }

    def generate(state: AgentState) -> dict:
        print("[node] generate")
        question = latest_human(state["messages"])
        context = latest_tool_context(state["messages"])
        response = llm.invoke([
            SystemMessage(content="Answer in a short paragraph using only the notes. If they do not contain the answer, say so."),
            HumanMessage(content=f"Question:\n{question}\n\nNotes:\n{context}"),
        ])
        return {"messages": [AIMessage(content=response.content.strip())]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("retrieve", ToolNode(tools))
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "agent")
    # tools_condition returns "tools" or "__end__". The map sends "tools" to our retrieve node.
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", "__end__": END},
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("rewrite", "agent")

    preview_retrieval(langgraph_retriever, "What does a checkpointer save?")
    return graph.compile()


def show_result(result: AgentState) -> None:
    print("\n--- conversation ---")
    for message in result["messages"]:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"AI tool call: {call['name']} {call.get('args')}")
            continue
        label = message.type
        text = message.content if isinstance(message.content, str) else str(message.content)
        print(f"{label}: {text[:500]}")
    print(f"\nrewrites used: {result['rewrites']}")


def run_demo() -> None:
    print_banner(
        "AGENTIC RAG",
        "The model chooses a retriever tool. A grader can send the question back once.",
    )
    graph = build_graph()
    print_mermaid(graph, "Agentic RAG")

    questions = [
        "What does a LangGraph checkpointer save, and what is a thread id?",
        "How does a LangChain chatbot remember earlier messages?",
    ]
    for question in questions:
        print("\n" + "=" * 72)
        print("QUESTION:", question)
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM),
                    HumanMessage(content=question),
                ],
                "relevant": False,
                "rewrites": 0,
            },
            {"recursion_limit": 12},
        )
        show_result(result)


if __name__ == "__main__":
    safe_run("Agentic RAG", run_demo)
