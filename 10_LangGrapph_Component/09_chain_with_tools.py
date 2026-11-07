"""
=============================================================================
LANGGRAPH COMPONENT 9: A CHAIN THAT CAN CALL A TOOL
=============================================================================
This is the second half of the "Chain Using LangGraph" notebook.

Three ideas, in the order you should run them:

1. bind_tools
   Hand the model a Python function. The model does NOT run Python.
   It only returns a structured "please call add(2, 2)" request.

2. A one-node chain
   StateGraph with a single node: read messages, call the model, write
   the reply back into state.

3. A chain that actually runs the tool
   ToolNode executes the request. tools_condition decides whether to
   visit that node. The edge from "tools" goes BACK to the model so the
   model can turn "4" into a normal sentence.

   The notebook sent "tools" straight to END. That stops after the raw
   tool result, so the user never gets a final answer. Example 3 fixes that.

STATE
The messages key uses add_messages. Without that reducer, each node would
REPLACE the message list and you would lose the user's question.
File 04_state_reducers.py shows the overwrite vs append difference on its own.

HOW TO STUDY THIS FILE
main() runs one example. Comment it and uncomment the next when you are ready.
=============================================================================
"""

from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config import get_llm, print_mermaid, safe_run


# ---------------------------------------------------------------------------
# TOOL
# ---------------------------------------------------------------------------
# @tool turns a typed function into something the model can call.
# The docstring is the description the model reads. Write it for the model.
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    # add_messages appends new messages. It also matches messages by id
    # so an update replaces a message instead of duplicating it.
    messages: Annotated[list[AnyMessage], add_messages]


def example_see_a_tool_call():
    """Ask a math question and print the tool call. Nothing is executed yet."""
    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools([add])

    # "Use the add tool" matters. A strong model will otherwise just answer 4,
    # and tool_calls stays empty. You would learn the wrong shape.
    question = HumanMessage(content="What is 2 plus 2? Use the add tool.", name="Krish")
    ai_message = llm_with_tools.invoke([question])

    print("The model replied with an AIMessage.")
    print("tool_calls is empty when it wants to answer in plain text.")
    print("tool_calls is filled when it wants US to run a function.\n")
    print(f"content:    {ai_message.content!r}")
    print(f"tool_calls: {ai_message.tool_calls}")

    if ai_message.tool_calls:
        call = ai_message.tool_calls[0]
        print(f"\nIt wants `{call['name']}` with arguments {call['args']}.")
        print("We have not called add() yet. The next examples do that inside a graph.")


def _assistant(state: State, llm_with_tools):
    """One graph node: read the thread, return the model's next message."""
    reply = llm_with_tools.invoke(state["messages"])
    # Returning {"messages": [reply]} APPENDS because of add_messages.
    return {"messages": [reply]}


def example_llm_only_chain():
    """Smallest graph: START -> model -> END. No tool is executed."""
    llm_with_tools = get_llm(temperature=0).bind_tools([add])

    def assistant(state: State) -> dict:
        return _assistant(state, llm_with_tools)

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    graph = builder.compile()

    print_mermaid(graph, "One-node chain")
    print("This graph can ASK for the tool, but nothing runs the tool.\n")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2 plus 2? Use the add tool.")]}
    )
    for message in result["messages"]:
        message.pretty_print()

    print("\nLook at the AIMessage. You should see a tool call, not the number 4.")
    print("Uncomment example_chain_that_runs_the_tool to finish the job.")


def example_chain_that_runs_the_tool():
    """START -> model -> (maybe tools) -> model -> END."""
    llm_with_tools = get_llm(temperature=0).bind_tools([add])

    def assistant(state: State) -> dict:
        return _assistant(state, llm_with_tools)

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    # ToolNode looks at the latest AIMessage, runs matching tools, and
    # appends a ToolMessage with the return value.
    builder.add_node("tools", ToolNode([add]))

    builder.add_edge(START, "assistant")
    # tools_condition reads the latest AIMessage:
    #   tool_calls present -> go to the node named "tools"
    #   no tool_calls       -> go to END
    builder.add_conditional_edges("assistant", tools_condition)
    # Loop back. The model reads the ToolMessage and writes the final answer.
    # The notebook used END here, which is why the answer stayed a raw "4".
    builder.add_edge("tools", "assistant")
    graph = builder.compile()

    print_mermaid(graph, "Chain with a tool loop")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2 plus 2? Use the add tool.")]}
    )
    print("Full message trace:\n")
    for message in result["messages"]:
        message.pretty_print()


def main():
    # Uncomment ONE example at a time. Run them from top to bottom.
    safe_run("See a tool call without running it", example_see_a_tool_call)
    # safe_run("One-node LLM chain", example_llm_only_chain)
    # safe_run("Chain that runs the tool and answers", example_chain_that_runs_the_tool)


if __name__ == "__main__":
    main()
