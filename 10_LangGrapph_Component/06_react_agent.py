"""
=============================================================================
LANGGRAPH COMPONENT 6: REACT AGENT ARCHITECTURE
=============================================================================
USE
Answer questions that need a calculator or a lookup, where you do not know
the order of calls ahead of time.

WHY THIS APPROACH
A chain (file 05) hard-codes Draft then Critique. That cannot handle
"add, then multiply the result" because the second call depends on the first
answer. ReAct lets the model request a tool, reads the tool result, and
either requests another tool or writes the final sentence.
The loop is `tools -> assistant`. Stopping at the tool node would show
the raw number and never the sentence.

WHAT IS A REACT AGENT?
ReAct stands for "Reasoning + Acting".
Instead of hardcoding a sequence of actions, the LLM dynamically decides:
1. ACT: When a user query needs external information, the model issues a tool call.
2. OBSERVE: LangGraph's `ToolNode` executes the tool and captures the observation.
3. REASON: The observation is fed back to the model. The model assesses whether
   it has enough information to formulate a final answer or needs to call another tool.

LANGGRAPH'S PREBUILT REACT COMPONENTS:
1. `ToolNode(tools)`: A ready-to-use graph node that executes any tool calls
   present in the latest AIMessage and emits `ToolMessage` objects.
2. `tools_condition`: A prebuilt conditional router that checks:
   - If `AIMessage.tool_calls` is non-empty -> routes to "tools"
   - If `AIMessage.tool_calls` is empty -> routes to `END`
3. Cyclic Edge: `builder.add_edge("tools", "assistant")` creates the loop!
=============================================================================
"""

import json
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Import shared helper
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# 1. DEFINE ROBUST, CRASH-FREE TOOLS
# ===========================================================================
# Use: @tool plus a docstring the model can read.
# Why: a plain function has no name/description schema. bind_tools sends
# that schema to the model. The docstring is the instruction for when to call it.
@tool
def add(a: float, b: float) -> float:
    """Add two numbers a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b safely."""
    if b == 0:
        return "Error: Division by zero is undefined."
    return str(round(a / b, 4))


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search indexed knowledge base for AI, machine learning, and science facts.
    Includes built-in facts on 'Attention Is All You Need', Transformer architectures, etc.
    """
    kb = {
        "attention": (
            "'Attention Is All You Need' is a landmark 2017 research paper by Vaswani et al. "
            "from Google Brain and Google Research that introduced the Transformer architecture, "
            "replacing recurrent and convolutional neural networks with self-attention mechanisms."
        ),
        "transformer": (
            "The Transformer model relies entirely on self-attention mechanisms to compute representations "
            "of its input and output without using sequence-aligned RNNs or convolution."
        ),
        "machine learning": (
            "Machine Learning is a field of artificial intelligence focused on building applications "
            "that learn from data and improve their accuracy over time without being explicitly programmed."
        ),
    }

    q_lower = query.lower()
    for key, content in kb.items():
        if key in q_lower:
            return content

    # If external wikipedia package happens to be available, optionally query it
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400))
        return wiki.invoke(query)
    except Exception:
        return f"Summary: Query '{query}' represents a foundational topic in computer science and artificial intelligence."


TOOLS = [add, multiply, divide, search_knowledge_base]


# ===========================================================================
# 2. DEFINE AGENT STATE SCHEMA WITH ADD_MESSAGES REDUCER
# ===========================================================================
class AgentState(TypedDict):
    # The add_messages reducer appends new messages and handles tool tracking
    messages: Annotated[List[AnyMessage], add_messages]


# ===========================================================================
# 3. BUILD THE REACT GRAPH
# ===========================================================================
def build_react_agent():
    # 1. Initialize LLM and bind available tools
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # 2. Assistant Node: invokes the LLM with current message history
    def assistant_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 3. Create Graph Builder
    builder = StateGraph(AgentState)

    # 4. Add Nodes
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))  # Prebuilt tool execution node

    # 5. Connect START to Assistant
    builder.add_edge(START, "assistant")

    # 6. Add Conditional Edge:
    # If assistant emitted tool calls -> routes to 'tools'
    # If assistant emitted final text -> routes to END
    builder.add_conditional_edges("assistant", tools_condition)

    # Use: send the tool result back to the assistant.
    # Why: ToolNode only runs Python. The model has to see the ToolMessage
    # before it can write the answer the user actually reads.
    builder.add_edge("tools", "assistant")

    return builder.compile()


# ===========================================================================
# 4. TRACE & VISUALIZE EXECUTION
# ===========================================================================
def run_and_print_trace(agent, prompt: str):
    print(f"\nUser: {prompt}")
    print("-" * 65)

    input_message = HumanMessage(content=prompt)
    result = agent.invoke({"messages": [input_message]})

    print("\n--- ReAct Step-by-Step Message Trace ---")
    for idx, msg in enumerate(result["messages"], start=1):
        if isinstance(msg, HumanMessage):
            print(f"Step {idx} [HUMAN]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"Step {idx} [AI - REASONED TO CALL TOOL]:")
                for call in msg.tool_calls:
                    print(f"   -> Tool: {call['name']} | Args: {call['args']}")
            else:
                print(f"Step {idx} [AI - FINAL SYNTHESIS]:\n   {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"Step {idx} [TOOL OBSERVATION ({msg.name})]:\n   {msg.content}")
    print("-" * 65 + "\n")


# ===========================================================================
# MAIN DEMO
# ===========================================================================
def main():
    def demo():
        agent = build_react_agent()
        print_mermaid(agent, "ReAct Agent Graph (with ToolNode Cycle)")

        # Query 1: Arithmetic requiring multi-step tool calls
        run_and_print_trace(
            agent,
            "Add 5 plus 5, and then multiply that result by 10."
        )

        # Query 2: Knowledge retrieval tool call
        run_and_print_trace(
            agent,
            "What is 'Attention Is All You Need' and what model did it introduce?"
        )

    safe_run("LangGraph ReAct Agent Architecture", demo)


if __name__ == "__main__":
    main()

