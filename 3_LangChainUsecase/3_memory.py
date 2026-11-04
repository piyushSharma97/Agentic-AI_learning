"""
=============================================================================
USE CASE 3: CONVERSATIONAL MEMORY IN LANGCHAIN
=============================================================================
Large Language Models are inherently stateless: each request is treated in
complete isolation. Memory gives agents and chat pipelines the ability to
remember past context, user preferences, and multi-turn dialogues.

Key Concepts Covered:
1. Stateless vs Stateful interaction
2. InMemoryChatMessageHistory for storing message histories
3. `RunnableWithMessageHistory` to wrap LCEL chains seamlessly
4. Multi-session management (tracking distinct users by session_id)
5. Context Window Management with `trim_messages` (preventing token overflow)
6. Checkpointing & Persistent State concepts (LangGraph MemorySaver)
=============================================================================
"""

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. RunnableWithMessageHistory: The Modern LangChain Memory Standard
# ---------------------------------------------------------------------------
def demo_runnable_with_message_history():
    print("--- 1. Multi-Turn Conversational Memory with RunnableWithMessageHistory ---")
    
    # In-memory store mapping session_id -> InMemoryChatMessageHistory
    session_store: Dict[str, InMemoryChatMessageHistory] = {}
    
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    llm = get_llm()
    
    # Prompt with MessagesPlaceholder for chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly personal fitness assistant. Keep your answers brief."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    base_chain = prompt | llm | StrOutputParser()
    
    # Wrap the chain with RunnableWithMessageHistory
    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    
    # Session 1: User Alice
    config_alice = {"configurable": {"session_id": "alice_session_001"}}
    
    print("\n--- Interaction 1 (Alice): Introduces her goal ---")
    q1 = "Hi, my name is Alice. I want to train for a half marathon in 3 months."
    print(f"Alice: {q1}")
    res1 = conversational_chain.invoke({"question": q1}, config=config_alice)
    print(f"Assistant: {res1}")
    
    print("\n--- Interaction 2 (Alice): Refers to past context without repeating name or goal ---")
    q2 = "What should my running distance target be for this week?"
    print(f"Alice: {q2}")
    res2 = conversational_chain.invoke({"question": q2}, config=config_alice)
    print(f"Assistant: {res2}")
    
    print("\n--- Interaction 3 (Alice): Asks if the assistant remembers her name ---")
    q3 = "What is my name and what event am I training for?"
    print(f"Alice: {q3}")
    res3 = conversational_chain.invoke({"question": q3}, config=config_alice)
    print(f"Assistant: {res3}")

    # Session 2: User Bob (Isolated state)
    config_bob = {"configurable": {"session_id": "bob_session_002"}}
    print("\n--- Interaction 4 (Bob): Completely new isolated session ---")
    q_bob = "Do you know who I am?"
    print(f"Bob: {q_bob}")
    res_bob = conversational_chain.invoke({"question": q_bob}, config=config_bob)
    print(f"Assistant: {res_bob}")


# ---------------------------------------------------------------------------
# 2. Context Window Management: trim_messages
# ---------------------------------------------------------------------------
def demo_context_window_trimming():
    print("\n--- 2. Managing Context Window with trim_messages ---")
    
    # Simulated long conversation history that would exceed small token limits
    messages = [
        SystemMessage(content="You are a financial advisor."),
        HumanMessage(content="Turn 1: What is compound interest?"),
        AIMessage(content="Turn 1: Compound interest is interest on principal plus accumulated interest."),
        HumanMessage(content="Turn 2: What is an index fund?"),
        AIMessage(content="Turn 2: An index fund tracks a specific market index like the S&P 500."),
        HumanMessage(content="Turn 3: What is a dividend?"),
        AIMessage(content="Turn 3: A dividend is a distribution of profits by a company to shareholders."),
        HumanMessage(content="Turn 4: Should I invest in high-dividend stocks right now?")
    ]
    
    print(f"Original message count: {len(messages)}")
    
    # Trim keeping only the last 3 messages while preserving system message
    trimmed = trim_messages(
        messages,
        max_tokens=4,               # When token_counter is count, max_tokens acts as message count
        token_counter=len,          # Count number of messages
        strategy="last",
        include_system=True,        # Retain the initial system prompt
        start_on="human"            # Ensure the dialogue starts logically with a human message
    )
    
    print(f"Trimmed message count: {len(trimmed)}")
    print("Retained messages:")
    for msg in trimmed:
        print(f"  [{msg.type.upper()}]: {msg.content}")
    print()


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Conversational Memory Demonstrations", lambda: [
        demo_runnable_with_message_history(),
        demo_context_window_trimming(),
    ])


if __name__ == "__main__":
    main()

