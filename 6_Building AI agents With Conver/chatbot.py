"""
Building A Conversational AI Chatbot with Memory and Trimming

What you will learn in this file:
1. LLM Statelessness: Why LLMs have no built-in memory and how to provide context manually.
2. Message Types: Using HumanMessage, AIMessage, and SystemMessage to structure dialogues.
3. Message History: Using ChatMessageHistory to record conversation turns per user session.
4. RunnableWithMessageHistory: Wrapping LangChain models and chains to automatically load and save history.
5. Session Management: Isolating conversation states using unique session IDs (e.g. chat1 vs chat2).
6. ChatPromptTemplate & MessagesPlaceholder: Dynamically injecting chat history into prompt templates.
7. Multi-parameter Chains: Combining conversation history with custom variables (e.g., language translation).
8. Context Window Management: Using trim_messages to prevent context overflow and reduce token consumption.
9. LCEL Pre-processing: Using RunnablePassthrough.assign and itemgetter to trim history on the fly.
"""

# ============================================================
# LANGCHAIN: CONVERSATIONAL CHATBOT - COMPLETE PRACTICE FILE
# ============================================================
#
# Examples covered:
#
# 1. Single turn — one HumanMessage, no memory
# 2. Manual history — resend HumanMessage and AIMessage turns
# 3. Session chat1 — RunnableWithMessageHistory remembers Krish
# 4. Isolated chat2 — a different session_id does not see chat1
# 5. Prompt template — system instructions plus MessagesPlaceholder
# 6. Language variable — answer in a runtime language
# 7. trim_messages — drop older messages to a token budget
# 8. Trimmed chain — trim inside LCEL, then keep session history
#
# ============================================================


# ============================================================
# 1. IMPORTS AND ENVIRONMENT CONFIGURATION
# ============================================================

import os
# os reads environment variables such as GROQ_API_KEY from the operating system.
from operator import itemgetter

from dotenv import load_dotenv
# python-dotenv loads environment variables from a .env file into the process environment.
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
# ChatGroq connects LangChain to Groq's high-speed inference engine.
# Here we use Gemma2-9b-It (or another Groq-supported chat model).
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. API Configuration & Model Initialization
# --------------------------------------------------

# Load environment variables from .env.
load_dotenv()

# Fetch the Groq API key from environment variables.
groq_api_key = os.getenv("GROQ_API_KEY")


def _make_model():
    """Create a fresh ChatGroq client so each example can run on its own."""
    return ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


def _make_session_history():
    """
    Build a private in-memory session store and the factory RunnableWithMessageHistory expects.

    Retrieve the message history for a specific session.
    If the session does not exist yet, initialize a new ChatMessageHistory for it.
    """
    # In-memory dictionary acting as our session store: mapping session_id -> ChatMessageHistory
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the message history for a specific session.
        If the session does not exist yet, initialize a new ChatMessageHistory for it.
        """
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return get_session_history


# ============================================================
# EXAMPLE 1: SINGLE TURN
# USE CASE: STATELESS MODEL CALL
# ============================================================

# --------------------------------------------------
# 2. Stateless Nature of LLMs
# --------------------------------------------------
# LLMs are inherently stateless: each call to invoke() is independent.
# The model does not remember past calls unless we explicitly send the history back.

def example_1_single_turn():

    print("\n========== EXAMPLE 1: SINGLE TURN ==========")

    model = _make_model()

    # A single user message sent to the model.
    response_single = model.invoke([HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")])
    print("[Example 1 | single turn] Assistant reply:")
    print(response_single.content)


# ============================================================
# EXAMPLE 2: MANUAL HISTORY
# USE CASE: RE-SEND THE TRANSCRIPT
# ============================================================

# --------------------------------------------------
# 3. Manual Conversation History
# --------------------------------------------------
# To give the illusion of memory, we must pass the entire previous conversation transcript
# (both HumanMessage and AIMessage) in every subsequent request.

def example_2_manual_history():

    print("\n========== EXAMPLE 2: MANUAL HISTORY ==========")

    model = _make_model()

    manual_conversation = [
        HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer"),
        AIMessage(content="Hello Krish! It's nice to meet you. \n\nAs a Chief AI Engineer, what kind of projects are you working on these days? \n\nI'm always eager to learn more about the exciting work being done in the field of AI.\n"),
        HumanMessage(content="Hey What's my name and what do I do?")
    ]

    response_manual = model.invoke(manual_conversation)
    print("[Example 2 | manual history] Assistant reply:")
    print(response_manual.content)


# ============================================================
# EXAMPLE 3: SESSION CHAT1
# USE CASE: AUTOMATED MEMORY FOR ONE SESSION
# ============================================================

# --------------------------------------------------
# 4. Automated Memory with RunnableWithMessageHistory
# --------------------------------------------------
# Instead of managing message lists manually in our code, LangChain provides
# RunnableWithMessageHistory to automatically append user inputs and model outputs
# to a stateful message history datastore.

def example_3_session_chat1():

    print("\n========== EXAMPLE 3: SESSION CHAT1 ==========")

    model = _make_model()
    get_session_history = _make_session_history()

    # Wrap the model with message history management.
    # Every time this runnable is invoked, it retrieves history using get_session_history,
    # appends the input messages, calls the model, and appends the model's reply.
    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    # Config specifies which session this invocation belongs to.
    config = {"configurable": {"session_id": "chat1"}}

    # Turn 1 in session 'chat1'
    response_chat1_turn1 = with_message_history.invoke(
        [HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")],
        config=config
    )
    print("[Example 3 | session chat1] Turn 1 assistant reply:")
    print(response_chat1_turn1.content)

    # Turn 2 in session 'chat1': The model now remembers the name from Turn 1!
    response_chat1_turn2 = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print("[Example 3 | session chat1] Turn 2 assistant reply:")
    print(response_chat1_turn2.content)


# ============================================================
# EXAMPLE 4: ISOLATED CHAT2
# USE CASE: A SECOND SESSION DOES NOT SEE CHAT1
# ============================================================

# --------------------------------------------------
# 5. Session Isolation (chat1 vs chat2)
# --------------------------------------------------
# Different session IDs maintain completely separate conversation memories.

def example_4_isolated_chat2():

    print("\n========== EXAMPLE 4: ISOLATED CHAT2 ==========")

    model = _make_model()
    get_session_history = _make_session_history()

    # This example owns its history store, so it runs without example 3.
    # Session id "chat2" is still isolated from any "chat1" history.
    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    config1 = {"configurable": {"session_id": "chat2"}}

    # In session 'chat2', the model does NOT know who we are yet.
    response_chat2_turn1 = with_message_history.invoke(
        [HumanMessage(content="Whats my name")],
        config=config1
    )
    print("[Example 4 | isolated chat2] Turn 1 assistant reply:")
    print(response_chat2_turn1.content)

    # Introduce a new user 'John' in chat2.
    response_chat2_turn2 = with_message_history.invoke(
        [HumanMessage(content="Hey My name is John")],
        config=config1
    )
    print("[Example 4 | isolated chat2] Turn 2 assistant reply:")
    print(response_chat2_turn2.content)

    # Now chat2 remembers 'John', while chat1 still remembers 'Krish'.
    response_chat2_turn3 = with_message_history.invoke(
        [HumanMessage(content="Whats my name")],
        config=config1
    )
    print("[Example 4 | isolated chat2] Turn 3 assistant reply:")
    print(response_chat2_turn3.content)


# ============================================================
# EXAMPLE 5: PROMPT TEMPLATE
# USE CASE: SYSTEM INSTRUCTIONS PLUS HISTORY
# ============================================================

# --------------------------------------------------
# 6. Integrating Prompt Templates & System Instructions
# --------------------------------------------------
# To give our assistant a consistent persona or behavior instructions,
# we use ChatPromptTemplate with a MessagesPlaceholder for the history.

def example_5_prompt_template():

    print("\n========== EXAMPLE 5: PROMPT TEMPLATE ==========")

    model = _make_model()
    get_session_history = _make_session_history()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer all the questions to the best of your ability"),
            # MessagesPlaceholder dynamically injects the list of conversation messages into the prompt
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # Chain the prompt template directly into the model using LCEL pipe (|)
    chain = prompt | model

    # Testing the chain with a raw dictionary input
    test_response = chain.invoke({"messages": [HumanMessage(content="Hi My name is Krish")]})
    print("[Example 5 | prompt template] Direct chain assistant reply:")
    print(test_response.content)

    # Wrap the prompt+model chain with session history
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)

    config = {"configurable": {"session_id": "chat3"}}
    response_chat3_turn1 = with_message_history.invoke(
        [HumanMessage(content="Hi My name is Krish")],
        config=config
    )
    print("[Example 5 | prompt template] Session chat3 turn 1 assistant reply:")
    print(response_chat3_turn1.content)

    response_chat3_turn2 = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print("[Example 5 | prompt template] Session chat3 turn 2 assistant reply:")
    print(response_chat3_turn2.content)


# ============================================================
# EXAMPLE 6: LANGUAGE VARIABLE
# USE CASE: A PROMPT INPUT BESIDES MESSAGES
# ============================================================

# --------------------------------------------------
# 7. Multi-Variable Prompts (e.g., Language Parameter)
# --------------------------------------------------
# When a prompt template requires variables beyond just messages (such as {language}),
# we must inform RunnableWithMessageHistory which input key contains the messages.

def example_6_language_variable():

    print("\n========== EXAMPLE 6: LANGUAGE VARIABLE ==========")

    model = _make_model()
    get_session_history = _make_session_history()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model

    # Test the multi-variable chain directly
    response_language_direct = chain.invoke(
        {"messages": [HumanMessage(content="Hi My name is Krish")], "language": "Hindi"}
    )
    print("[Example 6 | language variable] Direct chain assistant reply:")
    print(response_language_direct.content)

    # When using multiple input keys, input_messages_key specifies which key holds the user's messages.
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )

    config = {"configurable": {"session_id": "chat4"}}
    response_chat4_turn1 = with_message_history.invoke(
        {'messages': [HumanMessage(content="Hi, I am Krish")], "language": "Hindi"},
        config=config
    )
    print("[Example 6 | language variable] Session chat4 turn 1 assistant reply:")
    print(response_chat4_turn1.content)


# ============================================================
# EXAMPLE 7: TRIM MESSAGES
# USE CASE: COUNT HOW MANY MESSAGES SURVIVE A TOKEN BUDGET
# ============================================================

# --------------------------------------------------
# 8. Managing Conversation History & Message Trimming
# --------------------------------------------------
# Long conversations can exceed the LLM's context window limit (max tokens)
# and increase API latency/cost. trim_messages keeps only the most recent tokens.

def example_7_trim_messages():

    print("\n========== EXAMPLE 7: TRIM MESSAGES ==========")

    model = _make_model()

    # trim_messages configuration:
    # - max_tokens: maximum number of tokens allowed in history
    # - strategy="last": keep the most recent messages, discard oldest
    # - token_counter=model: use the model's tokenizer to count tokens accurately
    # - include_system=True: preserve system message regardless of trimming
    # - allow_partial=False: do not cut messages in half; drop entire message if needed
    # - start_on="human": ensure the trimmed conversation starts on a user message
    trimmer = trim_messages(
        max_tokens=45,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )

    sample_messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    trimmed_sample = trimmer.invoke(sample_messages)
    print(
        f"[Example 7 | trim messages] Sample messages before trim: {len(sample_messages)} "
        f"-> after trim: {len(trimmed_sample)}"
    )


# ============================================================
# EXAMPLE 8: TRIMMED CHAIN
# USE CASE: TRIM INSIDE LCEL, THEN STORE THE SESSION
# ============================================================

# --------------------------------------------------
# 9. Integrating Trimmer into LCEL Pipeline
# --------------------------------------------------
# RunnablePassthrough.assign modifies or adds a specific key to the input dictionary.
# Here it intercepts 'messages', pipes it through the trimmer, and updates 'messages' before the prompt.

def example_8_trimmed_chain():

    print("\n========== EXAMPLE 8: TRIMMED CHAIN ==========")

    model = _make_model()

    # Same language prompt used when this chain was first assembled.
    # The trimmer runs before the prompt, and {language} is still required.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # trim_messages configuration:
    # - max_tokens: maximum number of tokens allowed in history
    # - strategy="last": keep the most recent messages, discard oldest
    # - token_counter=model: use the model's tokenizer to count tokens accurately
    # - include_system=True: preserve system message regardless of trimming
    # - allow_partial=False: do not cut messages in half; drop entire message if needed
    # - start_on="human": ensure the trimmed conversation starts on a user message
    trimmer = trim_messages(
        max_tokens=45,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )

    sample_messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
    )

    response_ice_cream = chain.invoke(
        {
            "messages": sample_messages + [HumanMessage(content="What ice cream do i like")],
            "language": "English"
        }
    )
    print("[Example 8 | trimmed chain] Ice cream question assistant reply:")
    print(response_ice_cream.content)

    response_math = chain.invoke(
        {
            "messages": sample_messages + [HumanMessage(content="what math problem did i ask")],
            "language": "English",
        }
    )
    print("[Example 8 | trimmed chain] Math problem question assistant reply:")
    print(response_math.content)

    # --------------------------------------------------
    # 10. Complete Production-Ready Chatbot Chain
    # --------------------------------------------------
    # Combining trimming, prompt templates, language parameters, and automatic session history!

    get_session_history = _make_session_history()

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    config = {"configurable": {"session_id": "chat5"}}

    response_chat5_turn1 = with_message_history.invoke(
        {
            "messages": sample_messages + [HumanMessage(content="whats my name?")],
            "language": "English",
        },
        config=config,
    )
    print("[Example 8 | trimmed chain] Session chat5 turn 1 assistant reply:")
    print(response_chat5_turn1.content)

    response_chat5_turn2 = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="what math problem did i ask?")],
            "language": "English",
        },
        config=config,
    )
    print("[Example 8 | trimmed chain] Session chat5 turn 2 assistant reply:")
    print(response_chat5_turn2.content)


# ============================================================
# 11. MAIN FUNCTION
# ============================================================

def main():

    print("\n============================================")
    print("LANGCHAIN - CONVERSATIONAL CHATBOT PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_single_turn()

    # example_2_manual_history()

    # example_3_session_chat1()

    # example_4_isolated_chat2()

    # example_5_prompt_template()

    # example_6_language_variable()

    # example_7_trim_messages()

    # example_8_trimmed_chain()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — chatbot.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file is an end-to-end masterclass in building conversational AI chatbots.
# It progresses step-by-step from raw stateless LLM calls to manual memory,
# automated session-based memory using RunnableWithMessageHistory, multi-variable prompt templates,
# and finally token-efficient context management with trim_messages and RunnablePassthrough.
#
# KEY CONCEPTS USED:
# • Statelessness: LLMs do not retain memory between requests; context must be supplied each turn.
# • ChatMessageHistory: An in-memory datastore holding a list of HumanMessage and AIMessage objects.
# • RunnableWithMessageHistory: LangChain wrapper that handles reading from and writing to history automatically.
# • Session Isolation: Partitioning conversations using a session_id in the configurable dictionary.
# • MessagesPlaceholder: Prompt template slot where past conversation messages are inserted.
# • input_messages_key: Disambiguates which input key contains messages when passing multiple parameters (e.g. {language}).
# • trim_messages: Discards older messages to avoid context window overflow while preserving system instructions.
# • RunnablePassthrough.assign: Injects transformed data (trimmed messages) into the input dictionary in an LCEL chain.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file serves as the comprehensive reference notebook implementation for module 6,
# which is broken down into structured standalone sub-lessons (01_basic_chat through 07_complete_chatbot).
#
# THINGS TO REMEMBER:
# • Always specify input_messages_key when the input to the chain is a dictionary with multiple keys.
# • In trim_messages, strategy="last" keeps the most recent context, which is typically most relevant.
# • In production, replace the in-memory store dictionary with persistent stores like Redis, SQLite, or PostgreSQL.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `config={"configurable": {"session_id": "..."}}`, causing a missing session ID error.
# • Omitting `start_on="human"` in message trimming, which might leave a dangling AI response without a prompt.
# • Allowing conversation history to grow infinitely, eventually triggering token limit exceptions.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Step 10 summarizes the complete pattern:
#   chain = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt | model
#   chatbot = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")
# ============================================================


# ============================================================
# END OF CONVERSATIONAL CHATBOT PRACTICE
# ============================================================
