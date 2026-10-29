"""
Lesson 3: Session Memory — Automated Message History Management

What you will learn in this lesson:
1. How to automate message history tracking instead of manually managing message lists.
2. The role of ChatMessageHistory and BaseChatMessageHistory.
3. How to implement session-based isolation using session IDs and a session store dictionary.
4. How RunnableWithMessageHistory transparently intercepts inputs, appends history, calls the model, and stores replies.
5. How to build an interactive command-line terminal chatbot with session reset capability (/new).
"""

import os
# uuid4 generates unique random session identifiers (e.g. "c9bf9e57-1685-4c89-bafb-ff5af830be8a").
from uuid import uuid4

from dotenv import load_dotenv
from langchain_groq import ChatGroq
# HumanMessage wraps user input strings into message objects.
from langchain_core.messages import HumanMessage
# BaseChatMessageHistory is the abstract interface defining how message histories behave.
from langchain_core.chat_history import BaseChatMessageHistory
# RunnableWithMessageHistory wraps any Runnable (model/chain) to give it stateful message storage.
from langchain_core.runnables.history import RunnableWithMessageHistory
# ChatMessageHistory is an in-memory implementation of BaseChatMessageHistory.
from langchain_community.chat_message_histories import ChatMessageHistory

# --------------------------------------------------
# 1. Environment & Model Setup
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# --------------------------------------------------
# 2. Session Store & Factory Function
# --------------------------------------------------

# In-memory dictionary mapping session_id (str) -> ChatMessageHistory.
# Note: Because this is an in-memory dict, conversations reset when the script exits.
# In production, this can be backed by Redis, SQLite, MongoDB, or Postgres.
session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Factory function required by RunnableWithMessageHistory.
    Given a session_id, returns existing history or creates and stores a fresh one.
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# --------------------------------------------------
# 3. Wrapping the Model with RunnableWithMessageHistory
# --------------------------------------------------

# RunnableWithMessageHistory acts as a proxy:
# 1. Inspects config['configurable']['session_id']
# 2. Calls get_session_history(session_id) to retrieve previous messages
# 3. Merges previous messages with new incoming messages
# 4. Passes combined messages to the underlying model
# 5. Automatically records both the human input and model's response into history
chat = RunnableWithMessageHistory(model, get_session_history)

# Generate a fresh UUID for the initial conversation session.
session_id = str(uuid4())

# --------------------------------------------------
# 4. Interactive Chat Loop (Terminal REPL)
# --------------------------------------------------

print("Session-memory demo. Type /new for a separate chat, or exit to quit.")

while True:
    # Read user input from the console
    text = input("\nYou: ").strip()

    # Allow user to quit the application
    if text.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    # Command to simulate a new user session or wipe conversation context
    if text == "/new":
        session_id = str(uuid4())
        print(f"Started a new session (ID: {session_id[:8]}...). Memory is reset!")
        continue

    # Skip empty inputs
    if not text:
        continue

    # Invoke the stateful chat runner.
    # Notice we only send the CURRENT message; LangChain automatically pulls and updates history!
    response = chat.invoke(
        [HumanMessage(content=text)],
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Assistant: {response.content}")

# ============================================================
# REVISION NOTES — 03_session_memory/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script builds a stateful terminal chatbot that automatically retains memory across dialogue turns.
# By pairing LangChain's RunnableWithMessageHistory with an in-memory session dictionary, it eliminates
# the need to manually append messages in application code.
#
# KEY CONCEPTS USED:
# • ChatMessageHistory: Container that holds a list of messages for a single session.
# • RunnableWithMessageHistory: High-level LangChain wrapper that binds message persistence to runnables.
# • session_id: A unique key identifying which conversation thread a request belongs to.
# • Configurable parameters: The config dict pattern `{"configurable": {"session_id": "..."}}`
#   used by LangChain runnables to pass runtime execution parameters.
#
# HOW IT CONNECTS TO THE PROJECT:
# This lesson solves the manual history management issue seen in Lesson 2 and introduces the core session
# pattern used across all production conversational architectures.
#
# THINGS TO REMEMBER:
# • RunnableWithMessageHistory takes the runnable as its first argument and the session history function as the second.
# • In an interactive loop, each new input is sent as `[HumanMessage(content=text)]` with the same session_id.
# • Changing the session_id completely isolates the conversation from previous chats.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting the nested structure in config: it must be `{"configurable": {"session_id": ...}}`.
# • Relying on in-memory session_store for production; restarting the server clears all memory.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `chat = RunnableWithMessageHistory(model, get_session_history)` and trace how `config`
# supplies `session_id` on each turn of the `while True` loop.
# ============================================================
