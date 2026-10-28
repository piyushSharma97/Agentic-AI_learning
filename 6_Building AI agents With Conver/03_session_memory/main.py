"""Lesson 3: let LangChain store messages by session ID."""

import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# This dictionary exists only while this Python process is running.
session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return this session's history, creating it if needed."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


chat = RunnableWithMessageHistory(model, get_session_history)
session_id = str(uuid4())

print("Session-memory demo. Type /new for a separate chat, or exit to quit.")

while True:
    text = input("\nYou: ").strip()

    if text.lower() in {"exit", "quit"}:
        break

    if text == "/new":
        session_id = str(uuid4())
        print("Started a new session.")
        continue

    if not text:
        continue

    response = chat.invoke(
        [HumanMessage(content=text)],
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Assistant: {response.content}")
