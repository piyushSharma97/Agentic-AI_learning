"""
Lesson 7: Complete Production Chatbot Architecture

What you will learn in this lesson:
1. The 5-Step architecture of a production-grade conversational AI chatbot.
2. Step 1: Session-isolated memory storage using ChatMessageHistory.
3. Step 2: Parameterized prompt engineering with system instructions, history placeholder, and user input.
4. Step 3: Context-window safety guards using token trimmers.
5. Step 4: LCEL pipeline integration using RunnablePassthrough.assign and itemgetter to trim messages inline.
6. Step 5: Dual-key history binding with RunnableWithMessageHistory (input_messages_key and history_messages_key).
7. Building an interactive terminal application supporting runtime commands (/new, /language <lang>, exit).
"""

# os reads environment variables so secrets and model settings stay outside the source code.
import os
# uuid4 creates globally unique session identifiers for distinct conversations.
from uuid import uuid4
# itemgetter selects the "history" value from the input dictionary in the LCEL pipeline.
from operator import itemgetter

# load_dotenv makes values in the local .env file available through os.environ.
from dotenv import load_dotenv
# ChatMessageHistory stores message sequences in memory.
from langchain_community.chat_message_histories import ChatMessageHistory
# BaseChatMessageHistory is the standardized base interface for all history stores.
from langchain_core.chat_history import BaseChatMessageHistory
# trim_messages truncates lists of messages to fit safely inside token constraints.
from langchain_core.messages import trim_messages
# ChatPromptTemplate formats messages; MessagesPlaceholder slots in history lists.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# RunnablePassthrough passes inputs through unchanged or assigns modified keys to input dictionaries.
from langchain_core.runnables import RunnablePassthrough
# RunnableWithMessageHistory connects history retrieval and saving to an LCEL chain.
from langchain_core.runnables.history import RunnableWithMessageHistory
# ChatGroq delivers high-speed LLM inference.
from langchain_groq import ChatGroq

# --------------------------------------------------
# Environment & Model Initialization
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# --------------------------------------------------
# STEP 1: Session Store & History Provider
# --------------------------------------------------

# Map session_id string -> BaseChatMessageHistory instance.
# Each user or chat tab receives a unique session ID for complete conversation isolation.
session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Factory function returning the active ChatMessageHistory for a given session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# --------------------------------------------------
# STEP 2: Reusable Multi-Variable Prompt Template
# --------------------------------------------------

# The prompt combines:
# 1. System instructions: defines persona and enforces dynamic response language {language}.
# 2. MessagesPlaceholder: dynamically injects trimmed conversation turns under the variable name "history".
# 3. Human message: formats the latest user query {question}.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Be accurate and clear. "
        "Respond in {language}.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# --------------------------------------------------
# STEP 3: History Trimmer for Context Window Safety
# --------------------------------------------------

# Trimmer protects the LLM from token overflow errors.
# - max_tokens=1200: keeps history within 1200 tokens.
# - strategy="last": retains the most recent conversation context.
# - token_counter="approximate": fast character/token heuristic (approx 4 chars/token).
# - start_on="human": ensures history begins on a user turn.
history_trimmer = trim_messages(
    max_tokens=1200,
    strategy="last",
    token_counter="approximate",
    include_system=False,
    allow_partial=False,
    start_on="human",
)

# --------------------------------------------------
# STEP 4: Assembling the Processing Pipeline (LCEL)
# --------------------------------------------------

# RunnablePassthrough.assign takes incoming dictionary:
# {'question': '...', 'language': '...', 'history': [...]}
# It extracts 'history' via itemgetter("history"), pipes it through history_trimmer,
# and updates the 'history' key in place.
# The trimmed dictionary then flows cleanly into prompt -> model.
chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | history_trimmer
    )
    | prompt
    | model
)

# --------------------------------------------------
# STEP 5: Wrapping Chain with RunnableWithMessageHistory
# --------------------------------------------------

# When our chain expects multiple dictionary keys, we must tell RunnableWithMessageHistory:
# 1. input_messages_key="question": Which key contains the new user query to append to history.
# 2. history_messages_key="history": Which key should receive the stored conversation messages.
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --------------------------------------------------
# Interactive Terminal Chatbot Loop
# --------------------------------------------------

session_id = str(uuid4())
language = "English"

print("==================================================")
print("Complete Chatbot Architecture Demo")
print("Commands:")
print("  /new              - Start a fresh session (clears memory)")
print("  /language <lang>  - Switch language (e.g. /language Hindi, /language Spanish)")
print("  exit / quit       - Terminate the chatbot")
print("==================================================")

while True:
    user_text = input("\nYou: ").strip()

    # Exit command
    if user_text.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    # Reset session command
    if user_text == "/new":
        session_id = str(uuid4())
        print(f"Started a new session (ID: {session_id[:8]}...). Memory cleared.")
        continue

    # Change response language dynamically
    if user_text.startswith("/language "):
        language = user_text.removeprefix("/language ").strip() or "English"
        print(f"Response language set to: {language}")
        continue

    # Ignore blank lines
    if not user_text:
        continue

    # Invoke the full stateful, trimmed pipeline
    try:
        response = chatbot.invoke(
            {"question": user_text, "language": language},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {response.content}")
    except Exception as error:
        print(f"Request failed: {error}")

# ============================================================
# REVISION NOTES — 07_complete_chatbot/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script integrates all concepts from Lessons 1 through 6 into a robust, complete conversational AI agent.
# It features automated per-session memory, multi-lingual prompt steering, sliding-window token trimming,
# and an interactive CLI with slash commands.
#
# KEY CONCEPTS USED:
# • 5-Step Chatbot Pattern:
#   1. Session storage (get_session_history)
#   2. Prompt template with MessagesPlaceholder
#   3. Message trimmer (trim_messages)
#   4. LCEL composition with RunnablePassthrough.assign
#   5. Stateful binding with RunnableWithMessageHistory
# • input_messages_key & history_messages_key:
#   Crucial parameter pair when the pipeline input is a multi-key dictionary.
#   - input_messages_key identifies the new incoming user text.
#   - history_messages_key specifies where to inject prior conversation history.
# • RunnablePassthrough.assign: Injects calculated values into the pipeline dictionary without discarding existing keys.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is the capstone file of Module 6 ("Building AI agents With Conversation").
# It demonstrates the exact production pattern used to construct conversational assistants,
# customer support bots, and copilot interfaces.
#
# THINGS TO REMEMBER:
# • Setting `token_counter="approximate"` avoids calling model tokenizers for trimming, providing lower latency.
# • The `/language` command alters the system prompt dynamically on subsequent turns without breaking memory.
# • In cloud deployments, swap the `session_store` dictionary for persistent databases (Redis, Postgres, DynamoDB).
#
# COMMON MISTAKES TO AVOID:
# • Mismatching key names between `MessagesPlaceholder(variable_name="history")` and `history_messages_key="history"`.
# • Omitting `input_messages_key`, which causes RunnableWithMessageHistory to fail when passed a multi-key dict.
# • Letting conversation history grow without trimming, which inevitably causes context window crashes.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Review STEP 4 and STEP 5:
#   chain = RunnablePassthrough.assign(history=itemgetter("history") | history_trimmer) | prompt | model
#   chatbot = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question", history_messages_key="history")
# ============================================================
