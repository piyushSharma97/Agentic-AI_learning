"""
Interactive Web Chatbot: Streamlit & Local Ollama (Gemma)

What you will learn in this application:
1. Streamlit Execution Model: Why Streamlit reruns scripts from top-to-bottom on every user interaction.
2. Session State Management: Using st.session_state to persist conversation history across UI reruns.
3. Resource Caching: Using @st.cache_resource to initialize the LLM and chain once and reuse them.
4. Local AI Inference: Connecting Streamlit UI to a 100% offline, local Ollama model (Gemma).
5. Observability with LangSmith: Optional tracing and monitoring of every model invocation.
6. Error Handling: Gracefully notifying users if local AI daemons are offline.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# ChatOllama connects LangChain to locally running Ollama chat models.
from langchain_ollama import ChatOllama

# ChatPromptTemplate builds structured message templates.
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser extracts clean plain text from model responses.
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 1. Environment & Optional Observability (LangSmith)
# --------------------------------------------------

load_dotenv()

# If a LangSmith API key is provided in .env, enable automated tracing.
# This logs every prompt, latency metric, and token count to the LangSmith dashboard.
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT", "gemma-streamlit-demo"
    )

# --------------------------------------------------
# 2. Streamlit Web Page Configuration
# --------------------------------------------------

# Must be the first Streamlit command called in your script.
st.set_page_config(
    page_title="Gemma AI Chat",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangChain Chat with Local Gemma")
st.caption("Ask questions powered completely by your local computer via Ollama — zero cloud cost, 100% private.")

# --------------------------------------------------
# 3. Model & Pipeline Initialization (Cached)
# --------------------------------------------------

# CRITICAL STREAMLIT CONCEPT:
# Without `@st.cache_resource`, Streamlit would re-create `ChatOllama` and rebuild the chain
# on EVERY SINGLE user input, creating massive performance overhead.
# `@st.cache_resource` creates the chain ONCE and caches the object in memory across reruns!
@st.cache_resource
def create_chain():
    """Builds and caches the prompt -> model -> parser pipeline."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful, friendly assistant. "
            "Answer clearly and accurately. "
            "If you are unsure, say so.",
        ),
        ("human", "{question}"),
    ])

    # Connect to local Ollama running gemma:2b
    llm = ChatOllama(
        model="gemma:2b",
        temperature=0.3,  # Low temperature = focused, consistent answers
    )

    # Compose into an LCEL pipeline
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

chain = create_chain()

# --------------------------------------------------
# 4. Session State & Chat History Persistence
# --------------------------------------------------

# In Streamlit, normal Python variables reset when the page reruns.
# `st.session_state` acts like a dictionary that persists for the user's active browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render existing chat history on every rerun so previous messages stay visible
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 5. User Input Handling & Streaming/Generating Output
# --------------------------------------------------

# st.chat_input renders an input box anchored at the bottom of the screen.
question = st.chat_input("What would you like to ask?")

if question:
    # 1. Append user's message to session_state
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    # 2. Display the user message immediately in the UI
    with st.chat_message("user"):
        st.markdown(question)

    # 3. Generate response using local model inside a loading spinner
    try:
        with st.chat_message("assistant"):
            with st.spinner("Gemma is thinking..."):
                response = chain.invoke({"question": question})
                st.markdown(response)

        # 4. Save assistant reply to session_state so it persists on subsequent turns
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    except Exception as error:
        st.error(
            "Unable to connect to Ollama. "
            "Please ensure Ollama is running in the background and 'gemma:2b' is installed."
        )
        st.caption(f"Error details: {error}")

# ============================================================
# REVISION NOTES — app.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file provides a full-featured graphical chat interface in the browser using Streamlit,
# connected to a locally hosted Ollama model (`gemma:2b`). It manages UI state,
# renders conversational bubbles, caches expensive model initializations, and handles runtime errors.
#
# KEY CONCEPTS USED:
# • Streamlit: Rapid Python framework for turning data/AI scripts into interactive web applications.
# • st.session_state: Key-value dictionary that preserves variables across Streamlit's script reruns.
# • @st.cache_resource: Decorator ensuring heavy objects (like LLM clients or database connections) are only created once.
# • ChatOllama: LangChain wrapper connecting to local Ollama daemons.
# • st.chat_input & st.chat_message: Modern Streamlit UI primitives for chat dialogs.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file demonstrates building user-facing GUI applications around the LLM pipelines
# developed throughout the course.
#
# THINGS TO REMEMBER:
# • Run this application in terminal with: `streamlit run app.py` (do NOT run with `python app.py`).
# • Ollama must be active and serving at http://localhost:11434 with `gemma:2b` pulled.
#
# COMMON MISTAKES TO AVOID:
# • Trying to maintain chat history in a local variable like `messages = []` instead of `st.session_state.messages`.
# • Forgetting `@st.cache_resource`, causing slow model re-initializations on every message.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# 1. Look at `chain = create_chain()` (cached model pipeline).
# 2. Look at how `st.session_state.messages` preserves the conversation history.
# ============================================================
