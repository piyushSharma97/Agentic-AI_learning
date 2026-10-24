# os reads and sets process environment variables for optional LangSmith tracing.
import os

# Streamlit provides the web interface and caches the reusable model chain.
import streamlit as st
# python-dotenv loads local configuration such as API keys from a .env file.
from dotenv import load_dotenv
# ChatOllama connects LangChain to a locally running Ollama chat model.
from langchain_ollama import ChatOllama
# ChatPromptTemplate formats the system and user messages sent to the model.
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser extracts model output as plain text for display in the app.
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

# Load local environment settings before checking whether tracing is configured.
load_dotenv()
# Enable LangSmith tracing only when an API key exists
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT", "gemma-streamlit-demo"
    )

# --------------------------------------------------
# 2. Configure Streamlit
# --------------------------------------------------

# Set the browser tab title, icon, and centered layout before drawing the app.
st.set_page_config(
    page_title="Gemma AI Chat",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangChain Chat with Gemma")
st.caption("Ask questions using a locally running Ollama model.")

# --------------------------------------------------
# 3. Initialize the LLM and chain
# --------------------------------------------------

# Build and cache the prompt-to-model-to-text pipeline; it takes no arguments and returns a runnable chain.
@st.cache_resource
def create_chain():
    # The system message describes the assistant; {question} is filled in for each user turn.
    prompt = ChatPromptTemplate.from_messages([
          (
                "system",
                "You are a helpful, friendly assistant. "
                "Answer clearly and accurately. "
                "If you are unsure, say so.",
            ),
            ("human", "{question}"),
    ])

    # Connect to the local Gemma model; a lower temperature makes answers less random.
    llm = ChatOllama(
        model="gemma:2b",
        temperature=0.3
    )
    # Compose prompt, model, and parser into one pipeline that returns plain text.
    output_parser = StrOutputParser
    return prompt|llm|output_parser

# Reuse the cached chain so the model resources are not rebuilt on each Streamlit rerun.
chain = create_chain()
# --------------------------------------------------
# 4. Maintain conversation history
# --------------------------------------------------

# Session state survives Streamlit reruns, allowing previous turns to remain visible.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render each saved message so the conversation stays on screen after a rerun.
for message in   st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 5. Accept user input and generate a response
# --------------------------------------------------

# chat_input returns user text when submitted, or a false value when there is no new turn.
question = st.chat_input("What would you like to ask?")
if question:
    # Save and display the user's turn before invoking the model.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Show progress while the local model generates a reply; report runtime failures in the UI.
    try:
        with st.chat_message("assistant"):
            with st.spinner("Gemma is thinking..."):
                response = chain.invoke({"question": question})
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    except Exception as error:
        st.error(
            "Unable to generate a response. "
            "Check that Ollama is running and the model is installed."
        )
        st.caption(f"Error details: {error}")

# ============================================================
# REVISION NOTES — app.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This Streamlit app provides a browser-based chat interface for a locally running Gemma model through Ollama.
# It keeps the conversation in session state, sends each new question through a LangChain pipeline, and displays the text response.
#
# KEY CONCEPTS USED:
# • Prompt template: a reusable structure that combines system instructions with the current user question.
# • Model pipeline: connecting the prompt, chat model, and output parser so data flows through them in order.
# • Session state: Streamlit storage that preserves conversation messages across app reruns.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is the direct-chat example in 4_GenAI/2_ollama; Intro.py in the same folder demonstrates retrieval-augmented answers grounded in a web page.
#
# THINGS TO REMEMBER:
# • Ollama must be running locally and gemma:2b must be installed.
# • LangSmith tracing is optional and is enabled only when LANGCHAIN_API_KEY is available.
# • Streamlit reruns the script after user interactions, so session_state preserves and redraws the chat history.
#
# COMMON MISTAKES TO AVOID:
# • Treating StrOutputParser as an already-created parser instance; this file composes the parser class into its pipeline.
# • Expecting the chat history to persist after the app process ends; session state is for the active Streamlit session.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow chain.invoke({"question": question}): the question passes through the prompt, local model, and output parser.
# ============================================================