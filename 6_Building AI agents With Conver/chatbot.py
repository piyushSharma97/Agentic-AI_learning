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

import os
# os reads environment variables such as GROQ_API_KEY from the operating system.
from dotenv import load_dotenv
# python-dotenv loads environment variables from a .env file into the process environment.
load_dotenv() 

# --------------------------------------------------
# 1. API Configuration & Model Initialization
# --------------------------------------------------

# Fetch the Groq API key from environment variables.
groq_api_key = os.getenv("GROQ_API_KEY")

# ChatGroq connects LangChain to Groq's high-speed inference engine.
# Here we use Gemma2-9b-It (or another Groq-supported chat model).
from langchain_groq import ChatGroq
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# --------------------------------------------------
# 2. Stateless Nature of LLMs
# --------------------------------------------------
# LLMs are inherently stateless: each call to invoke() is independent.
# The model does not remember past calls unless we explicitly send the history back.

from langchain_core.messages import HumanMessage
# A single user message sent to the model.
response_single = model.invoke([HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")])
print("Single turn response:\n", response_single.content)

# --------------------------------------------------
# 3. Manual Conversation History
# --------------------------------------------------
# To give the illusion of memory, we must pass the entire previous conversation transcript
# (both HumanMessage and AIMessage) in every subsequent request.

from langchain_core.messages import AIMessage
manual_conversation = [
    HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer"),
    AIMessage(content="Hello Krish! It's nice to meet you. \n\nAs a Chief AI Engineer, what kind of projects are you working on these days? \n\nI'm always eager to learn more about the exciting work being done in the field of AI.\n"),
    HumanMessage(content="Hey What's my name and what do I do?")
]

response_manual = model.invoke(manual_conversation)
print("\nManual conversation response:\n", response_manual.content)

# --------------------------------------------------
# 4. Automated Memory with RunnableWithMessageHistory
# --------------------------------------------------
# Instead of managing message lists manually in our code, LangChain provides
# RunnableWithMessageHistory to automatically append user inputs and model outputs
# to a stateful message history datastore.

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

# Wrap the model with message history management.
# Every time this runnable is invoked, it retrieves history using get_session_history,
# appends the input messages, calls the model, and appends the model's reply.
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Config specifies which session this invocation belongs to.
config = {"configurable": {"session_id": "chat1"}}

# Turn 1 in session 'chat1'
response = with_message_history.invoke(
    [HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")],
    config=config
)
print("\nSession chat1 - Turn 1:\n", response.content)

# Turn 2 in session 'chat1': The model now remembers the name from Turn 1!
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print("\nSession chat1 - Turn 2:\n", response.content)

# --------------------------------------------------
# 5. Session Isolation (chat1 vs chat2)
# --------------------------------------------------
# Different session IDs maintain completely separate conversation memories.

config1 = {"configurable": {"session_id": "chat2"}}

# In session 'chat2', the model does NOT know who we are yet.
response = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
print("\nSession chat2 - Turn 1 (isolated):\n", response.content)

# Introduce a new user 'John' in chat2.
response = with_message_history.invoke(
    [HumanMessage(content="Hey My name is John")],
    config=config1
)
print("\nSession chat2 - Turn 2:\n", response.content)

# Now chat2 remembers 'John', while chat1 still remembers 'Krish'.
response = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
print("\nSession chat2 - Turn 3 (remembers John):\n", response.content)

# --------------------------------------------------
# 6. Integrating Prompt Templates & System Instructions
# --------------------------------------------------
# To give our assistant a consistent persona or behavior instructions,
# we use ChatPromptTemplate with a MessagesPlaceholder for the history.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
print("\nPrompt Template Chain direct test:\n", test_response.content)

# Wrap the prompt+model chain with session history
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "chat3"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi My name is Krish")],
    config=config
)
print("\nSession chat3 - Turn 1:\n", response.content)

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print("\nSession chat3 - Turn 2 (recalls Krish):\n", response.content)

# --------------------------------------------------
# 7. Multi-Variable Prompts (e.g., Language Parameter)
# --------------------------------------------------
# When a prompt template requires variables beyond just messages (such as {language}),
# we must inform RunnableWithMessageHistory which input key contains the messages.

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
response = chain.invoke({"messages": [HumanMessage(content="Hi My name is Krish")], "language": "Hindi"})
print("\nDirect multi-variable chain (Hindi):\n", response.content)

# When using multiple input keys, input_messages_key specifies which key holds the user's messages.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable": {"session_id": "chat4"}}
response = with_message_history.invoke(
    {'messages': [HumanMessage(content="Hi, I am Krish")], "language": "Hindi"},
    config=config
)
print("\nSession chat4 - Turn 1 (in Hindi):\n", response.content)

# --------------------------------------------------
# 8. Managing Conversation History & Message Trimming
# --------------------------------------------------
# Long conversations can exceed the LLM's context window limit (max tokens)
# and increase API latency/cost. trim_messages keeps only the most recent tokens.

from langchain_core.messages import SystemMessage, trim_messages

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
print(f"\nSample messages before trim: {len(sample_messages)} -> after trim: {len(trimmed_sample)}")

# --------------------------------------------------
# 9. Integrating Trimmer into LCEL Pipeline
# --------------------------------------------------
# RunnablePassthrough.assign modifies or adds a specific key to the input dictionary.
# Here it intercepts 'messages', pipes it through the trimmer, and updates 'messages' before the prompt.

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": sample_messages + [HumanMessage(content="What ice cream do i like")],
        "language": "English"
    }
)
print("\nResponse after trimming (ice cream question):\n", response.content)

response = chain.invoke(
    {
        "messages": sample_messages + [HumanMessage(content="what math problem did i ask")],
        "language": "English",
    }
)
print("\nResponse after trimming (math problem question):\n", response.content)

# --------------------------------------------------
# 10. Complete Production-Ready Chatbot Chain
# --------------------------------------------------
# Combining trimming, prompt templates, language parameters, and automatic session history!

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "chat5"}}

response = with_message_history.invoke(
    {
        "messages": sample_messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)
print("\nSession chat5 - Turn 1:\n", response.content)

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask?")],
        "language": "English",
    },
    config=config,
)
print("\nSession chat5 - Turn 2:\n", response.content)

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
