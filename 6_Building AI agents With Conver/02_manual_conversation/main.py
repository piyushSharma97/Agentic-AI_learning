"""
Lesson 2: Manual Conversation — Simulating Memory by Passing History

What you will learn in this lesson:
1. Why LLMs are stateless and have no inherent memory between API calls.
2. How chat applications create conversational memory by re-sending prior messages.
3. The role of AIMessage in representing past assistant responses.
4. How the model uses the full conversation array to resolve referential questions ("What is my name?").
"""

import os
# os reads system environment variables such as GROQ_API_KEY.
from dotenv import load_dotenv
# python-dotenv loads keys from .env into os.environ.

# HumanMessage represents user statements; AIMessage represents previous model outputs.
from langchain_core.messages import HumanMessage, AIMessage
# ChatGroq provides fast cloud inference for Llama/Mixtral models.
from langchain_groq import ChatGroq

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
# 2. Assembling Conversation History Manually
# --------------------------------------------------

# Large Language Models are completely stateless:
# If you ask "Hi, my name is Krish" in call 1, and "What is my name?" in call 2,
# call 2 has NO knowledge of call 1!
#
# To enable memory, we must construct a list containing the full dialogue history:
# 1. Turn 1 Human: user introduces themselves
# 2. Turn 1 AI: assistant greeted them
# 3. Turn 2 Human: user asks a question referring back to Turn 1
conversation = [
    HumanMessage(content="Hi, my name is Krish."),
    AIMessage(content="Hi Krish! Nice to meet you."),
    HumanMessage(content="What is my name?"),
]

# --------------------------------------------------
# 3. Invoking the Model with Multi-Turn Context
# --------------------------------------------------

# When the model processes this list, it reads all three messages in sequence.
# It identifies that the human previously stated their name is "Krish" and responds accordingly.
response = model.invoke(conversation)

print("Assistant response:")
print(response.content)

# ============================================================
# REVISION NOTES — 02_manual_conversation/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script illustrates how multi-turn conversation works under the hood in modern LLM systems.
# It manually constructs a chronological sequence of HumanMessage and AIMessage objects and submits
# them together to ChatGroq so the model can correctly recall facts mentioned earlier.
#
# KEY CONCEPTS USED:
# • Statelessness: LLMs store zero state on the server between requests.
# • In-Context Memory: "Memory" is created entirely by passing previous dialogue turns as input tokens.
# • AIMessage: Message class representing prior outputs from the assistant, maintaining conversational balance.
# • Token Accumulation: As conversation length increases, the number of input tokens grows linearly.
#
# HOW IT CONNECTS TO THE PROJECT:
# This lesson bridges basic single-turn chat (Lesson 1) with automated session memory (Lesson 3).
# It demonstrates the exact data structure that LangChain's automated memory modules manage behind the scenes.
#
# THINGS TO REMEMBER:
# • Both the user's questions and the AI's responses must be included so the model maintains conversational continuity.
# • Because every past turn is re-sent, longer conversations cost more tokens and can eventually exceed context limits.
#
# COMMON MISTAKES TO AVOID:
# • Sending only HumanMessage objects without AIMessage replies, which confuses the conversational rhythm.
# • Assuming the API remembers earlier calls on its own without re-sending the message history.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at the conversation list: [HumanMessage, AIMessage, HumanMessage].
# Passing that whole list to model.invoke() is what gives the model memory.
# ============================================================
