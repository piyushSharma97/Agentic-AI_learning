"""
Lesson 6: Message Trimming — Managing the Context Window

What you will learn in this lesson:
1. Why context management is critical: LLMs have finite context windows (maximum token limits).
2. How unconstrained conversation growth causes latency increases, high token bills, and context overflow errors.
3. How trim_messages automatically selects a bounded window of recent messages.
4. Key trimming parameters: max_tokens, strategy="last", token_counter, start_on="human", and include_system.
5. Best practice: preserving critical system instructions while trimming dynamic conversation history.
"""

import os
from dotenv import load_dotenv

# trim_messages is LangChain's built-in utility to prune message arrays to fit token budgets.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. Environment & Model Initialization
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# --------------------------------------------------
# 2. Simulated Long Conversation History
# --------------------------------------------------

# A conversation with multiple turns:
# As dialogues stretch to dozens or hundreds of messages, passing everything becomes infeasible.
messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="My favorite ice cream is vanilla."),
    AIMessage(content="Vanilla is a classic choice."),
    HumanMessage(content="I also enjoy hiking."),
    AIMessage(content="Hiking is a great outdoor activity."),
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(content="4."),
    HumanMessage(content="What is my favorite ice cream?"),
]

# Separate the system message from user/assistant chat history.
# System prompts contain rules/persona and should never be accidentally trimmed away.
system_message = messages[0]
conversation = messages[1:]

# --------------------------------------------------
# 3. Configuring trim_messages
# --------------------------------------------------

# trim_messages parameters explained:
# • max_tokens: Target maximum tokens to keep.
# • strategy="last": Keeps the most recent messages (sliding window) and drops older turns.
# • token_counter=model: Uses the model instance to accurately count tokens matching its tokenizer.
# • include_system=False: Excludes system messages from this trim operation (we prepend it separately).
# • allow_partial=False: Drops whole messages rather than cutting a message halfway through.
# • start_on="human": Guarantees the resulting slice begins with a human message, preventing
#   orphaned AI messages at the head of the context.
trimmed_conversation = trim_messages(
    conversation,
    max_tokens=350,
    strategy="last",
    token_counter=model,
    include_system=False,
    allow_partial=False,
    start_on="human",
)

# --------------------------------------------------
# 4. Inspecting Trimming Results & Invoking
# --------------------------------------------------

print("Messages before trimming:", len(conversation))
print("Messages after trimming:", len(trimmed_conversation))

# Re-assemble the system message with the trimmed conversation slice using list unpacking (*).
response = model.invoke([system_message, *trimmed_conversation])
print("\nAssistant:", response.content)

# ============================================================
# REVISION NOTES — 06_trim_history/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script demonstrates how to constrain conversation history using trim_messages.
# It splits a conversation into system instructions and dialogue turns, trims older dialogue messages
# to adhere to a token budget, and sends the bounded context to ChatGroq.
#
# KEY CONCEPTS USED:
# • Context Window: The maximum number of tokens an LLM can accept in a single request.
# • Sliding Window Strategy: Keeping the latest N tokens (strategy="last") so the bot remembers immediate context.
# • token_counter: Counting tokens using the actual model's tokenizer or an approximate counter.
# • start_on="human": Ensures proper conversation alternation so the model does not start with an orphaned AI reply.
# • allow_partial=False: Prevents truncation of individual sentences, ensuring coherent context.
#
# HOW IT CONNECTS TO THE PROJECT:
# In endless chat sessions, history accumulates indefinitely. Trimming is the essential guardrail
# that makes conversational agents viable for long-term production use.
#
# THINGS TO REMEMBER:
# • Always protect system instructions from trimming; either set include_system=True with adequate token budget
#   or keep the system message separate and prepend it manually.
# • strategy="last" is almost always the right choice for chatbots because users expect the bot to remember recent turns.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `start_on="human"`, which might leave an isolated `AIMessage` at the start of context, confusing the model.
# • Hardcoding a token limit that is too small for even a single question and answer.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Review the parameters of `trim_messages()`: `max_tokens`, `strategy="last"`, and `start_on="human"`.
# ============================================================
