"""Lesson 7: combine prompt templates, session history, and trimming."""

import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# STEP 1: store separate message histories for separate sessions.
session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# STEP 2: create the reusable prompt.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Be accurate and clear. "
        "Respond in {language}.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# STEP 3: trim history before it enters the prompt.
history_trimmer = trim_messages(
    max_tokens=1200,
    strategy="last",
    token_counter="approximate",
    include_system=False,
    allow_partial=False,
    start_on="human",
)

# STEP 4: assemble the processing pipeline.
chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | history_trimmer
    )
    | prompt
    | model
)

# STEP 5: wrap the chain so messages are loaded and saved automatically.
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

session_id = str(uuid4())
language = "English"

print("Complete chatbot demo.")
print("Commands: /new, /language English, /language Hindi, exit")

while True:
    user_text = input("\nYou: ").strip()

    if user_text.lower() in {"exit", "quit"}:
        break

    if user_text == "/new":
        session_id = str(uuid4())
        print("Started a new session.")
        continue

    if user_text.startswith("/language "):
        language = user_text.removeprefix("/language ").strip() or "English"
        print(f"Response language set to: {language}")
        continue

    if not user_text:
        continue

    try:
        response = chatbot.invoke(
            {"question": user_text, "language": language},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {response.content}")
    except Exception as error:
        print(f"Request failed: {error}")
