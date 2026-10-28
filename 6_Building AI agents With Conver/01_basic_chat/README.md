
# Lesson 1 — One message to the model

## Goal
Learn the smallest useful LangChain app: create a model, send a human message, print the answer.

## Run
From the project root:
```bash
python 01_basic_chat/main.py
```

## What to notice
- `ChatGroq(...)` configures the model connection.
- `HumanMessage(...)` represents a user message.
- `.invoke(...)` sends the message and waits for one response.
- The model does not automatically remember earlier calls.

## Try it
1. Change the message content.
2. Call `.invoke(...)` twice with unrelated questions. Does it remember the first call?
