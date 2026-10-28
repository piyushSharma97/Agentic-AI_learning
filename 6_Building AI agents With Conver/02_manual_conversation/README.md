
# Lesson 2 — Send a conversation manually

## Goal
Learn that a chat model receives a list of messages, not just a plain string.

## Run
```bash
python 02_manual_conversation/main.py
```

## What to notice
The list contains a `HumanMessage`, then an `AIMessage`, then another `HumanMessage`. The model can use the messages included in this single request.

This is **manual context**, not automatic memory. You are responsible for sending earlier messages again.

## Try it
1. Change the name in the first message.
2. Change the AI reply you provide.
3. Remove the earlier messages and observe whether the model still knows the name.
