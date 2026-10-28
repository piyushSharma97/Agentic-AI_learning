
# Lesson 4 — Prompt templates

## Goal
Learn how to put reusable system instructions in a `ChatPromptTemplate`.

## Run
```bash
python 04_prompt_template/main.py
```

## What to notice
- The `system` message sets the assistant's behavior.
- `MessagesPlaceholder` is a slot where a list of messages can be inserted.
- The prompt is a reusable template, not the model itself.

## Try it
Change the system instruction to make the model act as a friendly Python teacher. Then change the user question.
