
# Lesson 6 — Trim conversation history

## Goal
Learn why a chatbot should not always send an unlimited number of old messages to a model.

## Run
```bash
python 06_trim_history/main.py
```

## What to notice
- `trim_messages` chooses a smaller set of messages.
- `max_tokens` is a budget, not a count of messages.
- `strategy="last"` favors recent messages.
- The system message is kept separately in this demo.

This lesson trims a manually created list. Lesson 7 shows how trimming can fit into a complete chatbot pipeline.
