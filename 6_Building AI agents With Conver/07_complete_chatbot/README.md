
# Lesson 7 — Put the concepts together

## Goal
Build the complete learning version: prompt template + session history + trimming + language input.

## Run
```bash
python 07_complete_chatbot/main.py
```

Try:
1. Tell it your name.
2. Ask what your name is.
3. Change language using `/language English` or `/language Hindi`.
4. Use `/new` to start a new session.
5. Ask it about something mentioned much earlier, then experiment with the history budget.

## What to notice
This app is intentionally the most complex lesson. If it feels confusing, return to lessons 3–6 and then come back.

The trimming budget limits the conversation messages sent into the prompt. It does not delete messages from the stored session history.
