
# Lesson 3 — Automatic session history

## Goal
Learn how `RunnableWithMessageHistory` saves and reloads messages for a session.

## Run
```bash
python 03_session_memory/main.py
```

Try:
1. Tell the chatbot your name.
2. Ask for your name in a later message.
3. Enter `/new` and ask your name again.
4. Exit and restart the program.

## What to notice
- The `session_id` selects a conversation history.
- Reusing the same ID loads the same messages.
- A new ID starts a separate conversation.
- This lesson stores messages in memory, so restarting Python clears them.
