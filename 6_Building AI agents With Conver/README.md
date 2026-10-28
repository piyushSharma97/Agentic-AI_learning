# LangChain Learning Apps: One Concept at a Time

This project is intentionally split into **separate, runnable learning apps**. Each lesson has its own `main.py`. Run them one at a time, in order.

## Lessons

| Lesson | App | What you learn |
|---|---|---|
| 1 | `01_basic_chat/` | Send one message to the model |
| 2 | `02_manual_conversation/` | Send a sequence of human and AI messages |
| 3 | `03_session_memory/` | Let LangChain save messages per session |
| 4 | `04_prompt_template/` | Add reusable system instructions |
| 5 | `05_language_input/` | Pass extra variables, such as a response language |
| 6 | `06_trim_history/` | Trim an old list of messages before sending it |
| 7 | `07_complete_chatbot/` | Combine prompt templates, session history, and trimming |

Every lesson is a separate app. You can open only that lesson's folder and study its `main.py`, then compare it with the previous lesson.

## Setup once

Requires Python 3.10+ and a Groq API key.

### Create and activate a virtual environment

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies from the project root:
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:
```text
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Never share or commit your real `.env` file. If the default model is unavailable in your Groq account, change `GROQ_MODEL` to a model currently available to you.

## Run one app

From the project root, run one command at a time, for example:
```bash
python 01_basic_chat/main.py
python 02_manual_conversation/main.py
python 03_session_memory/main.py
python 04_prompt_template/main.py
python 05_language_input/main.py
python 06_trim_history/main.py
python 07_complete_chatbot/main.py
```

Some lessons run once and print the result; lessons 3 and 7 run an interactive chat. Stop an interactive app with `exit` or `quit`.

## How to learn from each lesson

For each app:
1. Read its `README.md`.
2. Run `main.py` without changing it.
3. Change one thing at a time.
4. Predict what you think will happen before running it again.
5. Compare the result with your prediction.
6. Move to the next lesson only when the current one makes sense.

## Important limitations

- The in-memory session store resets when the app exits. It is for learning, not durable storage.
- A session ID is not a security mechanism. A real app must authorize access to each user's session.
- Trimming history is not the same as long-term memory: messages omitted from the prompt may still exist in the in-memory history, but the model will not see them in that request.
- LangChain APIs and provider model names can change. If an API call fails, check the installed package version and the current Groq model list.
