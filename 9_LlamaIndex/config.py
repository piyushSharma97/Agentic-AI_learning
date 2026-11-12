"""
Load the project .env before any LlamaIndex import that reads API keys.

LlamaIndex looks for OPENAI_API_KEY when it builds an index or a query engine.
The key lives in the repo root, not in this folder. Import this module first.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
