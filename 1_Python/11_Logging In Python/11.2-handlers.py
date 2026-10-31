#!/usr/bin/env python
# coding: utf-8

"""
Handlers: where a log line goes.

A logger decides whether a message is important enough.
A handler decides where an accepted message is written.
This lesson sends INFO and above to the console, and also writes
every level to a file so you can open the file after the script ends.

JavaScript comparison:
- A StreamHandler writes to the terminal. That is console.log.
- A FileHandler writes to a path. Node uses fs.createWriteStream, or a pino transport that writes a file.
- setLevel on one handler is a filter. pino can use a different level per transport.
- Python's logging module wires this without an extra package. JavaScript needs a library, or a few lines of fs, to get the same split.
"""

import logging
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "practice_data" / "app.log"
LOG_PATH.parent.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Build a logger that does not use basicConfig
# --------------------------------------------------
# basicConfig only affects the root logger, and only the first time.
# An app usually creates its own logger, sets the level, and attaches handlers.
# propagate=False stops the same line from also being printed by the root logger.

logger = logging.getLogger("agent.file")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file_handler)

# --------------------------------------------------
# 2. DEBUG reaches the file only. INFO reaches both.
# --------------------------------------------------

logger.debug("embedding cache hit")
logger.info("answering the user")
logger.error("tool timeout")

print("--- file contents ---")
print(LOG_PATH.read_text(encoding="utf-8"))
