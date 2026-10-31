#!/usr/bin/env python
# coding: utf-8

"""
Text files: write, read, and append.

open(path, mode) connects your program to a file on disk.
Modes you will use most:
- "w" write (creates the file, or erases it if it already exists)
- "r" read
- "a" append (adds to the end, keeps what is already there)

Always close the file. The with statement does that for you,
even if a line inside the block raises an error.

JavaScript comparison:
- open(path, "w") is close to fs.writeFileSync(path, text) from node:fs.
- open(path, "r") and file.read() is fs.readFileSync(path, "utf8").
- Mode "a" is fs.appendFileSync.
- Python with closes the file for you. JavaScript sync calls do not leave a handle open. An fs.promises file handle uses await handle.close() in a finally block.
- Path.read_text() is the short form of readFileSync. pathlib is Python's path module.
"""

from pathlib import Path

# Files created by this lesson live next to the script, in practice_data/.
# Path(__file__) keeps the location correct no matter which folder you run from.

DATA_DIR = Path(__file__).resolve().parent / "practice_data"
DATA_DIR.mkdir(exist_ok=True)
NOTES = DATA_DIR / "notes.txt"

# --------------------------------------------------
# 1. Write a new text file
# --------------------------------------------------
# write() does not add a newline unless you put \n in the string.

with open(NOTES, "w", encoding="utf-8") as file:
    file.write("LangChain loads text before it splits it.\n")
    file.write("Each document has page_content and metadata.\n")

print("Wrote:", NOTES.name)

# --------------------------------------------------
# 2. Read the whole file
# --------------------------------------------------

with open(NOTES, "r", encoding="utf-8") as file:
    content = file.read()

print("--- full file ---")
print(content)

# --------------------------------------------------
# 3. Read line by line
# --------------------------------------------------
# A file object is iterable. Each loop step is one line, including \n.
# strip() removes that trailing newline for cleaner printing.

print("--- lines ---")
with open(NOTES, "r", encoding="utf-8") as file:
    for number, line in enumerate(file, start=1):
        print(number, line.strip())

# --------------------------------------------------
# 4. Append without deleting existing text
# --------------------------------------------------

with open(NOTES, "a", encoding="utf-8") as file:
    file.write("Append adds a third line.\n")

print("--- after append ---")
print(NOTES.read_text(encoding="utf-8"))

# --------------------------------------------------
# 5. pathlib shortcuts
# --------------------------------------------------
# Path.read_text and Path.write_text cover the common case
# of "whole file in, whole file out" without an open() call.

summary = DATA_DIR / "summary.txt"
summary.write_text("One paragraph is enough for a small note.\n", encoding="utf-8")
print("Summary exists:", summary.exists(), "size:", summary.stat().st_size, "bytes")
