#!/usr/bin/env python
# coding: utf-8

"""
CSV and JSON files.

CSV is a table saved as text: one row per line, commas between columns.
JSON is the text format used by web APIs and chat message logs.
Both are plain files, so the file-handling rules from 6.1 still apply.

JavaScript comparison:
- csv.DictWriter writes a header and rows. JavaScript has no csv module in the language. People use a library, or join columns with commas themselves.
- json.dumps(data, indent=2) is JSON.stringify(data, null, 2).
- json.loads(text) is JSON.parse(text).
- Python's indent makes the file readable. JavaScript's third argument to stringify does the same.
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "practice_data"
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Write a CSV
# --------------------------------------------------
# newline="" is required on Windows so csv does not insert blank lines.
# DictWriter uses the dict keys as the column headers.

scores_path = DATA_DIR / "scores.csv"
rows = [
    {"name": "Asha", "topic": "functions", "score": 88},
    {"name": "Ravi", "topic": "files", "score": 91},
    {"name": "Neha", "topic": "classes", "score": 76},
]

with open(scores_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["name", "topic", "score"])
    writer.writeheader()
    writer.writerows(rows)

print("Wrote", scores_path.name)

# --------------------------------------------------
# 2. Read a CSV back into dicts
# --------------------------------------------------
# Each row is a dict of strings. Convert score before doing math.

with open(scores_path, "r", encoding="utf-8", newline="") as file:
    loaded = list(csv.DictReader(file))

for row in loaded:
    row["score"] = int(row["score"])

average = sum(row["score"] for row in loaded) / len(loaded)
print("Students:", [row["name"] for row in loaded])
print("Average score:", average)

# --------------------------------------------------
# 3. Write JSON
# --------------------------------------------------
# indent=2 makes the file readable. ensure_ascii=False keeps Unicode as-is.
# A chat history is a list of message dicts, which is valid JSON.

history_path = DATA_DIR / "chat_history.json"
history = [
    {"role": "system", "content": "You are a Python tutor."},
    {"role": "user", "content": "How do I read a JSON file?"},
]

history_path.write_text(
    json.dumps(history, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# --------------------------------------------------
# 4. Read JSON
# --------------------------------------------------

messages = json.loads(history_path.read_text(encoding="utf-8"))
print("Last role:", messages[-1]["role"])
print("Last content:", messages[-1]["content"])
