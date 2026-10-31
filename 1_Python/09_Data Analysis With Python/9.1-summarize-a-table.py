#!/usr/bin/env python
# coding: utf-8

"""
Summarize a small table with the standard library.

Data analysis starts before pandas: load rows, convert types,
then compute a count, a mean, and a group breakdown.
statistics and csv are enough for that first pass.

JavaScript comparison:
- The csv module reads rows. JavaScript splits lines on \\n and cells on commas, or uses a library such as csv-parse.
- statistics.mean(scores) is scores.reduce((total, n) => total + n, 0) / scores.length.
- statistics.median has no built-in. You sort a copy and pick the middle.
- Grouping by topic is a loop into an object in both languages.
"""

import csv
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "practice_data"
DATA_DIR.mkdir(exist_ok=True)
TABLE = DATA_DIR / "study_hours.csv"

# --------------------------------------------------
# 1. Create a tiny dataset
# --------------------------------------------------

rows = [
    {"student": "Asha", "topic": "functions", "hours": "4.5", "score": "88"},
    {"student": "Ravi", "topic": "functions", "hours": "3.0", "score": "74"},
    {"student": "Neha", "topic": "classes", "hours": "6.0", "score": "93"},
    {"student": "Omar", "topic": "classes", "hours": "2.5", "score": "61"},
    {"student": "Asha", "topic": "files", "hours": "5.0", "score": "90"},
]

with open(TABLE, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["student", "topic", "hours", "score"])
    writer.writeheader()
    writer.writerows(rows)

# --------------------------------------------------
# 2. Load and convert text columns to numbers
# --------------------------------------------------
# CSV has no number type. Everything comes back as a string.

with open(TABLE, "r", encoding="utf-8", newline="") as file:
    records = list(csv.DictReader(file))

for record in records:
    record["hours"] = float(record["hours"])
    record["score"] = int(record["score"])

print("Rows loaded:", len(records))

# --------------------------------------------------
# 3. Overall summary
# --------------------------------------------------

scores = [record["score"] for record in records]
hours = [record["hours"] for record in records]

print("Score mean:", statistics.mean(scores))
print("Score median:", statistics.median(scores))
print("Hours total:", sum(hours))

# --------------------------------------------------
# 4. Group by topic
# --------------------------------------------------
# This is the same question a pivot table answers: average score per topic.

totals = {}
counts = {}
for record in records:
    topic = record["topic"]
    totals[topic] = totals.get(topic, 0) + record["score"]
    counts[topic] = counts.get(topic, 0) + 1

print("--- average score by topic ---")
for topic in totals:
    print(f"{topic}: {totals[topic] / counts[topic]:.1f}")
