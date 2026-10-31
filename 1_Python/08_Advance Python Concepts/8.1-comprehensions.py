#!/usr/bin/env python
# coding: utf-8

"""
Comprehensions: build a list, dict, or set in one expression.

They replace a short loop whose only job is to collect values.
You will see them when filtering retrieved chunks or mapping rows to names.

JavaScript comparison:
- A list comprehension [name for row in scores if row["score"] >= 60] is scores.filter(row => row.score >= 60).map(row => row.name).
- A dict comprehension is Object.fromEntries(rows.map(row => [row.name, row.score])).
- A set comprehension is new Set(topics.map(topic => topic.toUpperCase())).
- JavaScript has no comprehension syntax. filter and map are the usual replacement.
"""

scores = [
    {"name": "Asha", "score": 88},
    {"name": "Ravi", "score": 54},
    {"name": "Neha", "score": 91},
    {"name": "Omar", "score": 47},
]

# --------------------------------------------------
# 1. List comprehension
# --------------------------------------------------
# [expression for item in iterable if condition]
# Read it as: "name, for each row, if the score is at least 60".

passing_names = [row["name"] for row in scores if row["score"] >= 60]
print("Passing:", passing_names)

# The long form of the same idea:
manual = []
for row in scores:
    if row["score"] >= 60:
        manual.append(row["name"])
print("Same result:", manual == passing_names)

# --------------------------------------------------
# 2. Dict comprehension
# --------------------------------------------------

by_name = {row["name"]: row["score"] for row in scores}
print("Lookup:", by_name["Neha"])

# --------------------------------------------------
# 3. Set comprehension
# --------------------------------------------------
# A set keeps unique values only.

topics = ["rag", "tools", "rag", "memory", "tools"]
unique_topics = {topic.upper() for topic in topics}
print("Unique topics:", unique_topics)

# --------------------------------------------------
# 4. Nested data, still one expression
# --------------------------------------------------

chunks = ["Agents use tools.", "Short.", "Retrieval finds relevant chunks."]
long_chunks = [text for text in chunks if len(text.split()) >= 3]
print("Long chunks:", long_chunks)
