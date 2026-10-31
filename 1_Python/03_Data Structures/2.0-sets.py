#!/usr/bin/env python
# coding: utf-8

"""
Sets: a collection of unique values, with no indexes.

Use a set when the question is "have I already seen this?", not "what is item 3?"

JavaScript comparison:
- Python set                  ->  JS new Set()
- Python items.add(x)         ->  JS items.add(x)
- Python x in items           ->  JS items.has(x)
- Python items.remove(x)      ->  JS items.delete(x)
- Python len(items)           ->  JS items.size
- Python a | b  (union)       ->  JS new Set([...a, ...b])
- Python a & b  (intersection)->  no operator. Filter one set with has().
- JS Set remembers insertion order. A Python set does not promise order.
- There is no set[0]. Same in JS: a Set is not indexed.
"""

# --------------------------------------------------
# 1. Duplicates disappear
# --------------------------------------------------

topics = {"rag", "tools", "rag", "memory", "tools"}
print("Unique topics:", topics)
print("Count:", len(topics))

# Building a set from a list is the usual way to dedupe.
# JS: new Set(["rag", "tools", "rag"])

raw = ["rag", "tools", "rag", "memory"]
unique = set(raw)
print("From a list:", unique)

# --------------------------------------------------
# 2. Membership is the main operation
# --------------------------------------------------

print("Has memory?", "memory" in unique)
unique.add("agents")
unique.discard("missing")  # discard does nothing if the item is absent
print("After add:", unique)

# --------------------------------------------------
# 3. Union and intersection
# --------------------------------------------------

retrieved = {"chunk-1", "chunk-2", "chunk-3"}
allowed = {"chunk-2", "chunk-3", "chunk-9"}

print("Either set:", retrieved | allowed)
print("In both:", retrieved & allowed)
print("Only retrieved:", retrieved - allowed)
