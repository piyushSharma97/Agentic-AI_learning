#!/usr/bin/env python
# coding: utf-8

"""
match / case, and the JavaScript switch.

match compares one value against patterns, top to bottom, and runs
the first matching block.

JavaScript comparison:
- Python match / case     ->  JS switch / case
- Python case _:          ->  JS default:
- Python does not fall through. You do not write break.
  JS switch DOES fall through unless you break.
- Python can match a shape, not only a number or string.
  JS switch is usually a value. Destructuring in a separate if is the
  closer match for structure patterns.
"""

# --------------------------------------------------
# 1. Match a string — the switch you already know
# --------------------------------------------------

def describe_role(role):
    match role:
        case "system":
            return "instructions for the model"
        case "user":
            return "the human turn"
        case "assistant":
            return "the model turn"
        case _:
            return "unknown role"


for role in ("user", "tool", "system"):
    print(role, "->", describe_role(role))

# --------------------------------------------------
# 2. Match a number, with a guard
# --------------------------------------------------
# if on a case is an extra test. JS would use if inside the case,
# or skip switch and use if / else.

def band(score):
    match score:
        case s if s >= 90:
            return "A"
        case s if s >= 60:
            return "pass"
        case _:
            return "retry"


print("Score 95:", band(95))
print("Score 70:", band(70))
print("Score 40:", band(40))

# --------------------------------------------------
# 3. Match a sequence shape
# --------------------------------------------------
# [command, target] matches a list of exactly two items.
# JS has no switch pattern for this. You would write:
#   if (Array.isArray(event) && event.length === 2) { const [command, target] = event }

def run(event):
    match event:
        case ["search", query]:
            return f"search for {query}"
        case ["open", path]:
            return f"open {path}"
        case _:
            return "ignore"


print(run(["search", "langchain"]))
print(run(["open", "notes.txt"]))
print(run(["noop"]))
