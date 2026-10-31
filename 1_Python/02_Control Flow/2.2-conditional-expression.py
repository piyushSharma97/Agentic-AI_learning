#!/usr/bin/env python
# coding: utf-8

"""
The conditional expression, and and/or used as values.

JavaScript comparison:
- Python:  value_if_true if condition else value_if_false
- JS:      condition ? value_if_true : value_if_false

The order is different. Python names the result first. JS names the test first.

and / or also return a value, not only True or False.
- Python x and y    ->  JS x && y
- Python x or y     ->  JS x || y
Both languages return one of the original operands, not a forced boolean.
"""

score = 72

# --------------------------------------------------
# 1. One expression instead of a four-line if
# --------------------------------------------------

label = "pass" if score >= 60 else "fail"
print("Label:", label)

# The if / else block form does the same job when you need several lines:
if score >= 60:
    long_label = "pass"
else:
    long_label = "fail"
print("Same label:", long_label)

# --------------------------------------------------
# 2. and / or return the deciding operand
# --------------------------------------------------
# Empty string is falsy, so or keeps going and returns the default.
# JavaScript: name || "friend"

name = ""
display = name or "friend"
print("Display name:", display)

# and stops on the first falsy value and returns it.
# If the first value is truthy, and returns the second value.
# JavaScript: user && user.city

user = {"city": "Delhi"}
print("City:", user and user["city"])

missing = None
print("Missing city:", missing and missing["city"])

# --------------------------------------------------
# 3. Chain a default the way JS uses ??
# --------------------------------------------------
# Python has no ?? operator.
# `or` replaces every falsy value, including 0 and "".
# When 0 is a real answer, check for None instead.
# JS nullish coalescing (??) only replaces null and undefined.

temperature = 0
print("or treats 0 as missing:", temperature or 0.7)
print("None check keeps 0:", 0.7 if temperature is None else temperature)
