#!/usr/bin/env python
# coding: utf-8

"""
Dictionaries: a key points at a value.

Lookups use the key, not a position. Order of insertion is kept.

JavaScript comparison:
- Python dict                         ->  JS object, when every key is a string
- Python dict with any hashable key   ->  JS Map
- Python user["city"]                 ->  JS user.city  or  user["city"]
- Python user.get("city", "unknown")  ->  JS user.city ?? "unknown"
- Python "city" in user               ->  JS Object.hasOwn(user, "city")
  ("city" in user in JS also sees inherited keys. Python `in` does not.)
- Python user.keys() / values() / items()
                                      ->  JS Object.keys / values / entries
- Missing key: Python raises KeyError. JS returns undefined.
"""

# --------------------------------------------------
# 1. Create and read
# --------------------------------------------------

user = {
    "name": "Piyush",
    "city": "Delhi",
    "score": 91,
}
print("Name:", user["name"])
print("Keys:", list(user.keys()))
print("Items:", list(user.items()))

# --------------------------------------------------
# 2. A missing key
# --------------------------------------------------
# get returns the default instead of raising KeyError.
# JS would give undefined, and ?? would supply the default.

print("Theme:", user.get("theme", "light"))

# --------------------------------------------------
# 3. Add, update, delete
# --------------------------------------------------
# Same idea as JS: user.theme = "dark"; delete user.score

user["theme"] = "dark"
user["score"] = 95
removed = user.pop("city")
print("Removed city:", removed)
print("Now:", user)

# --------------------------------------------------
# 4. Loop
# --------------------------------------------------
# JS: for (const [key, value] of Object.entries(user))

print("--- fields ---")
for key, value in user.items():
    print(f"{key} = {value}")

# --------------------------------------------------
# 5. A dict of lists — common for grouped rows
# --------------------------------------------------

grades = {
    "Asha": [88, 91],
    "Ravi": [74],
}
grades["Asha"].append(95)
print("Asha grades:", grades["Asha"])
