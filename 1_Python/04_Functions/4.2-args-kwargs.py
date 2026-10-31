#!/usr/bin/env python
# coding: utf-8

"""
*args and **kwargs: accept a flexible number of arguments.

*args collects extra positional arguments into a tuple.
**kwargs collects extra keyword arguments into a dictionary.

You will see both when wrapping model calls or tool functions,
because the wrapper does not always know every argument in advance.

JavaScript comparison:
- Python *args packs extra positional arguments into a tuple. JavaScript ...args packs them into an array.
- Python **kwargs packs named arguments into a dict. JavaScript has no keyword arguments. You pass one object and read its fields.
- Calling with *numbers unpacks a sequence. JavaScript spreads an array with ...numbers.
- Calling with **options unpacks a dict. JavaScript spreads an object with ...options into another object, not into separate parameters.
"""

# --------------------------------------------------
# 1. *args — extra positional values
# --------------------------------------------------
# The star packs the arguments. Inside the function, scores is a tuple.

def average(*scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)


print("Average of three scores:", average(80, 90, 70))
print("Average of one score:", average(100))

# --------------------------------------------------
# 2. Required parameter plus *args
# --------------------------------------------------

def label_scores(student, *scores):
    print(student, "scores:", scores, "average:", average(*scores))


label_scores("Asha", 88, 92, 79)

# --------------------------------------------------
# 3. **kwargs — extra named values
# --------------------------------------------------
# The double star packs names and values into a dict.

def show_metadata(**details):
    for key, value in details.items():
        print(f"  {key} = {value}")


print("Document metadata:")
show_metadata(source="notes.txt", page=3, chunk_id="c-12")

# --------------------------------------------------
# 4. Mixing normal parameters, *args, and **kwargs
# --------------------------------------------------
# Order must be: normal, *args, keyword-only, **kwargs.

def call_tool(name, *args, **kwargs):
    print(f"Tool {name} got positional args {args} and keywords {kwargs}")


call_tool("search", "langchain agents", k=3, source="web")

# --------------------------------------------------
# 5. Unpacking when you already have a list or dict
# --------------------------------------------------
# * unpacks a sequence into positional arguments.
# ** unpacks a dict into keyword arguments.

numbers = [10, 20, 30]
print("Unpacked average:", average(*numbers))

options = {"temperature": 0, "max_tokens": 200}
call_tool("summarize", "meeting notes", **options)
