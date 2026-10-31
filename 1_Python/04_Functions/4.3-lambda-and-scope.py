#!/usr/bin/env python
# coding: utf-8

"""
Lambda functions and variable scope.

A lambda is a small anonymous function used for a short expression.
Scope is the set of names a line of code is allowed to see:
local names live inside a function, global names live at module level.

JavaScript comparison:
- Python lambda n: n * n. JavaScript (n) => n * n. Both are one expression.
- Python names inside a function are local. JavaScript let and const are local to the block, which is stricter than Python: a Python name assigned in an if still exists after the if.
- Python global count reaches the module name. JavaScript has no global keyword. A name declared outside any function is already visible, and omitting let makes an accidental global.
- A closure works the same way: an inner function keeps the variables of the function that built it.
"""

# --------------------------------------------------
# 1. lambda — one expression, no name required
# --------------------------------------------------
# lambda arguments: expression
# This is the same idea as def square(n): return n * n
# Use def when the function needs a name, a docstring, or more than one line.

square = lambda n: n * n
print("Square of 6:", square(6))

# sorted() can take a key function. A lambda is a common way to write that key.
chunks = [
    {"text": "long chunk about agents and tools", "score": 0.42},
    {"text": "short", "score": 0.91},
    {"text": "medium length note", "score": 0.77},
]
best_first = sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)
print("Highest score first:", [item["score"] for item in best_first])

# --------------------------------------------------
# 2. Local scope
# --------------------------------------------------
# city inside describe() is a different name from any city outside.

city = "Delhi"

def describe():
    city = "Mumbai"  # local variable, hides the global one inside this function
    print("Inside function:", city)


describe()
print("Outside function:", city)

# --------------------------------------------------
# 3. Reading a global is allowed; assigning creates a local
# --------------------------------------------------
count = 0

def read_global():
    print("Can read global count:", count)


read_global()

def increment():
    # global tells Python: the assignment below updates the module-level name.
    global count
    count += 1


increment()
increment()
print("Count after two increments:", count)

# --------------------------------------------------
# 4. Enclosing scope (a function inside a function)
# --------------------------------------------------
# The inner function remembers prefix even after make_tag() has returned.
# That remembered binding is a closure. Decorators and tool wrappers use this.

def make_tag(prefix):
    def tag(message):
        return f"[{prefix}] {message}"
    return tag


info = make_tag("INFO")
print(info("model loaded"))
print(info("retrieval finished"))
