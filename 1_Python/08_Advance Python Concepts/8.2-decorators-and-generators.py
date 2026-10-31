#!/usr/bin/env python
# coding: utf-8

"""
Decorators and generators.

A decorator wraps a function and adds behaviour before or after it runs.
@tool in LangChain is a decorator: it turns your function into a tool.

A generator yields values one at a time instead of building a full list.
That matters when a file or a stream is too large to hold in memory.

JavaScript comparison:
- A Python decorator is a function that wraps another function. JavaScript does the same by hand: const timed = (fn) => (...args) => { ... }.
- JavaScript also has a decorator proposal used by some frameworks. The @ syntax is not the everyday Node style yet. Python @timed is normal.
- yield exists in both. function* tokens() { yield word } is the JavaScript generator.
- A generator expression (len(word) for word in words) has no direct syntax. JS uses a generator function, or words.map(word => word.length), which builds the whole array immediately.
"""

import time

# --------------------------------------------------
# 1. A decorator is a function that receives a function
# --------------------------------------------------

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper


@timed
def word_count(text):
    return len(text.split())


print("Words:", word_count("decorators wrap a function without changing its call site"))

# --------------------------------------------------
# 2. functools.wraps keeps the original name and docstring
# --------------------------------------------------
# Without wraps, word_count.__name__ would become "wrapper".

import functools

def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


@logged
def normalize(text):
    """Lowercase and strip a query."""
    return text.strip().lower()


print(normalize("  RAG Pipeline  "))
print("Name preserved:", normalize.__name__)
print("Doc preserved:", normalize.__doc__)

# --------------------------------------------------
# 3. Generator function
# --------------------------------------------------
# yield pauses the function and hands one value to the caller.
# The next loop step resumes after that yield.

def tokens(text):
    for word in text.split():
        yield word.upper()


print("Tokens:", list(tokens("one chunk at a time")))

# --------------------------------------------------
# 4. Generator expression
# --------------------------------------------------
# Parentheses instead of square brackets. Nothing is computed until you iterate.

lengths = (len(word) for word in ["agent", "tool", "memory"])
print("First length:", next(lengths))
print("Rest:", list(lengths))
