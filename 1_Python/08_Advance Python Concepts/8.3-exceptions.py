#!/usr/bin/env python
# coding: utf-8

"""
Exceptions: expected failures you can handle.

An exception is how Python reports that a line could not finish.
try / except lets the program decide what to do instead of crashing.
else runs when the try block succeeded. finally runs either way.

Agent loops use this pattern around tool calls and model requests:
catch the failure, log it, and return a clear message.

JavaScript comparison:
- try / except / else / finally. JavaScript try / catch / finally. JavaScript has no else on try.
- except ValueError as error. JavaScript catch (error). The type is checked inside the catch with instanceof, not in the catch header.
- raise ValueError("...") is throw new Error("...").
- A custom class MissingApiKey(Exception) is class MissingApiKey extends Error.
"""

# --------------------------------------------------
# 1. Catch a specific error
# --------------------------------------------------
# Catch the narrowest type you know how to handle.
# A bare except hides bugs.

def read_score(text):
    try:
        score = int(text)
    except ValueError:
        print(f"'{text}' is not a whole number")
        return None
    else:
        print("Parsed score:", score)
        return score


read_score("91")
read_score("high")

# --------------------------------------------------
# 2. Raise your own exception
# --------------------------------------------------

def chunk_text(text, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [text[index:index + size] for index in range(0, len(text), size)]


print("Chunks:", chunk_text("abcdef", 2))

try:
    chunk_text("abcdef", 0)
except ValueError as error:
    print("Caught:", error)

# --------------------------------------------------
# 3. finally always runs
# --------------------------------------------------
# Use it to release something you opened: a file, a connection, a lock.
# The with statement is the usual way to do that for files.

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    finally:
        print("division attempt finished")


print("Result:", divide(10, 2))
print("Result:", divide(10, 0))

# --------------------------------------------------
# 4. A custom exception type
# --------------------------------------------------
# Subclass Exception so callers can catch your error and ignore others.

class MissingApiKey(Exception):
    pass


def require_key(value):
    if not value:
        raise MissingApiKey("GROQ_API_KEY is missing")
    return value


try:
    require_key("")
except MissingApiKey as error:
    print("Setup problem:", error)
