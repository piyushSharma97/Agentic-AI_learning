#!/usr/bin/env python
# coding: utf-8

"""
Functions: define a block once, call it many times.

A function is a named recipe. You give it inputs (parameters), it does work,
and it can hand a result back with return.

In agent code this shows up everywhere:
- a tool is just a function the model is allowed to call
- a prompt builder is a function that returns a string
- a retriever wrapper is a function that takes a question and returns chunks

JavaScript comparison:
- Python def greet():. JavaScript function greet() { }.
- Python return total. JavaScript return total. Same keyword.
- A default looks the same: def f(language="English") and function f(language = "English").
- Calling by name, f(language="Hindi"), is a Python feature. JavaScript passes an object instead: f({ language: "Hindi" }).
- A function with no return gives None. A JavaScript function with no return gives undefined.
"""

# --------------------------------------------------
# 1. A function with no parameters
# --------------------------------------------------
# def starts the definition. The indented lines are the body.
# Nothing runs until you call the function by name.

def greet():
    print("Hello from a function.")


greet()
greet()  # the same body can run again

# --------------------------------------------------
# 2. Parameters and arguments
# --------------------------------------------------
# Parameters are the names inside the definition (name, city).
# Arguments are the values you pass at the call site.

def introduce(name, city):
    print(f"{name} is learning Python in {city}.")


introduce("Piyush", "India")

# --------------------------------------------------
# 3. return hands a value back
# --------------------------------------------------
# print shows text on the screen. return gives a value to the caller,
# so the caller can store it, test it, or pass it to another function.
# A function with no return gives back None.

def add(a, b):
    return a + b


total = add(25, 17)
print("25 + 17 =", total)

def silent():
    pass  # pass is a placeholder: the block is empty but still valid


print("A function with no return gives:", silent())

# --------------------------------------------------
# 4. Default parameter values
# --------------------------------------------------
# If the caller skips an argument, Python uses the default.
# Put parameters that have defaults after parameters that do not.

def make_prompt(question, language="English"):
    return f"Answer in {language}: {question}"


print(make_prompt("What is a function?"))
print(make_prompt("What is a function?", language="Hindi"))

# --------------------------------------------------
# 5. Keyword arguments
# --------------------------------------------------
# You can pass arguments by name. Order does not matter when names are used.
# This is how LangChain calls are written: model="...", temperature=0.

def build_model_config(provider, model, temperature=0):
    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
    }


config = build_model_config(model="llama-3.3-70b", provider="groq", temperature=0.2)
print("Model config:", config)

# --------------------------------------------------
# 6. Docstring: the function's own documentation
# --------------------------------------------------
# The string on the first line of the body is __doc__.
# Tool decorators in LangChain read this text and send it to the model
# so the model knows when to call the function.

def get_weather(location):
    """Return a short weather note for a city name."""
    return f"It is sunny in {location}."


print(get_weather("Boston"))
print("Docstring:", get_weather.__doc__)

# --------------------------------------------------
# 7. One function calling another
# --------------------------------------------------

def word_count(text):
    return len(text.split())


def is_too_long(text, limit=8):
    count = word_count(text)
    return count > limit


note = "Agents use functions to keep each step small and testable."
print("Word count:", word_count(note))
print("Too long for a title?", is_too_long(note))
