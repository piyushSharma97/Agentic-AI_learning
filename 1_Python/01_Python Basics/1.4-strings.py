#!/usr/bin/env python
# coding: utf-8

"""
Strings, and the JavaScript equivalent.

A string is an ordered sequence of characters. In both Python and JavaScript
the text itself cannot be changed in place. Methods return a new string.

JavaScript comparison:
- Python f"Hello {name}"          ->  JS `Hello ${name}`
- Python len(text)                ->  JS text.length
- Python text[0]                  ->  JS text[0]
- Python text[-1]                 ->  JS text.at(-1)
- Python "ai" in text             ->  JS text.includes("ai")
- Python text.strip()             ->  JS text.trim()
- Python text.lower()             ->  JS text.toLowerCase()
- Python text.split(",")          ->  JS text.split(",")
- Python "ha" * 3                 ->  no JS equivalent ("ha".repeat(3) is the method)
- Python and JS both use quotes.  JS also has backticks for templates.
"""

name = "Piyush"
language = "Python"

# --------------------------------------------------
# 1. Building a string
# --------------------------------------------------
# An f-string evaluates the expressions inside { } .
# JavaScript does the same inside ${ } of a template literal.

greeting = f"{name} is learning {language}"
print(greeting)

# --------------------------------------------------
# 2. Index and length
# --------------------------------------------------
# Indexes start at 0 in both languages.
# Python allows a negative index. -1 is the last character.
# JavaScript uses text.at(-1) for that. text[-1] in JS is just undefined.

word = "agent"
print("Length:", len(word))
print("First:", word[0])
print("Last:", word[-1])

# --------------------------------------------------
# 3. Slice — Python's extra
# --------------------------------------------------
# word[1:4] takes indexes 1, 2, and 3. The end index is not included.
# JavaScript: word.slice(1, 4)

print("Slice [1:4]:", word[1:4])

# --------------------------------------------------
# 4. Membership and common methods
# --------------------------------------------------

sentence = "  Retrieval Augmented Generation  "
print("Contains 'Augmented'?", "Augmented" in sentence)
print("Stripped:", sentence.strip())
print("Lower:", sentence.strip().lower())
print("Words:", sentence.split())

# replace returns a new string. The original sentence is unchanged.
print("Replaced:", sentence.strip().replace("Generation", "Gen"))

# --------------------------------------------------
# 5. Repeat — Python only, as an operator
# --------------------------------------------------

print("ha" * 3)

# --------------------------------------------------
# 6. Strings are immutable
# --------------------------------------------------
# word[0] = "A" would raise TypeError.
# JavaScript also refuses: a string index cannot be assigned.

updated = "A" + word[1:]
print("New string:", updated, "| original:", word)
