#!/usr/bin/env python
# coding: utf-8

"""
Classes and objects.

A class is the blueprint. An object (also called an instance) is one
thing built from that blueprint.

Chat messages, documents, and Pydantic models in this course are all objects:
they bundle data (attributes) with behaviour (methods).

JavaScript comparison:
- class Document: is class Document { }.
- __init__(self, text, source) is constructor(text, source). Python writes self. JavaScript writes this.
- notes = Document(...) is const notes = new Document(...). JavaScript requires new. Python does not.
- A class attribute such as chunk_size is a field on the constructor in JavaScript: Document.chunkSize, or a static field.
"""

# --------------------------------------------------
# 1. Define a class and create objects
# --------------------------------------------------
# __init__ runs when you call Document(...). self is the new object.
# Attributes live on self, so each document keeps its own text.

class Document:
    def __init__(self, text, source):
        self.text = text
        self.source = source

    def preview(self, limit=30):
        if len(self.text) <= limit:
            return self.text
        return self.text[:limit] + "..."

    def word_count(self):
        return len(self.text.split())


notes = Document("Agents call tools and then read the tool result.", "notes.txt")
policy = Document("Refunds are allowed within 7 days.", "policy.pdf")

print(notes.source, "->", notes.preview())
print(policy.source, "words:", policy.word_count())

# Two objects from one class do not share attribute values.
print("Same class?", type(notes) is type(policy))
print("Same object?", notes is policy)

# --------------------------------------------------
# 2. Changing attributes
# --------------------------------------------------

notes.source = "notes-v2.txt"
print("Updated source:", notes.source)

# --------------------------------------------------
# 3. A class attribute is shared; an instance attribute is not
# --------------------------------------------------
# chunk_size is stored on the class. Every Chunk sees the same default
# until an instance sets its own value.

class Chunk:
    chunk_size = 500

    def __init__(self, text):
        self.text = text


first = Chunk("alpha")
second = Chunk("beta")
print("Shared default:", first.chunk_size, second.chunk_size)

first.chunk_size = 200  # this creates an instance attribute on first only
print("After override:", first.chunk_size, second.chunk_size)
