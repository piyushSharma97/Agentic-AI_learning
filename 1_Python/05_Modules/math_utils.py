#!/usr/bin/env python
# coding: utf-8

"""
A module is a Python file other files can import.

Keep this file as reusable functions only.
The lesson that calls them is 5.1-importing-modules.py.

JavaScript comparison:
- This file is a module of functions. In JavaScript you would export each function: export function clamp(value, low, high) { ... }.
- max() and min() exist in both. Python max(low, min(value, high)). JavaScript Math.max(low, Math.min(value, high)).
"""


def clamp(value, low, high):
    """Keep value inside the inclusive range low..high."""
    return max(low, min(value, high))


def tokens_to_words(token_count, words_per_token=0.75):
    """Rough estimate: one token is about 0.75 English words."""
    return int(token_count * words_per_token)
