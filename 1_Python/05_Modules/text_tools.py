#!/usr/bin/env python
# coding: utf-8

"""
Second small module, so the import lesson can show two import styles.

JavaScript comparison:
- " ".join(text.lower().split()) trims and collapses spaces. JavaScript text.trim().toLowerCase().split(" ").join(" ") is the simple match.
- text[:limit] is text.slice(0, limit).
"""


def normalize(text):
    """Lowercase and collapse extra spaces. Useful before storing a search query."""
    return " ".join(text.lower().split())


def preview(text, limit=40):
    """Shorten text for logs without cutting the whole document."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
