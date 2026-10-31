#!/usr/bin/env python
# coding: utf-8

"""
SQLite with the standard library: create, read, update, delete.

sqlite3 talks to a SQLite database. This lesson uses a file next to the
script so you can reopen it. The same SQL works against an in-memory
database if you connect to ":memory:" instead.

Chat history, session ids, and small logs are common reasons an agent
app opens a database instead of keeping a dict in RAM.

JavaScript comparison:
- sqlite3 is Python's built-in SQLite. Node uses a package such as better-sqlite3. The SQL itself is the same.
- connection.execute("... ?", (value,)) is db.prepare(sql).run(value). The ? placeholder exists in both and keeps values out of the SQL text.
- fetchall() is .all() in better-sqlite3.
- with sqlite3.connect(...) closes the connection. JavaScript calls db.close() in a finally block.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "practice_data" / "notes.db"
DB_PATH.parent.mkdir(exist_ok=True)

# Start from an empty file on every run so the lesson is repeatable.
if DB_PATH.exists():
    DB_PATH.unlink()

# --------------------------------------------------
# 1. Connect and create a table
# --------------------------------------------------
# with closes the connection when the block ends.
# ? placeholders keep values out of the SQL string.

with sqlite3.connect(DB_PATH) as connection:
    connection.execute(
        """
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------
    # 2. Insert
    # --------------------------------------------------
    connection.execute(
        "INSERT INTO notes (title, body) VALUES (?, ?)",
        ("RAG", "Retrieve chunks, then answer from them."),
    )
    connection.executemany(
        "INSERT INTO notes (title, body) VALUES (?, ?)",
        [
            ("Tools", "The model asks to call a Python function."),
            ("Memory", "Past turns are stored and sent back later."),
        ],
    )

    # --------------------------------------------------
    # 3. Read
    # --------------------------------------------------
    print("--- all notes ---")
    for row in connection.execute("SELECT id, title, body FROM notes ORDER BY id"):
        print(row)

    # --------------------------------------------------
    # 4. Update
    # --------------------------------------------------
    connection.execute(
        "UPDATE notes SET body = ? WHERE title = ?",
        ("Past turns are trimmed when the prompt gets long.", "Memory"),
    )

    # --------------------------------------------------
    # 5. Delete
    # --------------------------------------------------
    connection.execute("DELETE FROM notes WHERE title = ?", ("Tools",))

    print("--- after update and delete ---")
    rows = connection.execute("SELECT title, body FROM notes ORDER BY id").fetchall()
    for title, body in rows:
        print(f"{title}: {body}")

print("Database file:", DB_PATH)
