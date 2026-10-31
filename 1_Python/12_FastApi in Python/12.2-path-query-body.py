#!/usr/bin/env python
# coding: utf-8

"""
Exercise every route in notes_api.py.

Read this file next to notes_api.py. Each block sends one request and
prints the status code and the JSON body.

Path, query, and body, in one glance:
- Path:   /notes/1                         the 1 is note_id
- Query:  /search?q=function&limit=1      q and limit are function arguments
- Body:   JSON sent with POST or PUT      validated as a NoteIn before the function runs

Status codes this lesson prints:
- 200  success
- 201  created (POST /notes)
- 404  the id is not in NOTES (HTTPException)
- 422  the path, query, or JSON did not match the types (FastAPI rejects it)

Serve the app from this folder with:

    uv run uvicorn notes_api:app --reload

JavaScript comparison:
- /notes/{note_id} is Express /notes/:noteId. Python names it in the function argument. Express reads req.params.noteId.
- A query parameter q: str is req.query.q. FastAPI converts the type. Express leaves query values as strings until you convert them.
- A Pydantic body is req.body after express.json(). FastAPI rejects a bad body with 422 before your function runs. Express enters the function and you check the body yourself, unless you add a validator such as zod.
"""

from fastapi.testclient import TestClient

from notes_api import app

client = TestClient(app)


def show(label, response):
    """Print one request's status and JSON so each example reads the same way."""
    print(label)
    print("  status:", response.status_code)
    print("  body:", response.json())


# --------------------------------------------------
# 1. Collection and one item
# --------------------------------------------------
# GET /notes has no id. GET /notes/1 puts 1 in the path.

show("GET /notes", client.get("/notes"))
show("GET /notes/1", client.get("/notes/1"))

# 99 is a valid integer, so the function runs, then raises 404.
missing = client.get("/notes/99")
show("GET /notes/99", missing)

# "abc" is not an int. FastAPI stops with 422 and get_note() does not run.
not_an_id = client.get("/notes/abc")
show("GET /notes/abc", not_an_id)

# --------------------------------------------------
# 2. Query string
# --------------------------------------------------
# params= builds ?q=function&limit=1
# The second call omits limit, so the function uses its default of 2.

show("GET /search?q=function&limit=1", client.get("/search", params={"q": "function", "limit": 1}))
show("GET /search?q=the", client.get("/search", params={"q": "the"}))

# q is required. An empty query string is 422.
show("GET /search  (no q)", client.get("/search"))

# --------------------------------------------------
# 3. JSON body — create
# --------------------------------------------------
# json= is the body. It is not part of the URL.
# A valid body returns 201 because the route set status_code=201.

created = client.post("/notes", json={"title": "Memory", "body": "Keep prior turns."})
show("POST /notes", created)
new_id = created.json()["id"]

# title is an empty string. Field(min_length=1) fails. Status 422.
# The function create_note is not called, so no new row appears.
rejected = client.post("/notes", json={"title": "", "body": "no title"})
show("POST /notes with an empty title", rejected)

# The key "title" is missing entirely. That is also 422.
show("POST /notes with no title", client.post("/notes", json={"body": "only a body"}))

# --------------------------------------------------
# 4. Replace and delete
# --------------------------------------------------
# PUT sends a path id AND a JSON body. Both arrive as function arguments.

show(
    f"PUT /notes/{new_id}",
    client.put(
        f"/notes/{new_id}",
        json={"title": "Memory", "body": "Trim history when the prompt is long."},
    ),
)

# Read it back. The body should be the replaced text, not the original.
show(f"GET /notes/{new_id} after replace", client.get(f"/notes/{new_id}"))

show(f"DELETE /notes/{new_id}", client.delete(f"/notes/{new_id}"))

# The id was removed, so the next read is 404.
show(f"GET /notes/{new_id} after delete", client.get(f"/notes/{new_id}"))

# Deleting an id that is already gone is also 404.
show("DELETE /notes/99", client.delete("/notes/99"))
