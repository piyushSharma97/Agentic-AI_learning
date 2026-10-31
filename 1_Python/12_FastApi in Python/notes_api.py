"""
A notes API: path, query, JSON body, update, and delete.

Four ways data enters a function:

1. Path parameter — inside the URL. /notes/1 means note_id is 1.
   Use this when the value identifies one resource.

2. Query parameter — after the ?. /search?q=tools&limit=1
   Use this for filters and options. The resource is still "the search".

3. JSON body — a document sent with POST or PUT, not written in the URL.
   Use this when you are sending several fields, or a long piece of text.
   The parameter type is a Pydantic model (NoteIn). FastAPI reads the JSON,
   builds a NoteIn, and only then calls your function.

4. No input — GET /notes takes nothing. It returns the whole collection.

Status codes used here:
- 200 the default. The request worked.
- 201 set on POST /notes. A new record was created.
- 404 raised with HTTPException when the id is not in NOTES.
- 422 sent by FastAPI, not by our code, when the JSON fails validation.

NOTES is a dict in memory. It disappears when the process stops.
A real app would use the SQLite lesson instead of this dict.

Serve it from this folder with:

    uv run uvicorn notes_api:app --reload

JavaScript comparison:
- {note_id} in the path is :noteId in Express.
- q: str and limit: int = 2 are req.query.q and req.query.limit. The default and the int conversion are manual in Express.
- NoteIn is a zod schema in a typical Node app: z.object({ title: z.string().min(1), body: z.string() }).
- HTTPException(status_code=404) is res.status(404).json({ detail: "..." }).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Notes API")


class NoteIn(BaseModel):
    """The shape of JSON the client must send when creating or replacing a note.

    title must be a string with at least one character.
    body must be a string. It may be empty, because we did not set min_length.
    Extra keys in the JSON are ignored by default.
    """

    title: str = Field(min_length=1, description="Short name of the note")
    body: str = Field(description="The note text")


# Keys are note ids. Values are plain dicts, not NoteIn objects.
# A NoteIn is only the incoming request. We store the result of model_dump().
NOTES = {
    1: {"title": "RAG", "body": "Retrieve, then generate."},
    2: {"title": "Tools", "body": "The model calls a function."},
}


@app.get("/notes")
def list_notes():
    """Return every note. The path has no {id}, so this is the collection."""
    # A list of dicts becomes a JSON array of objects.
    return [{"id": note_id, **note} for note_id, note in NOTES.items()]


@app.get("/notes/{note_id}")
def get_note(note_id: int):
    """Return one note.

    FastAPI converts the path text "1" into the integer 1 because the
    argument is annotated as int. GET /notes/hello fails with 422,
    because "hello" is not an integer. That happens before this function runs.
    """
    note = NOTES.get(note_id)
    if note is None:
        # raise stops the function. FastAPI turns this into a JSON error
        # and sets the HTTP status to 404. The client can tell "missing"
        # apart from a normal note, which comes back as 200.
        raise HTTPException(status_code=404, detail=f"Note {note_id} was not found.")
    return {"id": note_id, **note}


@app.get("/search")
def search(q: str, limit: int = 2):
    """Filter notes.

    q is required. Callers who omit it get 422.
    limit defaults to 2 when the query string does not include it.
    Both values arrive as text in the URL. FastAPI converts limit to int.
    """
    matches = [
        {"id": note_id, **note}
        for note_id, note in NOTES.items()
        if q.lower() in note["title"].lower() or q.lower() in note["body"].lower()
    ]
    # [:limit] keeps the first matches and drops the rest.
    return {"query": q, "matches": matches[:limit]}


@app.post("/notes", status_code=201)
def create_note(note: NoteIn):
    """Create a note from a JSON body.

    status_code=201 changes the success code from 200 to 201 Created.
    FastAPI will not call this function if title is missing or empty.
    The client receives 422 in that case, and NOTES is unchanged.
    """
    new_id = max(NOTES) + 1 if NOTES else 1
    # model_dump() turns the validated model into a normal dict we can store.
    NOTES[new_id] = note.model_dump()
    return {"id": new_id, **NOTES[new_id]}


@app.put("/notes/{note_id}")
def replace_note(note_id: int, note: NoteIn):
    """Replace one existing note.

    PUT here means "write the whole note again", not "change one field".
    The id comes from the path. The new title and body come from the JSON body.
    FastAPI can fill both in the same call because they come from different places.
    """
    if note_id not in NOTES:
        raise HTTPException(status_code=404, detail=f"Note {note_id} was not found.")
    NOTES[note_id] = note.model_dump()
    return {"id": note_id, **NOTES[note_id]}


@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    """Remove one note. There is no JSON body. The id in the path is enough."""
    if note_id not in NOTES:
        raise HTTPException(status_code=404, detail=f"Note {note_id} was not found.")
    removed = NOTES.pop(note_id)
    return {"deleted": note_id, **removed}
