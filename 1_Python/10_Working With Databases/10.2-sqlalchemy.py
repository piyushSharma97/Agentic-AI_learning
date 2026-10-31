#!/usr/bin/env python
# coding: utf-8

"""
SQLAlchemy: tables as Python classes.

sqlite3 makes you write SQL strings. SQLAlchemy maps a class to a table
so inserts and queries go through Python objects. This project already
depends on SQLAlchemy, which LangChain uses for some storage helpers.

The engine here is SQLite in memory, so running the file leaves no database file.

JavaScript comparison:
- SQLAlchemy maps a class to a table. Prisma and Drizzle do that job in the JavaScript world. Knex stays closer to SQL strings.
- Note.title == "Memory" builds a query. A Prisma call looks like prisma.note.findMany({ where: { title: "Memory" } }).
- session.add and session.commit are prisma.note.create and the client committing for you.
- sqlite:///:memory: is an in-memory database. better-sqlite3 uses the path ":memory:".
"""

from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# --------------------------------------------------
# 1. Model — one class, one table
# --------------------------------------------------

class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(80))
    body: Mapped[str] = mapped_column(String(300))

    def __repr__(self):
        return f"Note(id={self.id}, title={self.title!r})"


engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)

# --------------------------------------------------
# 2. Insert through a session
# --------------------------------------------------
# A session tracks objects. commit() writes them. rollback() would undo them.

with Session(engine) as session:
    session.add(Note(title="RAG", body="Retrieve, then generate."))
    session.add_all([
        Note(title="Tools", body="Call a function when the model asks."),
        Note(title="Memory", body="Send earlier messages back on the next turn."),
    ])
    session.commit()

    # --------------------------------------------------
    # 3. Query
    # --------------------------------------------------
    statement = select(Note).where(Note.title != "Tools").order_by(Note.id)
    notes = session.scalars(statement).all()
    print("Notes:", notes)

    # --------------------------------------------------
    # 4. Update an object, then commit
    # --------------------------------------------------
    memory = session.scalars(select(Note).where(Note.title == "Memory")).one()
    memory.body = "Trim history when the prompt is too long."
    session.commit()
    print("Updated:", memory.body)
