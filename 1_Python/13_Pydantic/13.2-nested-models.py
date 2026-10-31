#!/usr/bin/env python
# coding: utf-8

"""
Nested models and field validators.

A field can be another model, or a list of models.
A validator is a method that cleans or rejects one field after the type check.

This is the same pattern as a resume extractor: one Candidate model
holds a list of Skill models instead of one flat string.

JavaScript comparison:
- A nested model is z.object({ skills: z.array(Skill) }).
- @field_validator("email") is .refine() or a transform on that field.
- model_validate(dict) is schema.parse(data).
- list[Skill] is Skill[]. Python needs the class. zod needs z.array(skillSchema).
"""

from pydantic import BaseModel, Field, field_validator

class Skill(BaseModel):
    name: str
    years: int = Field(ge=0)


class Candidate(BaseModel):
    name: str
    email: str
    skills: list[Skill]

    @field_validator("email")
    @classmethod
    def email_must_have_at(cls, value):
        # The type check already proved this is a string.
        # The validator adds a rule Pydantic does not have built in.
        if "@" not in value:
            raise ValueError("email must contain @")
        return value.strip().lower()

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("name cannot be blank")
        return cleaned


profile = Candidate.model_validate({
    "name": "  Asha   Sharma ",
    "email": "Asha@Example.com",
    "skills": [
        {"name": "Python", "years": 3},
        {"name": "SQL", "years": 2},
    ],
})

print("Name:", profile.name)
print("Email:", profile.email)
print("First skill:", profile.skills[0].name, profile.skills[0].years)
print(profile.model_dump())
