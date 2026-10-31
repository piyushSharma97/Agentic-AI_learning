#!/usr/bin/env python
# coding: utf-8

"""
Pydantic models: describe the shape of data, then validate it.

A BaseModel class lists fields and types. When you construct it,
Pydantic checks the values and converts compatible ones (the string "12"
can become the integer 12). Invalid data raises ValidationError.

LangChain structured output and FastAPI request bodies both use this.
You already saw Product and TravelPlan in 7_Langchain Version V1.

JavaScript comparison:
- A Pydantic BaseModel is a zod schema in most TypeScript apps: const Product = z.object({ name: z.string().min(1), price: z.number().positive() }).
- Product(name="Notebook", price="499.0") both checks and converts. zod's parse does the same. "499.0" becomes a number only if you add z.coerce.number().
- model_dump() is the parsed object zod returns.
- ValidationError is the error zod throws from parse, or the result object from safeParse.
"""

from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------
# 1. A model is a class with typed fields
# --------------------------------------------------

class Product(BaseModel):
    name: str = Field(description="Product name", min_length=1)
    price: float = Field(description="Price in INR", gt=0)
    in_stock: bool = True


phone = Product(name="Notebook", price="499.0")
print(phone)
print("Price is a float now:", phone.price, type(phone.price))
print("Default in_stock:", phone.in_stock)

# --------------------------------------------------
# 2. model_dump turns the object back into a dict
# --------------------------------------------------
# JSON responses and logs want a dict or a JSON string, not the model object.

print("As dict:", phone.model_dump())
print("As JSON:", phone.model_dump_json())

# --------------------------------------------------
# 3. Invalid data is rejected
# --------------------------------------------------

try:
    Product(name="", price=-5)
except ValidationError as error:
    print("--- validation errors ---")
    for item in error.errors():
        print(item["loc"], item["msg"])

# --------------------------------------------------
# 4. model_validate reads a dict from the outside world
# --------------------------------------------------
# Use this when the dict came from json.loads or a form, not from your own code.

raw = {"name": "Pen", "price": 20}
pen = Product.model_validate(raw)
print("Validated:", pen.name, pen.price)
