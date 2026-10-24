#!/usr/bin/env python
# coding: utf-8

# Data Type 
# 

# In[1]:


#Integer
# int stores whole numbers; type() lets us inspect the runtime type of a value.
age=30
print(type(age))


# In[ ]:


#Float
# float stores numbers with a fractional part.
length=5.6
print(type(length))


# In[3]:


#boolean
# bool represents one of two truth values: True or False.
is_bird=True
print(type(is_bird))


# In[5]:


a=4
b=4
type(bool(a==b))


# In[12]:


#common error
#res="hello"+5

# res will throw error

# Convert the integer to text before concatenating it with another string.
res2 = "hello"+str(5)

print(type(res2))
print(res2)

# ============================================================
# REVISION NOTES — 1.2-DATAtYPE.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file introduces common Python value types: integers, floating-point numbers, booleans, and strings.
# It also demonstrates checking types and converting an integer to text before concatenation.
#
# KEY CONCEPTS USED:
# • int and float: represent whole numbers and numbers with fractional parts.
# • bool: represents a truth value, either True or False.
# • Type conversion: changes a value's type, such as converting 5 to the string "5".
#
# HOW IT CONNECTS TO THE PROJECT:
# These basic types are the building blocks for variables, user input, model settings, and data passed to LangChain.
#
# THINGS TO REMEMBER:
# • Python distinguishes values by type, even when they look similar when printed.
# • int("5") converts valid numeric text to an integer.
# • Strings and integers cannot be concatenated directly without conversion.
#
# COMMON MISTAKES TO AVOID:
# • Assuming input text is automatically numeric.
# • Adding a string and integer without converting one of them first.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the commented-out string-plus-integer example with the working str(5) conversion below it.
# ============================================================
