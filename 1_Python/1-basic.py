#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Python uses indentation, capitalization, and line structure as part of its syntax.
## basic syntax rules of python

## python is case sensitive 
# Assigning the same name again replaces its previous binding.
name='Piyush'
name='Sharma'

print(name)


# In[ ]:


## indentation
## python use indentation to define block of code .consistnace use of space(usually 4 ) or tab is required
# Indentation marks the body of a block, such as the statements controlled by if.
age=32
if age>30:
    print(age)

print(age)


# In[ ]:


## line  continuation 
## use blackshash (\)
# A backslash continues this expression onto the next physical line.
total=1+2+3+4+5+\
    6
print(total)


# In[13]:


## multiple statement in one line
# Semicolons can separate statements, though separate lines are usually easier to read.
x=5;y=6;z=x*y
print(z)
print(total)


# In[14]:


#exmaple of indentation
# Nested blocks need consistent indentation to show which statements belong together.
if True:
    print("this is true")
    if False:
        print("this is false")
    print("this will also peint")
print('asdas')       

# ============================================================
# REVISION NOTES — 1-basic.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates introductory Python syntax, including assignment, indentation, line continuation, and multiple statements.
# It also shows how nested blocks determine which statements run conditionally.
#
# KEY CONCEPTS USED:
# • Case sensitivity: names with different capitalization are treated as different names.
# • Indentation: leading spaces define the body and nesting level of a block.
# • Assignment: rebinding a variable name stores a new value for later use.
#
# HOW IT CONNECTS TO THE PROJECT:
# Correct Python syntax and indentation are prerequisites for every Python example in this project.
#
# THINGS TO REMEMBER:
# • Use consistent indentation, commonly four spaces.
# • A later assignment to a name replaces its earlier value.
# • Prefer one statement per line for readability.
#
# COMMON MISTAKES TO AVOID:
# • Mixing tabs and spaces in the same indentation block.
# • Assuming a statement indented inside a block runs after that block ends.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the indentation around the nested if statements to see which print calls belong to each block.
# ============================================================
