#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Variables are names bound to values; assignment stores a value under a name.
## Declaring and assigning varible
age=30
height=5.4
name="Piyush"
is_employee=True;

print(is_employee)
print(name)
print(age)
print(height)


# In[ ]:


## nameing convention
## variable name should start with alphabate or '_'
## should be descriptive
# it can contain letter , number and underscore
# is case sensitive


# In[4]:


# using input() to take user input
# input() always returns text, so numeric input must be converted before arithmetic.
num = input('Enter a number: ')

print('You Entered:', num)

print('Data type of num:', type(num))


# In[5]:


integer_number = 123
float_number = 1.23

# Mixing int and float in arithmetic produces a float result.
new_number = integer_number + float_number

# display new value and resulting data type
print("Value:",new_number)
print("Data Type:",type(new_number))


# In[6]:


# Convert numeric text with int() before adding it to an integer.
num_string = '12'
num_integer = 23

print("Data type of num_string before Type Casting:",type(num_string))

# explicit type conversion
num_string = int(num_string)

print("Data type of num_string after Type Casting:",type(num_string))

num_sum = num_integer + num_string

print("Sum:",num_sum)
print("Data type of num_sum:",type(num_sum))

# ============================================================
# REVISION NOTES — 1.1-variable.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file introduces variable assignment, naming conventions, user input, and type conversion.
# The examples print values and types to show how Python stores and combines different kinds of data.
#
# KEY CONCEPTS USED:
# • Variables: names that refer to values.
# • input(): reads user input as a string.
# • Type conversion: converts compatible values, such as numeric text to an integer.
#
# HOW IT CONNECTS TO THE PROJECT:
# These Python fundamentals appear throughout the project wherever configuration, questions, or model results are stored.
#
# THINGS TO REMEMBER:
# • Python variable names are case-sensitive.
# • input() returns str even when the user types digits.
# • Combining an int and float produces a float.
#
# COMMON MISTAKES TO AVOID:
# • Attempting arithmetic on raw input text.
# • Using a variable name that begins with a digit or contains spaces.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Trace num_string through int() and compare its type before and after the conversion.
# ============================================================
