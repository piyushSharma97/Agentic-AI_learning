#!/usr/bin/env python
# coding: utf-8

# Python if...else Statement 

# if condition:
#     # This code runs only if condition is True
# 

# In[2]:


# Convert the user's age from text to an integer before comparing it with 18.
age = int(input("Enter your age: "))

# Check if age is 18 or more
if age >= 18:
    print("Grant access to the website.")

print("Program complete.")


# In[3]:


# if and else choose one branch according to whether the condition is true.
age = int(input("Enter age"))

if age>=18:
    print("Major adult")
    print("can vote")
else:
    print('Cant vote')    


# In[4]:


# This example compares input values with sample credentials; it is for conditionals, not real authentication.
# Username and password stored in database
username_db = "admin"
password_db = "sparrow@123"

# Username and password entered by the user
username = input("Enter username: ")
password = input("Enter password: ") 

# Check if username & password in database matches user's input
if (username == username_db) and (password == password_db):
    print("Welcome back.")
else:
    print("Access denied.")


# In[6]:


# elif lets the program check another condition when the earlier if condition was false.
age = int(input("Enter age"))

if age<0:
    print("Invalid age")
elif age>=18:
    print("Access granted")
else:
    print("Access Denied")

# ============================================================
# REVISION NOTES — 1.4conditonal.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates if, elif, and else statements for choosing actions based on conditions.
# Its examples check age thresholds and compare entered text with sample credentials.
#
# KEY CONCEPTS USED:
# • Conditional: code that runs only when a Boolean condition is true.
# • Branching: if, elif, and else select among different code paths.
# • Boolean logic: and combines conditions so both must be true.
#
# HOW IT CONNECTS TO THE PROJECT:
# Conditionals are core Python foundations for validating input and handling different outcomes in later examples.
#
# THINGS TO REMEMBER:
# • Use == to compare values and = to assign values.
# • elif and else must follow an if branch.
# • input() returns text; convert it before numeric comparisons.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting to convert age input from str to int.
# • Treating hard-coded sample credentials as secure authentication.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Read the age if/elif/else example from top to bottom and check which condition matches each input.
# ============================================================
