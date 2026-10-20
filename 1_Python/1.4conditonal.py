#!/usr/bin/env python
# coding: utf-8

# Python if...else Statement 

# if condition:
#     # This code runs only if condition is True
# 

# In[2]:


age = int(input("Enter your age: "))

# Check if age is 18 or more
if age >= 18:
    print("Grant access to the website.")

print("Program complete.")


# In[3]:


age = int(input("Enter age"))

if age>=18:
    print("Major adult")
    print("can vote")
else:
    print('Cant vote')    


# In[4]:


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


age = int(input("Enter age"))

if age<0:
    print("Invalid age")
elif age>=18:
    print("Access granted")
else:
    print("Access Denied")

