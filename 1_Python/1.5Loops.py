#!/usr/bin/env python
# coding: utf-8

# Python for Loop

# In[2]:


# A list of three AI models
# A for loop visits each item in an iterable once, assigning it to the loop variable.
models = ["Fable", "ChatGPT", "Gemini"]
for model in models:
    print("Modal name:=",model)
    print("---")


# In[5]:


# For Loop with Python range()
# Generate numbers from 1 to 7
# range's stop value is excluded, so range(1, 8) generates 1 through 7.
values = range(1, 8)
for i in values:
    print("value=",i)


# In[7]:


#Iterating Through a String

# Strings are iterable too, so this loop processes one character at a time.
string ="Piyush"
for s in string:
    print(s)


# In[9]:


#Break and continue Statements
#The break Statement
#The break statement terminates the for loop immediately when it's encountered. For example,

# break exits the loop immediately when its condition becomes true.
for num in range(1, 11):
    if num == 3:
        break
    print(num)


# In[10]:


#The continue Statement
#The continue statement skips the current iteration of the loop and continues with the next iteration. For example,

# continue skips only the current iteration and proceeds with the next number.
for num in range(1, 6):
    if num == 3:
        continue
    print(num)


# In[ ]:


#For Loop with else
# A for-else loop runs its else block only if the loop finishes without break.
stock = ['Laptop', 'Keyboard', 'Mouse']

order = input("Enter the product you want to buy: ")

for product in stock:
    if product == order:
        print(f"{order} is available. Adding to cart.")
        break
else:
    print(f"Sorry, {order} is out of stock.")


# In[13]:


#Using for Loop Without Using Items#

# Iterate from i = 0 to 3
# An underscore names a loop value that the code intentionally does not use.
for _ in range(0, 4):
    print("Hi")



#Nested for loops

# Nested loops combine every attribute with every car.
attributes = ['Electric', 'Fast']
cars = ['Tesla', 'Porsche', 'Mercedes']

for att in attributes:
    for car in cars:
        print(att,car)
    print("---")
print('end')


# Python while Loop

# In[ ]:


# A while loop repeats as long as its condition remains true; input changes the value each round.
number = float(input("Enter a number: "))

while number >= 0.0:
    print(number-1)
        # Take number input again
    number = float(input("Enter another number: "))


# In[20]:


# Print Numbers from 1 to n
# This loop counts n down to 1 by decreasing it each iteration.
n=10
i=1
while i<=n:
    print(n)
    n-=1

# ============================================================
# REVISION NOTES — 1.5Loops.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates for and while loops, including iteration over lists, ranges, and strings.
# It also shows nested loops, loop control statements, and the for-else pattern.
#
# KEY CONCEPTS USED:
# • Iteration: a loop repeats code for items in a collection or sequence of values.
# • break and continue: break exits a loop, while continue skips the current iteration.
# • Loop conditions: while loops continue as long as their condition is true.
#
# HOW IT CONNECTS TO THE PROJECT:
# Loops are fundamental Python tools used to process documents, chunks, search results, and conversation messages.
#
# THINGS TO REMEMBER:
# • range excludes its stop value.
# • A loop's else clause executes only if the loop did not exit with break.
# • A while loop needs progress toward its stopping condition to avoid running forever.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting to update a value used by a while condition.
# • Using break when the intent is only to skip one iteration; use continue for that.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the break and continue examples to see the difference between exiting a loop and skipping one turn.
# ============================================================
