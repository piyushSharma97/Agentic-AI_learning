#!/usr/bin/env python
# coding: utf-8

# Python Lists

# 
# Lists allow us to add, remove and change items. A shopping cart can change.
# Items in a list are ordered. A shopping cart shows items in the order they are added.
# Lists allow duplicate values (might not be relevant for a shopping cart, though).
# it is an array with dynamic

# In[13]:


# Lists preserve item order and can be changed after creation, making them useful for collections that grow or shrink.
cart = ["T-shirt", "Lamp", "Pen"]
print(cart)

# A list of mixed data types
my_list = [1, "Python", 3.14]
print(my_list)

# Empty list
my_list = []
print(my_list)
print(type(my_list))


# In[5]:


# List indexes start at zero, so index 0 selects the first item.
languages = ["Python", "Swift", "C++"]

# Access the first item
print(f"languages[0] = {languages[0]}")

# Access the third item
print(f"languages[2] = {languages[2]}")


# Negative Indexing
# Python also supports negative indexing. The index of the last item is -1, the index of the second last item is -2 and so on.

# In[6]:


# Negative indexes count backward from the end; -1 is the last item.
languages = ["Python", "Swift", "C++"]

# Access the last item
print('languages[-1] =', languages[-1])

# Access the third last item
print('languages[-3] =', languages[-3]) 


# Adding and Updating Items

# In[ ]:


# Updating List Items
# Lists are mutable, so an item at a valid index can be replaced in place.
cart = ["T-shirt", "Lamp", "Pen"]

# Update second item to "Shoes"
cart[1] = "Shoes"

print(cart)    # ['T-shirt', 'Shoes', 'Pen']


# In[9]:


#Adding Items to a List
#We can use the append() method to add an item to the end of the list.
# append adds one item, extend adds each item from another iterable, and insert places an item at an index.
cart = ["T-shirt", "Lamp", "Pen"]

# Add "Book" to the list
cart.append("Book")

print(cart)    # ['T-shirt', 'Lamp', 'Pen', 'Book']


# In[10]:


#If we need to add all the items from another list (or any other iterable), we can use the extend() method.
# remove deletes a matching value, pop removes and returns an item, and clear empties the list.
cart = ["T-shirt", "Lamp", "Pen"]
fav_items = ["Headphones", "Phone"]

# Add all the items from fav_items to cart
cart.extend(fav_items)

print(cart)    # ['T-shirt', 'Lamp', 'Pen', 'Headphones', 'Phone']


# In[ ]:


#We can also insert an item at a specified position, using the insert() method.
cart = ["T-shirt", "Lamp", "Pen"]

# Add "Book" at index 2 (3rd position)
cart.insert(2, "Book")

print(cart)    # ['T-shirt', 'Lamp', 'Book', 'Pen']


# Remove Items From a List

# In[ ]:


#The remove() method removes the specified item from a list.
cart = ["T-shirt", "Lamp", "Pen", "Book"]

# Remove "Pen" from the list , it remove first occurance of an item
cart.remove("Pen")    # ['T-shirt', 'Lamp', 'Book']

#The pop() method removes and returns the last item (if an index is not provided). 
# This helps us implement lists as stacks (last in, first out data structure).
last_item = cart.pop()
print(cart)   # ['T-shirt', 'Lamp']
print(last_item)    # Book
#The clear() method empties a list.
# Clear the list
cart.clear()
print(cart)    # []


# In[ ]:


#we can also delete individual items or even the whole list itself using the del keyword.
cart = ['T-shirt', 'Lamp', 'Pen', 'Book']

# Delete the third item (index 2)
del cart[2]
print(cart)    # ['T-shirt', 'Lamp', 'Book']

# Delete the list itself
del cart
# print(cart)    # NameError: name 'cart' is not defined


# In[22]:


# Slices select a range; the ending index is excluded and omitted bounds use the list's beginning or end.
fruits=["apple","banana","mango","cherry","papaya","watermelon","orange"]
print(fruits[2:4])
print(fruits[-2:-4])
print(fruits[3:])


# In[23]:


# Assigning a slice replaces that section; because strings are iterable, this assigns individual characters.
fruits[2:]="kiwi"
print(fruits)


# In[30]:


#count
fruits.insert(1,"apple")
print(fruits)
print(fruits.count("banana"))
print(fruits.count("apple"))
print(fruits.reverse())


# In[31]:


# Assuming initial list:
fruits = ["banana", "orange"]

# 1. Insert: splice-like in JS (fruits.splice(1, 0, "apple"))
fruits.insert(1, "apple")
print(fruits)
# Output: ['banana', 'apple', 'orange']

# 2. Count: counts total occurrences (JS has no native .count(), requires filter)
print(fruits.count("banana"))  # Output: 1
print(fruits.count("apple"))   # Output: 1

# 3. Reverse: mutates in-place, returns None!
print(fruits.reverse())  # Output: None
print(fruits)            # Output: ['orange', 'apple', 'banana'] (reversed successfully!)


# In[ ]:


#slicing

# ============================================================
# REVISION NOTES — 1.6List.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates how to create, inspect, update, and remove items from Python lists.
# It also shows indexing, slicing, and common list methods using shopping-cart and fruit examples.
#
# KEY CONCEPTS USED:
# • Mutable sequence: a list keeps its order and can be changed after it is created.
# • Indexing and slicing: select one item or a range of items using zero-based positions.
# • List methods: append, extend, insert, remove, pop, clear, count, and reverse provide common operations.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is a foundational Python exercise; lists are used throughout the project's document, chunk, and result examples.
#
# THINGS TO REMEMBER:
# • The first list index is 0, and -1 refers to the last item.
# • reverse changes the list in place and returns None.
# • A slice's stop index is excluded, and assigning a string to a slice inserts its characters.
#
# COMMON MISTAKES TO AVOID:
# • Expecting reverse() to return a reversed list instead of modifying the existing one.
# • Confusing append (one item) with extend (items from an iterable).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the cart examples to compare changing one indexed item with adding, inserting, and removing items.
# ============================================================
