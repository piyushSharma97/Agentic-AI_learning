#!/usr/bin/env python
# coding: utf-8

# Python for Loop

# In[2]:


# A list of three AI models
models = ["Fable", "ChatGPT", "Gemini"]
for model in models:
    print("Modal name:=",model)
    print("---")


# In[5]:


# For Loop with Python range()
# Generate numbers from 1 to 7
values = range(1, 8)
for i in values:
    print("value=",i)


# In[7]:


#Iterating Through a String

string ="Piyush"
for s in string:
    print(s)


# In[9]:


#Break and continue Statements
#The break Statement
#The break statement terminates the for loop immediately when it's encountered. For example,

for num in range(1, 11):
    if num == 3:
        break
    print(num)


# In[10]:


#The continue Statement
#The continue statement skips the current iteration of the loop and continues with the next iteration. For example,

for num in range(1, 6):
    if num == 3:
        continue
    print(num)


# In[ ]:


#For Loop with else
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
for _ in range(0, 4):
    print("Hi")



#Nested for loops

attributes = ['Electric', 'Fast']
cars = ['Tesla', 'Porsche', 'Mercedes']

for att in attributes:
    for car in cars:
        print(att,car)
    print("---")
print('end')


# Python while Loop

# In[ ]:


number = float(input("Enter a number: "))

while number >= 0.0:
    print(number-1)
        # Take number input again
    number = float(input("Enter another number: "))


# In[20]:


# Print Numbers from 1 to n
n=10
i=1
while i<=n:
    print(n)
    n-=1

