#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
num = input('Enter a number: ')

print('You Entered:', num)

print('Data type of num:', type(num))


# In[5]:


integer_number = 123
float_number = 1.23

new_number = integer_number + float_number

# display new value and resulting data type
print("Value:",new_number)
print("Data Type:",type(new_number))


# In[6]:


num_string = '12'
num_integer = 23

print("Data type of num_string before Type Casting:",type(num_string))

# explicit type conversion
num_string = int(num_string)

print("Data type of num_string after Type Casting:",type(num_string))

num_sum = num_integer + num_string

print("Sum:",num_sum)
print("Data type of num_sum:",type(num_sum))

