#!/usr/bin/env python
# coding: utf-8

# In[2]:


## basic syntax rules of python

## python is case sensitive 
name='Piyush'
name='Sharma'

print(name)


# In[ ]:


## indentation
## python use indentation to define block of code .consistnace use of space(usually 4 ) or tab is required
age=32
if age>30:
    print(age)

print(age)


# In[ ]:


## line  continuation 
## use blackshash (\)
total=1+2+3+4+5+\
    6
print(total)


# In[13]:


## multiple statement in one line
x=5;y=6;z=x*y
print(z)
print(total)


# In[14]:


#exmaple of indentation
if True:
    print("this is true")
    if False:
        print("this is false")
    print("this will also peint")
print('asdas')       

