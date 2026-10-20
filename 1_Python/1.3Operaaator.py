#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#arthmatic
# add(a+b) minus(a-b) divison (a/b) multi (a*b)
#floor_divison =a//b
#modulas = a%b
#exponantion = a**b


# In[ ]:


x = 15
y = 4
print('x + y = ',x+y)
print('x - y = ',x-y)
print('x * y = ',x*y)
print('x / y = ',x/y)
print('x // y = ',x//y)
print('x ** y = ',x**y)


# Comparison operators

# In[ ]:


x = 10
y = 12
print('x > y  is',x>y)
print('x < y  is',x<y)
print('x == y is',x==y)
print('x != y is',x!=y)
print('x >= y is',x>=y)
print('x <= y is',x<=y)


# Logical operators
# and	True  if both the operands are true	x and y
# or	True  if either of the operands is true	x or y
# not	True  if operand is false (complements the operand)	not x

# In[2]:


x = True
y = False
print('x and y is',x and y)
print('x or y is',x or y)
print('not x is',not x)


# Bitwise operators
# &	Bitwise AND	x& y = 0 (0000 0000)
# |	Bitwise OR	x | y = 14 (0000 1110)
# ~	Bitwise NOT	~x = -11 (1111 0101)
# ^	Bitwise XOR	x ^ y = 14 (0000 1110)
# >>	Bitwise right shift	x>> 2 = 2 (0000 0010)
# <<	Bitwise left shift	x<< 2 = 42 (0010 1000)

# Assignment operators
# 
# =	Simple Assignment	x = 5	x = 5
# +=	Addition Assignment	x += 5	x = x + 5
# -=	Subtraction Assignment	x -= 5	x = x - 5
# *=	Multiplication Assignment	x *= 5	x = x * 5
# /=	Division Assignment	x /= 5	x = x / 5 (returns float)
# %=	Modulus Assignment	x %= 5	x = x % 5 (remainder)
# //=	Floor Division Assignment	x //= 5	x = x // 5 (integer division)
# **=	Exponentiation Assignment	x **= 5	x = x ** 5 (power)
# &=	Bitwise AND Assignment	x &= 5	x = x & 5
# |=	Bitwise OR Assignment	x |= 5	x = x | 5
# ^=	Bitwise XOR Assignment	x ^= 5	x = x ^ 5
# >>=	Bitwise Right Shift Assignment	x >>= 5	x = x >> 5
# <<=	Bitwise Left Shift Assignment	x <<= 5	x = x << 5

# In[3]:


# 1. Simple Assignment (=)
val_assign = 10
print(f"val_assign: {val_assign}")  # Output: 10

# 2. Addition Assignment (+=)
val_add = 20
val_add += 5  # 20 + 5
print(f"val_add: {val_add}")  # Output: 25

# 3. Subtraction Assignment (-=)
val_sub = 30
val_sub -= 7  # 30 - 7
print(f"val_sub: {val_sub}")  # Output: 23

# 4. Multiplication Assignment (*=)
val_mul = 6
val_mul *= 4  # 6 * 4
print(f"val_mul: {val_mul}")  # Output: 24

# 5. Division Assignment (/=) -> Always returns a float
val_div = 25
val_div /= 2  # 25 / 2
print(f"val_div: {val_div}")  # Output: 12.5

# 6. Modulus Assignment (%=) -> Remainder
val_mod = 17
val_mod %= 5  # 17 % 5
print(f"val_mod: {val_mod}")  # Output: 2

# 7. Floor Division Assignment (//=) -> Drops decimal
val_floor = 17
val_floor //= 5  # 17 // 5
print(f"val_floor: {val_floor}")  # Output: 3

# 8. Exponentiation Assignment (**=) -> Power
val_exp = 2
val_exp **= 4  # 2 ** 4 (2^4)
print(f"val_exp: {val_exp}")  # Output: 16

# 9. Bitwise AND Assignment (&=)
# 12 (binary 1100) & 5 (binary 0101) = 4 (binary 0100)
val_band = 12
val_band &= 5
print(f"val_band: {val_band}")  # Output: 4

# 10. Bitwise OR Assignment (|=)
# 12 (binary 1100) | 5 (binary 0101) = 13 (binary 1101)
val_bor = 12
val_bor |= 5
print(f"val_bor: {val_bor}")  # Output: 13

# 11. Bitwise XOR Assignment (^=)
# 12 (binary 1100) ^ 5 (binary 0101) = 9 (binary 1001)
val_bxor = 12
val_bxor ^= 5
print(f"val_bxor: {val_bxor}")  # Output: 9

# 12. Bitwise Right Shift Assignment (>>=)
# 16 (binary 10000) >> 2 = 4 (binary 00100)
val_rshift = 16
val_rshift >>= 2
print(f"val_rshift: {val_rshift}")  # Output: 4

# 13. Bitwise Left Shift Assignment (<<=)
# 3 (binary 00011) << 2 = 12 (binary 01100)
val_lshift = 3
val_lshift <<= 2
print(f"val_lshift: {val_lshift}")  # Output: 12

