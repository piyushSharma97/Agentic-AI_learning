
# Tuples are ordered collections of items that are immutable. They are similar to lists, but their immutability makes them different.
## creating a tuple
empty_tuple=()
print(empty_tuple)
print(type(empty_tuple))


lst=list()
print(type(lst))
tpl=tuple()
print(type(tpl))


mixed_tuple=(1,"Hello World",3.14, True)
print(mixed_tuple)


## Tuple Operations

concatenation_tuple=numbers + mixed_tuple
print(concatenation_tuple)