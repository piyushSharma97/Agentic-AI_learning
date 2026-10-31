"""
Conditional Statements (if, elif, else)


Introduction to Conditional Statements
if Statement
else Statement
elif Statement
Nested Conditional Statements
Practical Examples
Common Errors and Best Practices
"""


## if statement
age=20

if age>=18:
    print("You are allowed to vote in the elections")

    
    
age>=18

## else
## The else statement executes a block of code if the condition in the if statement is False.

age=16

if age>=18:
    print("You are eligible for voting")
else:
    print("You are a minor")

num=int(input("Enter the number"))

if num>0:
    print("The number is positive")
    if num%2==0:
        print("The number is even")
    else:
        print("The number is odd")

else:
    print("The number is zero or negative")
