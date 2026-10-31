#!/usr/bin/env python
# coding: utf-8

"""
List practice: to-do items, grades, inventory, and feedback.

These examples use the list operations from 1.6List.py on everyday data.
append adds one item at the end. remove deletes the first matching value.
in checks whether a value is present. A for loop visits every item.

JavaScript comparison:
- A list is an array.
- append(x) is push(x). remove(x) is close to splice(items.indexOf(x), 1).
- "Pay bills" in to_do_list is toDoList.includes("Pay bills").
- len(grades) is grades.length. sum(grades) has no array method in JS.
  JS uses grades.reduce((total, grade) => total + grade, 0).
- max(grades) and min(grades) are Math.max(...grades) and Math.min(...grades).
- for task in to_do_list is for (const task of toDoList).
"""

# --------------------------------------------------
# 1. A to-do list grows and shrinks
# --------------------------------------------------

to_do_list = ["Buy Groceries", "Clean the house", "Pay bills"]

# append puts the new string at the end. JS: toDoList.push("Schedule meeting")
to_do_list.append("Schedule meeting")
to_do_list.append("Go For a Run")

# remove deletes by value, not by index. The first "Clean the house" is gone.
to_do_list.remove("Clean the house")

# in asks "is this value somewhere in the list?"
if "Pay bills" in to_do_list:
    print("Don't forgrt to pay the utility bills")

print("To Do List remaining")
for task in to_do_list:
    print(f"-{task}")

# --------------------------------------------------
# 2. Grades: add one score, then summarize the list
# --------------------------------------------------

grades = [85, 92, 78, 90, 88]
grades.append(95)

# sum adds every number. len counts them. Dividing those two is the average.
average_grade = sum(grades) / len(grades)
print(f"Average Grade: {average_grade:.2f}")

# max and min walk the list for you. You do not write the loop.
highest_grade = max(grades)
lowest_grade = min(grades)
print(f"Highest Grade: {highest_grade}")
print(f"Lowest Grade: {lowest_grade}")

# --------------------------------------------------
# 3. Inventory: membership decides the message
# --------------------------------------------------

inventory = ["apples", "bananas", "oranges", "grapes"]
inventory.append("strawberries")
inventory.remove("bananas")

item = "oranges"
if item in inventory:
    print(f"{item} are in stock.")
else:
    print(f"{item} are out of stock.")

print("Inventory List:")
for item in inventory:
    print(f"- {item}")

# --------------------------------------------------
# 4. Feedback: count comments that contain a word
# --------------------------------------------------
# The generator expression walks each comment, lowercases it, and adds 1
# when the comment looks positive. sum totals those 1s.
# JS: feedback.filter(comment => comment.toLowerCase().includes("great")).length

feedback = ["Great service!", "Very satisfied", "Could be better", "Excellent experience"]
feedback.append("Not happy with the service")

positive_feedback_count = sum(
    1
    for comment in feedback
    if "great" in comment.lower() or "excellent" in comment.lower()
)
print(f"Positive Feedback Count: {positive_feedback_count}")

print("User Feedback:")
for comment in feedback:
    print(f"- {comment}")
