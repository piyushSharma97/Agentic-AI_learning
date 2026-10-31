#!/usr/bin/env python
# coding: utf-8

"""
Inheritance: a child class reuses a parent class and can change parts of it.

HumanMessage, AIMessage, and ToolMessage in LangChain follow this idea.
They are different message types that share the same base behaviour.

JavaScript comparison:
- class HumanMessage(Message) is class HumanMessage extends Message.
- super().__init__(content) is super(content) inside the constructor.
- isinstance(question, Message) is question instanceof Message.
- Overriding render in the child is the same idea as overriding a method in a JavaScript subclass.
"""

# --------------------------------------------------
# 1. Parent and child
# --------------------------------------------------
# Message stores the shared content.
# HumanMessage and AIMessage add a role and reuse Message.__init__ through super().

class Message:
    def __init__(self, content):
        self.content = content

    def render(self):
        return self.content


class HumanMessage(Message):
    role = "user"

    def render(self):
        return f"{self.role}: {self.content}"


class AIMessage(Message):
    role = "assistant"

    def render(self):
        return f"{self.role}: {self.content}"


turns = [
    HumanMessage("What is inheritance?"),
    AIMessage("A child class reuses and extends a parent class."),
]

for turn in turns:
    print(turn.render())

# --------------------------------------------------
# 2. isinstance checks the family, type checks the exact class
# --------------------------------------------------

question = turns[0]
print("Is a HumanMessage?", isinstance(question, HumanMessage))
print("Is a Message?", isinstance(question, Message))
print("Exact type is Message?", type(question) is Message)

# --------------------------------------------------
# 3. A child can add its own attributes
# --------------------------------------------------

class ToolMessage(Message):
    def __init__(self, content, tool_name):
        super().__init__(content)
        self.tool_name = tool_name

    def render(self):
        return f"tool:{self.tool_name}: {self.content}"


weather = ToolMessage("Sunny, 22C", tool_name="get_weather")
print(weather.render())
