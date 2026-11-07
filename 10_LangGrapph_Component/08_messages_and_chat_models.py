"""
=============================================================================
LANGGRAPH COMPONENT 8: CHAT MESSAGES AND CHAT MODELS
=============================================================================
USE
Build the conversation object you will later store in graph state.
A chat model call takes that list and returns one new AIMessage.

WHY THIS APPROACH
One big string cannot say who is the tutor and who is the student.
Separate message classes carry the role, so the provider knows which
lines are instructions, which are the user, and which are earlier replies.
A list is used because order is the history: the model reads it top to bottom.
This file stops before a graph. You need to see a message list on its own
before file 09 stores that list in state.

This is the first half of the "Chain Using LangGraph" notebook.

A chat model does not want one big string. It wants a LIST of messages, and
each message has a role:

- SystemMessage  -> instructions for the model ("You are a Python tutor")
- HumanMessage   -> something the user said
- AIMessage      -> something the model said
- ToolMessage    -> the result of a tool (covered in 09_chain_with_tools.py)

Every message has:
- content            the text
- name               optional author label
- response_metadata  extra data the provider sends back (model name, tokens)

The notebook called result.response_metadata and stopped. In a .py file that
expression does nothing visible, so the examples below PRINT the useful parts.

HOW TO STUDY THIS FILE
main() runs one example. Comment that line and uncomment the next example
when you are ready.
=============================================================================
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm, safe_run


def example_message_types():
    """Build a short conversation and print each role."""
    print("A conversation is just a list. Order matters: earlier messages are the history.\n")

    # Use: a list, oldest message first.
    # Why: the model treats earlier items as history. SystemMessage is first
    # so the instruction applies to every reply, not as something the user said.
    messages = [
        SystemMessage(
            content="You are a friendly coding tutor. Keep answers short.",
            name="System",
        ),
        AIMessage(content="How can I help you?", name="Tutor"),
        HumanMessage(content="I want to learn coding.", name="Krish"),
        AIMessage(content="Which language do you want to learn?", name="Tutor"),
        HumanMessage(content="I want to learn Python.", name="Krish"),
    ]

    for message in messages:
        # pretty_print() shows the role and the text. Much clearer than print(message).
        message.pretty_print()

    print(f"\nThis thread has {len(messages)} messages.")
    print("Next example: send a thread like this into a real chat model.")


def example_invoke_chat_model():
    """Send a message list to the chat model and read the reply."""
    # Use: get_llm() so this lesson shares the model in config.py.
    # Why: temperature=0 asks for the most likely next words. A higher
    # temperature is for variety. While learning, a stable reply is easier to compare.
    llm = get_llm(temperature=0)

    # Use: system instruction, then the user's question.
    # Why: two roles in one list. The system line is not shown as the user's
    # words, and the human line is the only thing the reply should answer.
    messages = [
        SystemMessage(content="You are a friendly coding tutor. Answer in two sentences."),
        HumanMessage(content="I want to learn Python. Where should I start?", name="Krish"),
    ]

    print("Input thread:")
    for message in messages:
        message.pretty_print()

    # Use: invoke(list_of_messages) for one complete reply.
    # Why: the return is an AIMessage, not a str. You need the object later
    # so a graph can store it, read tool_calls, and print the role.
    result = llm.invoke(messages)

    print("\nModel reply (AIMessage):")
    result.pretty_print()

    # The notebook stopped at `result.response_metadata`, which only displays
    # inside Jupyter. Here we print the fields that tell you what happened.
    metadata = result.response_metadata or {}
    print("\nresponse_metadata (what the provider sent back):")
    print(f"  model:        {metadata.get('model_name') or metadata.get('model')}")
    print(f"  token usage:  {metadata.get('token_usage') or metadata.get('usage')}")


def main():
    # Uncomment ONE example at a time.
    safe_run("Chat message types", example_message_types)
    # safe_run("Invoke a chat model", example_invoke_chat_model)


if __name__ == "__main__":
    main()
