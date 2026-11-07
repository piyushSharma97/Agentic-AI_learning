"""
=============================================================================
LANGGRAPH COMPONENT 8: CHAT MESSAGES AND CHAT MODELS
=============================================================================
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
    # get_llm() reads GROQ / Google / OpenAI from the project .env.
    # temperature=0 makes the reply stable, which is what you want while learning.
    llm = get_llm(temperature=0)

    messages = [
        SystemMessage(content="You are a friendly coding tutor. Answer in two sentences."),
        HumanMessage(content="I want to learn Python. Where should I start?", name="Krish"),
    ]

    print("Input thread:")
    for message in messages:
        message.pretty_print()

    # invoke() is one request. It returns an AIMessage, not a plain string.
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
