# ============================================================
# LANGCHAIN V1: AGENT MIDDLEWARE
# ============================================================
#
# Middleware is code that runs inside the agent loop, around
# model calls and tool calls. It lets you change what the agent
# does without rewriting the agent itself.
#
# Common uses:
# - Logging, analytics, and debugging
# - Rewriting prompts, choosing tools, and formatting output
# - Retries, fallbacks, and stopping early
# - Rate limits, guardrails, and PII checks
#
# Examples covered:
#
# 1. Summarize history after a message count
# 2. Summarize history after a token count
# 3. Summarize history after a fraction of the context window
# 4. Human-in-the-loop: approve a tool call
# 5. Human-in-the-loop: reject a tool call
# 6. Human-in-the-loop: edit a tool call, then let it run
#
# Each example builds its own agent, checkpointer, and thread id,
# so state from one demo never leaks into the next.
#
# Uncomment the example you want inside main().
# ============================================================


# ============================================================
# 1. IMPORTS AND ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Load API keys from the project .env file before any model call.
load_dotenv()


def print_section(title, explanation):
    """Print a banner so each demo is easy to spot in the terminal."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(explanation)
    print("-" * 72)


def estimate_token_count(messages):
    """
    Rough token estimate: about 4 characters per token.

    This is only for the teaching prints. SummarizationMiddleware
    uses the model's own tokenizer, which will not match this number
    exactly.
    """
    total_characters = sum(len(str(message.content)) for message in messages)
    return total_characters // 4


# gpt-4o-mini's context window. The fraction demo divides by this
# only so the printed percentage is easy to read.
GPT4O_MINI_CONTEXT_TOKENS = 128_000


# ============================================================
# EXAMPLE 1: SUMMARIZE BY MESSAGE COUNT
# ============================================================
#
# trigger=("messages", 10)
#     Start summarizing once this thread holds 10 messages.
# keep=("messages", 4)
#     After a summary is written, leave the latest 4 messages
#     untouched and compress everything older into one summary.
#
# A checkpointer plus a stable thread_id is required. Without them
# every invoke() starts from an empty history, so the counter never
# reaches the trigger.


def example_1_summarize_by_message_count():
    print_section(
        "EXAMPLE 1: SummarizationMiddleware — trigger by message count",
        "Same thread is reused. After 10 stored messages, older turns are\n"
        "compressed and only the latest 4 messages are kept in full.",
    )

    message_count_agent = create_agent(
        model="gpt-4o-mini",
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("messages", 10),
                keep=("messages", 4),
            )
        ],
    )

    # One thread id means every question below is appended to the same chat.
    message_count_thread_config = {"configurable": {"thread_id": "summary-by-message-count"}}

    message_count_questions = [
        "What is 2+2?",
        "What is 10*5?",
        "What is 100/4?",
        "What is 15-7?",
        "What is 3*3?",
        "What is 4*4?",
    ]

    for question_number, math_question in enumerate(message_count_questions, start=1):
        message_count_response = message_count_agent.invoke(
            {"messages": [HumanMessage(content=math_question)]},
            config=message_count_thread_config,
        )
        stored_message_count = len(message_count_response["messages"])

        print(f"[Example 1 | turn {question_number}] User asked: {math_question}")
        print(f"[Example 1 | turn {question_number}] Messages now stored in this thread: {stored_message_count}")
        print(f"[Example 1 | turn {question_number}] Latest reply: {message_count_response['messages'][-1].content}")
        print(f"[Example 1 | turn {question_number}] Full message state: {message_count_response['messages']}")
        print("-" * 72)


# ============================================================
# EXAMPLE 2: SUMMARIZE BY TOKEN COUNT
# ============================================================
#
# trigger=("tokens", 550)
#     Summarize once the stored history is about 550 tokens.
# keep=("tokens", 200)
#     Keep roughly the newest 200 tokens verbatim.
#
# The hotel tool returns a long string on purpose so the history
# grows fast enough to cross 550 tokens within a few cities.


@tool
def search_hotels_for_token_trigger(city: str) -> str:
    """Search hotels and return a long description so the token demo fills up faster."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


def example_2_summarize_by_token_count():
    print_section(
        "EXAMPLE 2: SummarizationMiddleware — trigger by token count",
        "Each city search adds a long tool result. When the thread reaches\n"
        "about 550 tokens, older context is summarized and about 200 tokens stay.",
    )

    token_trigger_agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_hotels_for_token_trigger],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("tokens", 550),
                keep=("tokens", 200),
            ),
        ],
    )

    token_trigger_thread_config = {"configurable": {"thread_id": "summary-by-token-count"}}
    token_trigger_cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

    for token_demo_city in token_trigger_cities:
        token_trigger_response = token_trigger_agent.invoke(
            {"messages": [HumanMessage(content=f"Find hotels in {token_demo_city}")]},
            config=token_trigger_thread_config,
        )
        token_trigger_messages = token_trigger_response["messages"]
        estimated_tokens_in_token_demo = estimate_token_count(token_trigger_messages)

        print(f"[Example 2 | {token_demo_city}] Approximate tokens still stored: {estimated_tokens_in_token_demo}")
        print(f"[Example 2 | {token_demo_city}] Messages still stored: {len(token_trigger_messages)}")
        print(f"[Example 2 | {token_demo_city}] Latest reply: {token_trigger_messages[-1].content}")
        print(f"[Example 2 | {token_demo_city}] Full message state: {token_trigger_messages}")
        print("-" * 72)


# ============================================================
# EXAMPLE 3: SUMMARIZE BY FRACTION OF THE CONTEXT WINDOW
# ============================================================
#
# trigger and keep can also be a fraction of the model's max context.
# 0.005 means 0.5% of the window (about 640 tokens of 128,000).
# 0.002 means 0.2% of the window (about 256 tokens).
#
# These fractions are intentionally tiny so a short demo can fire the
# middleware. In a real app you would use a much larger fraction,
# closer to when the window is actually getting full.


@tool
def search_hotels_for_fraction_trigger(city: str) -> str:
    """Search hotels for the fraction-based summarization demo."""
    return f"Hotels in {city}: Grand Hotel $350, City Inn $180, Budget Stay $75"


def example_3_summarize_by_fraction():
    print_section(
        "EXAMPLE 3: SummarizationMiddleware — trigger by context-window fraction",
        "Summarize at 0.5% of the context window and keep about 0.2%.\n"
        "The printed percentage uses a 128,000-token window for gpt-4o-mini.",
    )

    fraction_trigger_agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_hotels_for_fraction_trigger],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("fraction", 0.005),
                keep=("fraction", 0.002),
            ),
        ],
    )

    fraction_trigger_thread_config = {"configurable": {"thread_id": "summary-by-fraction"}}
    fraction_trigger_cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

    for fraction_demo_city in fraction_trigger_cities:
        fraction_trigger_response = fraction_trigger_agent.invoke(
            {"messages": [HumanMessage(content=f"Hotels in {fraction_demo_city}")]},
            config=fraction_trigger_thread_config,
        )
        fraction_trigger_messages = fraction_trigger_response["messages"]
        estimated_tokens_in_fraction_demo = estimate_token_count(fraction_trigger_messages)
        fraction_of_context_window = estimated_tokens_in_fraction_demo / GPT4O_MINI_CONTEXT_TOKENS

        print(
            f"[Example 3 | {fraction_demo_city}] "
            f"Approximate tokens: {estimated_tokens_in_fraction_demo} "
            f"({fraction_of_context_window:.4%} of the context window)"
        )
        print(f"[Example 3 | {fraction_demo_city}] Messages still stored: {len(fraction_trigger_messages)}")
        print(f"[Example 3 | {fraction_demo_city}] Latest reply: {fraction_trigger_messages[-1].content}")
        print(f"[Example 3 | {fraction_demo_city}] Full message state: {fraction_trigger_messages}")
        print("-" * 72)


# ============================================================
# HUMAN-IN-THE-LOOP (shared idea for examples 4, 5, and 6)
# ============================================================
#
# HumanInTheLoopMiddleware pauses the agent BEFORE a chosen tool runs.
# The graph stays frozen until you resume it with a Command.
#
# interrupt_on maps a tool name to a policy:
# - a dict of allowed_decisions: pause and accept only those decisions
# - False: do not pause; let that tool run on its own
#
# A decision is one of:
# - approve: run the tool with the arguments the model chose
# - reject:  do not run the tool; tell the model the call was refused
# - edit:    run the tool, but with arguments you replace
#
# The checkpointer is what makes the pause possible. The same thread_id
# must be passed on the resume call, or the graph cannot find the pause.


# ============================================================
# EXAMPLE 4: APPROVE A TOOL CALL
# ============================================================


@tool
def read_email_for_approve_demo(email_id: str) -> str:
    """Mock read of an email. This demo does not pause on this tool."""
    return f"Email content for ID: {email_id}"


@tool
def send_email_for_approve_demo(recipient: str, subject: str, body: str) -> str:
    """Mock send of an email. This demo pauses before the email is sent."""
    return f"Email sent to {recipient} with subject '{subject}'"


def example_4_approve_tool_call():
    print_section(
        "EXAMPLE 4: HumanInTheLoopMiddleware — approve",
        "send_email_for_approve_demo pauses for a human decision.\n"
        "read_email_for_approve_demo is set to False, so reading would not pause.\n"
        "This run approves the email and lets the original arguments through.",
    )

    approve_demo_agent = create_agent(
        model="gpt-4o",
        tools=[read_email_for_approve_demo, send_email_for_approve_demo],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email_for_approve_demo": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    # False means this tool is allowed to run with no human pause.
                    "read_email_for_approve_demo": False,
                }
            )
        ],
    )

    approve_demo_thread_config = {"configurable": {"thread_id": "hitl-approve"}}

    approve_demo_paused_result = approve_demo_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
                )
            ]
        },
        config=approve_demo_thread_config,
    )

    print("[Example 4] State returned on the first invoke (this should contain the pause):")
    print(approve_demo_paused_result)

    # "__interrupt__" is present only when middleware stopped the graph
    # before the tool executed. Resume with the same thread id.
    if "__interrupt__" in approve_demo_paused_result:
        print("[Example 4] Paused before send_email_for_approve_demo. Decision: approve.")
        approve_demo_final_result = approve_demo_agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=approve_demo_thread_config,
        )
        print(f"[Example 4] Reply after approval: {approve_demo_final_result['messages'][-1].content}")
    else:
        print("[Example 4] No pause was returned, so there was nothing to approve.")


# ============================================================
# EXAMPLE 5: REJECT A TOOL CALL
# ============================================================


@tool
def read_email_for_reject_demo(email_id: str) -> str:
    """Mock read of an email for the reject demo."""
    return f"Email content for ID: {email_id}"


@tool
def send_email_for_reject_demo(recipient: str, subject: str, body: str) -> str:
    """Mock send of an email. This demo rejects the call before it runs."""
    return f"Email sent to {recipient} with subject '{subject}'"


def example_5_reject_tool_call():
    print_section(
        "EXAMPLE 5: HumanInTheLoopMiddleware — reject",
        "The model again tries to send an email.\n"
        "The human decision is reject, so the tool does not run.",
    )

    reject_demo_agent = create_agent(
        model="gpt-4o",
        tools=[read_email_for_reject_demo, send_email_for_reject_demo],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email_for_reject_demo": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    "read_email_for_reject_demo": False,
                }
            ),
        ],
    )

    reject_demo_thread_config = {"configurable": {"thread_id": "hitl-reject"}}

    reject_demo_paused_result = reject_demo_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
                )
            ]
        },
        config=reject_demo_thread_config,
    )

    print("[Example 5] State returned on the first invoke (this should contain the pause):")
    print(reject_demo_paused_result)

    if "__interrupt__" in reject_demo_paused_result:
        print("[Example 5] Paused before send_email_for_reject_demo. Decision: reject.")
        reject_demo_final_result = reject_demo_agent.invoke(
            Command(resume={"decisions": [{"type": "reject"}]}),
            config=reject_demo_thread_config,
        )
        print(f"[Example 5] Reply after rejection: {reject_demo_final_result['messages'][-1].content}")
    else:
        print("[Example 5] No pause was returned, so there was nothing to reject.")


# ============================================================
# EXAMPLE 6: EDIT A TOOL CALL, THEN RUN IT
# ============================================================
#
# type "edit" replaces the tool name and arguments before execution.
# The model asked for wrong@email.com. The human changes the recipient,
# subject, and body, and the tool then runs with those new arguments.


@tool
def read_email_for_edit_demo(email_id: str) -> str:
    """Mock read of an email for the edit demo."""
    return f"Email content for ID: {email_id}"


@tool
def send_email_for_edit_demo(recipient: str, subject: str, body: str) -> str:
    """Mock send of an email. This demo rewrites the arguments before sending."""
    return f"Email sent to {recipient} with subject '{subject}' and body '{body}'"


def example_6_edit_tool_call():
    print_section(
        "EXAMPLE 6: HumanInTheLoopMiddleware — edit",
        "The model proposes an email to wrong@email.com.\n"
        "The human replaces the arguments, then the tool runs with the edited values.",
    )

    edit_demo_agent = create_agent(
        model="gpt-4o",
        tools=[read_email_for_edit_demo, send_email_for_edit_demo],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email_for_edit_demo": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    "read_email_for_edit_demo": False,
                }
            ),
        ],
    )

    edit_demo_thread_config = {"configurable": {"thread_id": "hitl-edit"}}

    edit_demo_paused_result = edit_demo_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Send email to wrong@email.com with subject 'Test' and body 'Hello'"
                )
            ]
        },
        config=edit_demo_thread_config,
    )

    print("[Example 6] State returned on the first invoke (this should contain the pause):")
    print(edit_demo_paused_result)

    if "__interrupt__" in edit_demo_paused_result:
        print("[Example 6] Paused before send_email_for_edit_demo. Decision: edit the arguments.")
        edit_demo_final_result = edit_demo_agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                # Must be the registered tool name.
                                "name": "send_email_for_edit_demo",
                                "args": {
                                    "recipient": "correct@email.com",
                                    "subject": "Corrected Subject",
                                    "body": "This was edited by human before sending",
                                },
                            },
                        }
                    ]
                }
            ),
            config=edit_demo_thread_config,
        )
        print(f"[Example 6] Reply after the edited send: {edit_demo_final_result['messages'][-1].content}")
    else:
        print("[Example 6] No pause was returned, so there was nothing to edit.")


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():
    print("\n============================================")
    print("LANGCHAIN V1 - MIDDLEWARE PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_summarize_by_message_count()

    # example_2_summarize_by_token_count()

    # example_3_summarize_by_fraction()

    # example_4_approve_tool_call()

    # example_5_reject_tool_call()

    # example_6_edit_tool_call()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# END OF MIDDLEWARE PRACTICE
# ============================================================
