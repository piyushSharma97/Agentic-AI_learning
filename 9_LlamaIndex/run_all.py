"""
Run every LlamaIndex lesson in order.

Each lesson's main() already calls its examples. Comment a call here
to skip a whole lesson, or comment a call inside that lesson to skip
one example.
"""

from importlib import import_module


LESSONS = [
    "01_quickstart_rag",
    "02_documents_and_nodes",
    "03_settings_and_models",
    "04_retrieval_and_chat_engine",
    "05_agent_with_tools",
]


def main() -> None:
    for name in LESSONS:
        print("\n" + "#" * 72)
        print(f"# {name}")
        print("#" * 72)
        import_module(name).main()


if __name__ == "__main__":
    main()
