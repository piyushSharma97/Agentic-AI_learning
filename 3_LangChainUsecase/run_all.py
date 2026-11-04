"""
Interactive Runner for all 10 LangChain Use Cases.
Run this script to select and execute any individual example or run all of them.
"""

import sys
import subprocess
from pathlib import Path

EXAMPLES = [
    ("1", "Prompt Management", "1_prompt_management.py"),
    ("2", "Structured Output", "2_structured_output.py"),
    ("3", "Memory", "3_memory.py"),
    ("4", "Tool Calling", "4_tool_calling.py"),
    ("5", "Agents", "5_agent.py"),
    ("6", "Chains & LCEL", "6_chain.py"),
    ("7", "Document Processing & RAG", "7_document_processing.py"),
    ("8", "Model Switching", "8_model_switching.py"),
    ("9", "Multi-Agent System", "9_multi_agent_system.py"),
    ("10", "Evaluation", "10_evaluation.py"),
]

BASE_DIR = Path(__file__).resolve().parent


def run_script(script_name: str):
    script_path = BASE_DIR / script_name
    print(f"\n{'='*75}")
    print(f"Executing: {script_name}")
    print(f"{'='*75}\n")
    subprocess.run([sys.executable, str(script_path)])


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        if choice.lower() == "all":
            for _, _, script in EXAMPLES:
                run_script(script)
            return
        for num, title, script in EXAMPLES:
            if choice == num or choice.lower() == script.lower():
                run_script(script)
                return
        print(f"Unknown selection: '{choice}'")
        return

    print("=" * 65)
    print("   LANGCHAIN USE CASES: RUNNER MENU")
    print("=" * 65)
    for num, title, script in EXAMPLES:
        print(f" [{num:>2}] {title:<30} ({script})")
    print(" [all] Run all 10 examples sequentially")
    print(" [ q ] Exit")
    print("=" * 65)

    choice = input("\nEnter choice (1-10, all, q): ").strip()
    if choice.lower() in ("q", "quit", "exit"):
        print("Goodbye!")
        return
    elif choice.lower() == "all":
        for _, _, script in EXAMPLES:
            run_script(script)
    else:
        for num, title, script in EXAMPLES:
            if choice == num:
                run_script(script)
                return
        print("Invalid option selected.")


if __name__ == "__main__":
    main()

