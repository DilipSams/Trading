"""
Multi-Agent System Runner
=========================
Entry point to run the multi-agent system with a task.

Usage:
    python run.py "Build a REST API for a todo app"
    python run.py --interactive
    python run.py --demo
"""

import asyncio
import argparse
import sys
import os

from orchestrator import run as orchestrator_run

DEMO_TASKS = [
    "Research Python best practices for error handling, then create a utility "
    "module called error_utils.py with custom exception classes and decorators "
    "for retry logic. Review the code when done.",
    "Create a simple Python CLI calculator that supports +, -, *, / and "
    "handles edge cases like division by zero. Include unit tests.",
    "Research the latest FastAPI patterns, then build a minimal REST API "
    "with health check and CRUD endpoints for a notes resource using "
    "in-memory storage. Review for security issues.",
]

async def interactive_mode():
    print("\n  Multi-Agent System - Interactive Mode")
    print("  Type a task and press Enter. Type 'quit' to stop.\n")
    working_dir = os.getcwd()
    while True:
        try:
            task = input("\nYour task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        await orchestrator_run(task, working_dir=working_dir)

async def demo_mode():
    print("\n  Multi-Agent System - Demo Mode\n")
    print("Available demo tasks:")
    for i, task in enumerate(DEMO_TASKS, 1):
        preview = task[:80] + "..." if len(task) > 80 else task
        print(f"  {i}. {preview}")
    try:
        choice = input("\nPick a task number (or press Enter for #1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(DEMO_TASKS) - 1))
    task = DEMO_TASKS[idx]
    await orchestrator_run(task, working_dir=os.getcwd())

async def main():
    parser = argparse.ArgumentParser(description="Multi-Agent System")
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--demo", "-d", action="store_true")
    parser.add_argument("--working-dir", "-w", default=os.getcwd())
    args = parser.parse_args()
    if args.interactive:
        await interactive_mode()
    elif args.demo:
        await demo_mode()
    elif args.task:
        await orchestrator_run(args.task, working_dir=args.working_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
