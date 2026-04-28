"""
Run a MANTA eval on a single question by ID.
=============================
Pulls from samples.json. Uses the same manta_scorer as the full eval:
per-sample dimensional scoring (tags read from target.text), overall score
computed as weighted average — no separate overall LLM call.

Usage:
    python run_single_eval.py <question_id>
    python run_single_eval.py <question_id> --turns 5
    python run_single_eval.py <question_id> --agentic
    python run_single_eval.py <question_id> --agentic --model openai/gpt-4o
    python run_single_eval.py <question_id> --animal cricket
    python run_single_eval.py <question_id> --log-dir logs/Allen_March2026

Example:
    python run_single_eval.py 16
    python run_single_eval.py 16 --turns 6
    python run_single_eval.py 16 --agentic
    python run_single_eval.py 16 --agentic --model openai/gpt-5.4-mini

--turns overrides the default turn count (default: 5).
Valid values: 5, 10

Log directory resolution (first match wins):
    1. --log-dir <path> CLI flag
    2. MANTA_LOG_DIR environment variable
    3. MANTA_USER environment variable → auto-generates logs/{MANTA_USER}_{Month}{Year}
    4. Default: logs/
"""

import sys
import json
import ast
import os
from datetime import datetime
from inspect_ai import eval
from inspect_ai import Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import chain, solver, use_tools
from inspect_ai.tool import web_search
from dynamic_multiturn_solver import dynamic_multi_turn_conversation
from manta_scorer import manta_scorer


def get_log_dir(args=None):
    """Resolve log directory from --log-dir arg, MANTA_LOG_DIR env, MANTA_USER env, or default logs/.
    Auto-creates the directory if it doesn't exist. If it already exists, reuses it.
    Set MANTA_USER=Allen in your shell to auto-route to logs/Allen_March2026 each month.
    """
    if args:
        for i, arg in enumerate(args):
            if arg.startswith("--log-dir="):
                log_dir = arg.split("=", 1)[1]
                os.makedirs(log_dir, exist_ok=True)
                return log_dir
            elif arg == "--log-dir" and i + 1 < len(args):
                log_dir = args[i + 1]
                os.makedirs(log_dir, exist_ok=True)
                return log_dir
    if os.environ.get("MANTA_LOG_DIR"):
        log_dir = os.environ["MANTA_LOG_DIR"]
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    if os.environ.get("MANTA_USER"):
        month_year = datetime.now().strftime("%B%Y")
        log_dir = f"logs/{os.environ['MANTA_USER']}_{month_year}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    return "logs"


def parse_tags(tags_val) -> list[str]:
    """Parse tags to a list, handling both actual lists and CSV string reprs."""
    if not tags_val:
        return []
    if isinstance(tags_val, list):
        return tags_val
    try:
        result = ast.literal_eval(tags_val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def find_question(question_id: int, samples_file: str = "samples.json"):
    """Find a question by ID across all turn groups. Returns (question, turn_count)."""
    with open(samples_file, "r") as f:
        all_samples = json.load(f)

    for q in all_samples["all"]:
        if q["id"] == question_id:
            return q, 5

    return None, None

@solver
def customSolver(_isAgentic, _turnCount):
    steps = []
    if _isAgentic:
        steps.append(use_tools([web_search()]))
    steps.append(dynamic_multi_turn_conversation(turn_count=_turnCount))
    return chain(*steps)


def main():
    # print error message
    if len(sys.argv) < 2:
        print("Usage: python run_single_eval.py <question_id> [--agentic]")
        sys.exit(1)

    question_id = int(sys.argv[1])
    agentic = "--agentic" in sys.argv

    model = "anthropic/claude-sonnet-4-20250514"
    turns_override = None
    animal_override = None
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg == "--model" and sys.argv.index(arg) + 1 < len(sys.argv):
            model = sys.argv[sys.argv.index(arg) + 1]
        elif arg.startswith("--turns="):
            turns_override = int(arg.split("=", 1)[1])
        elif arg == "--turns" and sys.argv.index(arg) + 1 < len(sys.argv):
            turns_override = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg.startswith("--animal="):
            animal_override = arg.split("=", 1)[1]
        elif arg == "--animal" and sys.argv.index(arg) + 1 < len(sys.argv):
            animal_override = sys.argv[sys.argv.index(arg) + 1]

    if turns_override is not None and turns_override not in (5, 10):
        print(f"Error: --turns must be 5 or 10 (got {turns_override})")
        sys.exit(1)

    question, turn_count = find_question(question_id)
    if turns_override is not None:
        turn_count = turns_override

    if question is None:
        print(f"Error: Question ID {question_id} not found in samples.json")
        sys.exit(1)

    if not question.get("question"):
        print(f"Error: Question ID {question_id} has no question text.")
        sys.exit(1)

    # Resolve animal: use --animal override, else first animal in list, else None
    animals = question.get("animals", [])
    animal = animal_override or (animals[0] if animals else None)
    if animal and "{{animal}}" not in question.get("question", ""):
        print(f"Warning: --animal specified but question {question_id} has no {{{{animal}}}} placeholder. Ignoring.")
        animal = None

    question_text = question["question"].replace("{{animal}}", animal) if animal else question["question"]
    sample_id = f"{question['id']}_{animal}" if animal else str(question["id"])

    mode = "agentic (web search enabled)" if agentic else "standard"
    turn_source = "overridden" if turns_override is not None else "from samples.json"
    animal_note = f", animal={animal}" if animal else ""
    print(f"Running eval on question {question_id} ({turn_count}-turn {turn_source}, {mode}{animal_note})")
    print(f"Tags: {question.get('tags', 'none')}")
    print(f"Question: {question_text[:120]}...")

    tags = parse_tags(question.get("tags", []))
    pressure = question.get("pressure") or ["economic", "economic"]
    metadata = {
        "tags": tags,
        "language": question.get("language", "en"),
        "pressure": pressure,
    }
    if animal:
        metadata["animal"] = animal
        metadata["base_id"] = str(question["id"])

    sample = Sample(
        input=question_text,
        target=json.dumps({"tags": tags}),
        id=sample_id,
        metadata=metadata,
    )

    test_task = Task(
        dataset=MemoryDataset(samples=[sample], name=f"manta_single_{question_id}"),
        solver=customSolver(agentic, turn_count),
        scorer=manta_scorer()
    )

    log_dir = get_log_dir(sys.argv[1:])
    print(f"Saving logs to: {log_dir}")
    eval([test_task], model=model, log_dir=log_dir, timeout=180)


if __name__ == "__main__":
    main()
