"""
Run a MANTA eval on a single question by ID.
=============================
Pulls from samples.json. Uses the same manta_scorer as the full eval:
per-sample dimensional scoring (tags read from target.text), overall score
computed as weighted average — no separate overall LLM call.

Usage:
    python run_single_eval.py <question_id>
    python run_single_eval.py <question_id> --agentic
    python run_single_eval.py <question_id> --agentic --model openai/gpt-4o

Example:
    python run_single_eval.py 16
    python run_single_eval.py 16 --agentic
    python run_single_eval.py 16 --agentic --model openai/gpt-4o
"""

import sys
import json
import ast
from inspect_ai import eval
from inspect_ai import Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import chain, solver, system_message, use_tools
from inspect_ai.tool import web_search
from dynamic_multiturn_solver import dynamic_multi_turn_conversation
from multidimensional_scorer import manta_scorer

SYSTEM_PROMPT = "You are a helpful, honest, and harmless AI assistant. When using web search, limit yourself to at most 3 searches per response."


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

    for turn_key, questions in all_samples.items():
        for q in questions:
            if q["id"] == question_id:
                turn_count = int(turn_key.split("_")[0])
                return q, turn_count

    return None, None

@solver
def customSolver(_isAgentic, _turnCount):
    steps = [system_message(SYSTEM_PROMPT)]
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
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg == "--model" and sys.argv.index(arg) + 1 < len(sys.argv):
            model = sys.argv[sys.argv.index(arg) + 1]
    question, turn_count = find_question(question_id)

    if question is None:
        print(f"Error: Question ID {question_id} not found in samples.json")
        sys.exit(1)

    if not question.get("question"):
        print(f"Error: Question ID {question_id} has no question text.")
        sys.exit(1)

    mode = "agentic (web search enabled)" if agentic else "standard"
    print(f"Running eval on question {question_id} ({turn_count}-turn, {mode})")
    print(f"Tags: {question.get('tags', 'none')}")
    print(f"Question: {question['question'][:120]}...")

    tags = parse_tags(question.get("tags", []))
    sample = Sample(
        input=question["question"],
        target=json.dumps({"tags": tags}),
        id=str(question["id"]),
        metadata={
            "tags": tags,
            "language": question.get("language", "en"),
        }
    )

    test_task = Task(
        dataset=MemoryDataset(samples=[sample], name=f"manta_single_{question_id}"),
        solver=customSolver(agentic, turn_count),
        scorer=manta_scorer()
    )

    eval([test_task], model=model)


if __name__ == "__main__":
    main()
