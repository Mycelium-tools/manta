"""
MANTA Inspect AI Evaluation
=============================
Dynamic multi-turn adversarial evaluation of animal welfare alignment.

Scoring:
- Uses manta_scorer (multidimensional_scorer.py)
- Each sample's target field carries its dimension tags as JSON: {"tags": ["Pressure Robustness"]}
- Scorer runs one LLM grader call per dimension; overall score = weighted average (no extra LLM call)
- To add dimensions: update MANTA_DIMENSIONS in multidimensional_scorer.py and tag questions in Google Sheets

Tasks:
- manta_test5         — first 5 questions, 2-turn (quick sanity check)
- manta_2turn         — all 2-turn questions
- manta_3turn         — all 3-turn questions
- manta_agentic_*     — agentic variants where the model can use web search

Usage:
    inspect eval manta_eval.py@manta_test5 --model anthropic/claude-sonnet-4-20250514
    python manta_eval.py   (runs all tasks across all models in MODELS list)
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import use_tools
from inspect_ai.tool import web_search
from dynamic_multiturn_solver import dynamic_multi_turn_conversation
from manta_scorer import manta_scorer
import json
import ast
import os
import sys
from datetime import datetime



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


def load_samples(turn_key: str, samples_file: str = "samples.json"):
    """
    Load questions from samples.json and convert to Inspect AI Sample objects.

    Args:
        turn_key: "2_turn", "3_turn", or "4_turn"
        samples_file: Path to samples.json

    Returns:
        List of Inspect AI Sample objects
    """
    with open(samples_file, "r") as f:
        all_samples = json.load(f)

    questions = all_samples[turn_key]

    samples = []
    for q in questions:
        tags = parse_tags(q.get("tags", []))
        samples.append(Sample(
            input=q["question"],
            # target carries per-sample dimension tags so the scorer knows
            # which dimensions to evaluate for this specific question
            target=json.dumps({"tags": tags}),
            id=str(q["id"]),
            metadata={
                "tags": tags,
                "language": q.get("language", "en"),
                # pressure type for Turn 2 follow-up (e.g. "economic", "social")
                # subsequent turns are chosen dynamically by the follow-up generator
                "pressure": q.get("pressure") or "economic",
            }
        ))
    return samples


@task
def manta_test5():
    """Quick 2-turn eval on the first 5 questions from samples.json."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("2_turn")[:5],
            name="manta_test5"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=2, epoch_store=True)
        ],
        scorer=manta_scorer()
    )


@task
def manta_2turn():
    """MANTA dynamic 2-turn evaluation (all 2-turn questions from samples.json)."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("2_turn"),
            name="manta_2turn"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=2, epoch_store=True)
        ],
        scorer=manta_scorer()
    )


@task
def manta_3turn():
    """MANTA dynamic 3-turn evaluation (all 3-turn questions from samples.json)."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("3_turn"),
            name="manta_3turn"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=3, epoch_store=True)
        ],
        scorer=manta_scorer()
    )


# --- Agentic variants (model has access to web_search tool) ---
# Requires a search API key in .env: TAVILY_API_KEY or GOOGLE_CSE_ID + GOOGLE_CSE_API_KEY

@task
def manta_agentic_test5():
    """Agentic 2-turn eval — first 5 questions, model can use web search."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("2_turn")[:5],
            name="manta_agentic_test5"
        ),
        solver=[
            use_tools([web_search()]),
            dynamic_multi_turn_conversation(turn_count=2, epoch_store=True)
        ],
        scorer=manta_scorer()
    )


@task
def manta_agentic_2turn():
    """Agentic 2-turn eval — model can use web search."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("2_turn"),
            name="manta_agentic_2turn"
        ),
        solver=[
            use_tools([web_search()]),
            dynamic_multi_turn_conversation(turn_count=2, epoch_store=True)
        ],
        scorer=manta_scorer()
    )


@task
def manta_agentic_3turn():
    """Agentic 3-turn eval — model can use web search."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples("3_turn"),
            name="manta_agentic_3turn"
        ),
        solver=[
            use_tools([web_search()]),
            dynamic_multi_turn_conversation(turn_count=3, epoch_store=True)
        ],
        scorer=manta_scorer()
    )

# test on frontier models
MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4-mini",
    "gemini-3-flash-preview"
]

if __name__ == "__main__":
    log_dir = get_log_dir(sys.argv[1:])
    print(f"Saving logs to: {log_dir}")
    for model in MODELS:
        print(f"\nRunning eval for model: {model}")
        eval(
            manta_test5(),
            model=model,
            log_dir=log_dir
        )
    print(f"\nEvaluation complete! Ran 2 tasks across {len(MODELS)} models.")

    # DEBUG: show hybrid-epoch follow-up store (remove after verifying)
    from dynamic_multiturn_solver import _followup_store
    print("\n=== Followup store (first 2 entries) ===")
    for qid in list(_followup_store.keys())[:2]:
        print(f"Q{qid}:", json.dumps(_followup_store[qid], indent=2))
