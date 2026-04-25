"""
MANTA Inspect AI Evaluation
=============================
Dynamic multi-turn adversarial evaluation of animal welfare alignment.
All questions run as 3-turn conversations.

Scoring:
- Uses manta_scorer (manta_scorer.py)
- Dimensions: Pressure Robustness (PR) + Contextual Welfare Salience (CWS)
- PR is scored per-turn (Turn 2 and Turn 3 separately); CWS gates PR (if CWS=0, PR=N/A)
- Custom metrics: mean_pr_turn_2, mean_pr_turn_3, worst_turn_pr, degradation_delta, cws_gate_rate

Tasks:
- manta_test5         — first 5 questions, 3-turn (quick sanity check)
- manta_3turn         — all questions, 3-turn (MVP full eval)
- manta_agentic_3turn — all questions, 3-turn, model can use web search

Usage:
    # Quick test
    inspect eval manta_eval.py@manta_test5 --model anthropic/claude-sonnet-4-6

    # Full eval
    inspect eval manta_eval.py@manta_3turn --model anthropic/claude-sonnet-4-6

    # Run all models across NUM_EPOCHS epochs
    python manta_eval.py
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import use_tools
from inspect_ai.tool import web_search
from dynamic_multiturn_solver import dynamic_multi_turn_conversation, clear_followup_store
from manta_scorer import manta_scorer
import json
import ast
import os
import sys
from datetime import datetime

NUM_EPOCHS = 2  # number of independent follow-up epochs per eval run


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


def load_samples(samples_file: str = "samples.json"):
    """
    Load all questions from samples.json and convert to Inspect AI Sample objects.

    Returns:
        List of Inspect AI Sample objects
    """
    with open(samples_file, "r") as f:
        all_samples = json.load(f)

    questions = all_samples["all"]

    samples = []
    for q in questions:
        tags = parse_tags(q.get("tags", []))
        animals = q.get("animals", [])
        base_metadata = {
            "tags": tags,
            "language": q.get("language", "en"),
            "pressure": q.get("pressure") or ["economic", "economic"],
            "reference_answer": q.get("reference_answer", ""),
        }
        if animals:
            # Expand into one Sample per animal, substituting {{animal}} in question text
            for animal in animals:
                question_text = q["question"].replace("{{animal}}", animal)
                samples.append(Sample(
                    input=question_text,
                    target=json.dumps({"tags": tags}),
                    id=f"{q['id']}_{animal}",
                    metadata={**base_metadata, "animal": animal, "base_id": str(q["id"])},
                ))
        else:
            samples.append(Sample(
                input=q["question"],
                target=json.dumps({"tags": tags}),
                id=str(q["id"]),
                metadata=base_metadata,
            ))
    return samples


@task
def manta_test5():
    """Quick 3-turn eval on the first 5 questions from samples.json."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples()[:5],
            name="manta_test5"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=3, epoch_store=False)
        ],
        scorer=manta_scorer()
    )


@task
def manta_3turn():
    """MANTA 3-turn evaluation (all questions from samples.json)."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_3turn"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=3, epoch_store=False)
        ],
        scorer=manta_scorer()
    )


# --- Agentic variant (model has access to web_search tool) ---
# Requires a search API key in .env: TAVILY_API_KEY or GOOGLE_CSE_ID + GOOGLE_CSE_API_KEY

@task
def manta_agentic_3turn():
    """Agentic 3-turn eval — model can use web search."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_agentic_3turn"
        ),
        solver=[
            use_tools([web_search()]),
            dynamic_multi_turn_conversation(turn_count=3, epoch_store=False)
        ],
        scorer=manta_scorer()
    )


MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4-nano-2026-03-17",
    "grok/grok-4-1-fast",
    "openai-api/deepseek/deepseek-chat",
    "mistral/mistral-large-latest",
    "openrouter/meta-llama/llama-3.1-8b-instruct"
]

if __name__ == "__main__":
    log_dir = get_log_dir(sys.argv[1:])
    print(f"Saving logs to: {log_dir}")

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        # clear_followup_store()  # re-enable if epoch_store=True

        for model in MODELS:
            print(f"\nRunning eval for model: {model}")
            eval(
                manta_3turn(),
                model=model,
                log_dir=log_dir,
                metadata={"epoch": epoch + 1},
                timeout=180,
            )

    print(f"\nEvaluation complete! Ran {NUM_EPOCHS} epochs across {len(MODELS)} models.")
