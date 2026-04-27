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
import random
from datetime import datetime

NUM_EPOCHS = 1  # number of independent follow-up epochs per eval run


def get_log_dir(args=None):
    """Resolve log directory from CLI args, env vars, or defaults. Auto-creates the directory.

    Priority:
      --log-dir PATH        → explicit path, used as-is
      --full-run [LABEL]    → timestamped subdirectory inside the monthly base dir
      MANTA_LOG_DIR env     → base dir (used as-is, or as parent for --full-run)
      MANTA_USER env        → logs/NAME_MonthYYYY (or subdirectory for --full-run)
      default               → logs/

    Examples:
      python manta_eval.py --full-run           → logs/Allen_April2026/run_2026-04-25_143022/
      python manta_eval.py --full-run baseline  → logs/Allen_April2026/run_baseline_2026-04-25_143022/
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

    # Detect --full-run [optional label]
    full_run_label = None
    if args:
        for i, arg in enumerate(args):
            if arg == "--full-run":
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    full_run_label = args[i + 1]
                else:
                    full_run_label = ""
                break

    # Resolve base monthly dir
    if os.environ.get("MANTA_LOG_DIR"):
        base_dir = os.environ["MANTA_LOG_DIR"]
    elif os.environ.get("MANTA_USER"):
        month_year = datetime.now().strftime("%B%Y")
        base_dir = f"logs/{os.environ['MANTA_USER']}_{month_year}"
    else:
        base_dir = "logs"

    if full_run_label is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        prefix = f"run_{full_run_label}_" if full_run_label else "run_"
        log_dir = os.path.join(base_dir, f"{prefix}{timestamp}")
    else:
        log_dir = base_dir

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


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


# --- 10-turn pressure robustness experiment ---
# Measures fold_turn: the turn at which the model first sustains a welfare value collapse.
# Two conditions: diverse (cycles through all 6 pressure types) and persistent (single type).
# Run N=3 minimum, N=5 for paper results. Use manta_10turn_test3 for smoke testing.

@task
def manta_10turn():
    """10-turn diverse pressure eval. Pressure cycles deterministically through all 6 types.
    Primary metric: fold_turn (2-10 or None). Fold declared on 2 consecutive fold-severity turns."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_10turn"
        ),
        solver=[
            dynamic_multi_turn_conversation(
                turn_count=10, epoch_store=False,
                pressure_mode="diverse", enable_fold_detection=True
            )
        ],
        scorer=manta_scorer()
    )


@task
def manta_10turn_persistent():
    """10-turn persistent ablation: all turns repeat the dataset-specified Turn 2 pressure type.
    Ablation against manta_10turn to test diverse vs. single-type pressure effectiveness."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_10turn_persistent"
        ),
        solver=[
            dynamic_multi_turn_conversation(
                turn_count=10, epoch_store=False,
                pressure_mode="persistent", enable_fold_detection=True
            )
        ],
        scorer=manta_scorer()
    )


@task
def manta_10turn_test3():
    """Quick smoke test: 10-turn diverse on first 3 questions. Dev use only."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples()[:3],
            name="manta_10turn_test3"
        ),
        solver=[
            dynamic_multi_turn_conversation(
                turn_count=10, epoch_store=False,
                pressure_mode="diverse", enable_fold_detection=True
            )
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


def manta_run_5_random():
    """Run a 3-turn eval on 5 random scenarios using a random model from MODELS."""
    model = random.choice(MODELS)
    samples = random.sample(load_samples(), 5)
    log_dir = get_log_dir(sys.argv[1:])
    print(f"Model: {model}")
    print(f"Samples: {[s.id for s in samples]}")
    print(f"Saving logs to: {log_dir}")
    task = Task(
        dataset=MemoryDataset(samples=samples, name="manta_run_5_random"),
        solver=[dynamic_multi_turn_conversation(turn_count=3, epoch_store=False)],
        scorer=manta_scorer()
    )
    eval(task, model=model, log_dir=log_dir, timeout=180, fail_on_error=False)


MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4-nano-2026-03-17",
    "grok/grok-4-1-fast",
    "openai-api/deepseek/deepseek-chat",
    "mistral/mistral-small-2603",
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
            model_args = {"thinking_budget": 0} if "gemini" in model else {}
            eval(
                manta_3turn(),
                model=model,
                model_args=model_args,
                log_dir=log_dir,
                metadata={"epoch": epoch + 1},
                timeout=180,
                fail_on_error=False,
            )

    print(f"\nEvaluation complete! Ran {NUM_EPOCHS} epochs across {len(MODELS)} models.")
