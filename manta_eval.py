"""
MANTA Inspect AI Evaluation
=============================
Dynamic multi-turn adversarial evaluation of animal welfare alignment.
All questions run as 5-turn conversations (turn_count=5):
  Turn 1: implicit-framing scenario; model answers (AWMS scored on this response)
  Turn 2: explicit anchor — welfare made central, no adversarial pressure
  Turns 3–5: 3 distinct pressure turns explicitly contesting welfare (AWVS scored on this slice)

Scoring (manta_scorer.py):
- AWVS (Animal Welfare Value Stability, Rest Component 4) — headline metric, continuous 0.0-1.0
- AWMS (Animal Welfare Moral Sensitivity, Rest Component 1) — secondary metric, snapped to {0.0, 0.5, 1.0}
- AWMS does NOT gate AWVS; both are computed independently so the recognition-vs-resilience
  correlation can be measured (see analysis/correlation_awms_awvs.py)

Tasks:
- manta_test5         — first 5 questions, 5-turn (smoke test)
- manta_5turn         — all questions, 5-turn (primary eval)
- manta_agentic_5turn — all questions, 5-turn, model can use web search

Usage:
    # Smoke test
    inspect eval manta_eval.py@manta_test5 --model anthropic/claude-sonnet-4-6

    # Full eval
    inspect eval manta_eval.py@manta_5turn --model anthropic/claude-sonnet-4-6

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
from dotenv import load_dotenv

load_dotenv()

if os.environ.get("GROK_API_KEY") and not os.environ.get("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.environ["GROK_API_KEY"]

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


AWVS_TAG = "Animal Welfare Value Stability"


def _validate_pressure_list(qid, tags, pressure_list):
    """Each AWVS-tagged question must list exactly 3 distinct pressure types
    (one per pressure turn: Turn 3, 4, 5). Raises ValueError naming the offending qid."""
    if AWVS_TAG not in tags:
        return
    if len(pressure_list) != 3:
        raise ValueError(
            f"Question {qid}: AWVS-tagged samples must have exactly 3 pressure types "
            f"(got {len(pressure_list)}: {pressure_list})"
        )
    if len(set(pressure_list)) != 3:
        raise ValueError(
            f"Question {qid}: AWVS-tagged samples must have 3 DISTINCT pressure types "
            f"(got {pressure_list} with duplicates)"
        )


def load_samples(samples_file: str = "samples.json"):
    """
    Load all questions from samples.json and convert to Inspect AI Sample objects.

    Returns:
        List of Inspect AI Sample objects
    """
    with open(samples_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    questions = all_samples["all"]

    samples = []
    for q in questions:
        tags = parse_tags(q.get("tags", []))
        animals = q.get("animals", [])
        pressure = q.get("pressure") or ["economic", "social", "pragmatic"]
        _validate_pressure_list(q.get("id"), tags, pressure)
        base_metadata = {
            "tags": tags,
            "language": q.get("language", "en"),
            "pressure": pressure,
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
    """Smoke 5-turn eval on the first 5 questions from samples.json."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples()[:5],
            name="manta_test5"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=5, epoch_store=False)
        ],
        scorer=manta_scorer()
    )


@task
def manta_5turn():
    """MANTA 5-turn evaluation (all questions from samples.json).
    Turn 1: initial answer (AWMS). Turn 2: anchor. Turns 3-5: 3 pressure turns (AWVS)."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_5turn"
        ),
        solver=[
            dynamic_multi_turn_conversation(turn_count=5, epoch_store=False)
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
def manta_agentic_5turn():
    """Agentic 5-turn eval — model can use web search."""
    return Task(
        dataset=MemoryDataset(
            samples=load_samples(),
            name="manta_agentic_5turn"
        ),
        solver=[
            use_tools([web_search()]),
            dynamic_multi_turn_conversation(turn_count=5, epoch_store=False)
        ],
        scorer=manta_scorer()
    )



# Smoke-test models for the methodology-lock phase.
# Restore the full list (Haiku, Gemini, GPT, DeepSeek, Mistral, Llama) for paper-grade runs.
MODELS = [
    "grok/grok-4-1-fast",
    "anthropic/claude-sonnet-4-6",
]


def validate_environment(models: list[str]) -> None:
    """Fail fast for credentials required by the configured eval pipeline."""
    missing = []
    if any(model.startswith("anthropic/") for model in models):
        missing.append("ANTHROPIC_API_KEY")
    if any(model.startswith("grok/") for model in models):
        missing.append("XAI_API_KEY or GROK_API_KEY")
    # Follow-up generation and the default scorer panel currently use Anthropic.
    if "ANTHROPIC_API_KEY" not in os.environ:
        missing.append("ANTHROPIC_API_KEY")

    missing = list(dict.fromkeys(missing))
    if missing:
        raise RuntimeError(
            "Missing required API credentials: "
            + ", ".join(missing)
            + ". Add them to .env or export them before running manta_eval.py."
        )

if __name__ == "__main__":
    validate_environment(MODELS)
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
                manta_5turn(),
                model=model,
                model_args=model_args,
                log_dir=log_dir,
                metadata={"epoch": epoch + 1},
                timeout=180,
                fail_on_error=False,
            )

    print(f"\nEvaluation complete! Ran {NUM_EPOCHS} epochs across {len(MODELS)} models.")
