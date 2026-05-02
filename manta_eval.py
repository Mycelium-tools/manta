"""
MANTA Inspect AI Evaluation
=============================
Dynamic multi-turn adversarial evaluation of animal welfare alignment.
All questions run as 5-turn conversations (turn_count=5):
  Turn 1: implicit-framing scenario; model answers
  Turn 2: explicit anchor — welfare made central, no adversarial pressure
  Turns 3–5: 3 distinct pressure turns explicitly contesting welfare

Scoring (manta_scorer.py):
- AWVS (Animal Welfare Value Stability) — headline metric, scored per-turn 0.0-1.0; benchmark = mean of turns 3-5
- AWMS (Animal Welfare Moral Sensitivity) — secondary metric, continuous 0.0-1.0 on Turn 1 only
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
from manta_scorer import manta_scorer, manta_per_turn_scorer
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
      --log-dir PATH               → explicit path, used as-is
      --full-run [LABEL]           → timestamped subdirectory inside the monthly base dir
      --sample-range START END     → sample_range_START_END_TIMESTAMP subdirectory
      MANTA_LOG_DIR env            → base dir (used as-is, or as parent for --full-run)
      MANTA_USER env               → logs/NAME_MonthYYYY (or subdirectory for --full-run)
      default                      → logs/

    Examples:
      python manta_eval.py --full-run           → logs/Allen_April2026/run_2026-04-25_143022/
      python manta_eval.py --full-run baseline  → logs/Allen_April2026/run_baseline_2026-04-25_143022/
      python manta_eval.py --sample-range 250 500 → logs/Allen_April2026/sample_range_250_500_2026-04-25_143022/
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

    # Detect --sample-range START END [LABEL] (for log routing when --full-run not specified).
    # Falls back to module-level globals since sys.argv may already be stripped by call time.
    sample_range_label = None
    if full_run_label is None:
        if args:
            for i, arg in enumerate(args):
                if arg == "--sample-range" and i + 2 < len(args):
                    base = f"sample_range_{args[i + 1]}_{args[i + 2]}"
                    if i + 3 < len(args) and not args[i + 3].startswith("--"):
                        base = f"{base}_{args[i + 3]}"
                    sample_range_label = base
                    break
        if sample_range_label is None:
            try:
                if SAMPLE_START is not None:
                    base = f"sample_range_{SAMPLE_START}_{SAMPLE_END}"
                    if SAMPLE_LABEL is not None:
                        base = f"{base}_{SAMPLE_LABEL}"
                    sample_range_label = base
            except NameError:
                pass

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
    elif sample_range_label is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_dir = os.path.join(base_dir, f"{sample_range_label}_{timestamp}")
    else:
        log_dir = base_dir

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_sample_range(args: list[str]) -> tuple[int | None, int | None, str | None]:
    """Parse --sample-range START END [LABEL] from argv. Returns (start, end, label)."""
    for i, arg in enumerate(args):
        if arg == "--sample-range" and i + 2 < len(args):
            label = None
            if i + 3 < len(args) and not args[i + 3].startswith("--"):
                label = args[i + 3]
            return int(args[i + 1]), int(args[i + 2]), label
    return None, None, None


def parse_tags(tags_val) -> list[str]:
    """Parse tags to a list, handling both actual lists and CSV string reprs."""
    if not tags_val:
        return []
    if isinstance(tags_val, list):
        return tags_val
    if not isinstance(tags_val, str):
        return []
    try:
        result = ast.literal_eval(tags_val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


SAMPLE_START, SAMPLE_END, SAMPLE_LABEL = get_sample_range(sys.argv[1:])

# Strip --sample-range and its args from sys.argv so inspect's CLI doesn't see unknown flags
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == "--sample-range" and _i + 2 < len(sys.argv):
        n = 4 if (_i + 3 < len(sys.argv) and not sys.argv[_i + 3].startswith("--")) else 3
        del sys.argv[_i:_i + n]
        break
    _i += 1

if SAMPLE_START is not None and not os.environ.get("MANTA_USER") and not os.environ.get("MANTA_LOG_DIR"):
    print(
        "\n[MANTA WARNING] --sample-range is set but MANTA_USER is not configured.\n"
        "  Logs will be saved to: logs/\n"
        "  For named log routing, add to ~/.zshrc:  export MANTA_USER=YOUR_NAME\n"
        "  Then run: source ~/.zshrc\n",
        file=sys.stderr,
        flush=True,
    )

AWVS_TAG = "Animal Welfare Value Stability"
TAG_ALIASES = {
    "Value Stability": "Animal Welfare Value Stability",
    "Contextual Welfare Salience": "Animal Welfare Moral Sensitivity",
}
KNOWN_TAGS = {AWVS_TAG, "Animal Welfare Moral Sensitivity", *TAG_ALIASES.keys()}


def normalize_tags(tags: list[str]) -> list[str]:
    """Map legacy tag names to the current AWMS/AWVS names."""
    return [TAG_ALIASES.get(tag, tag) for tag in tags]


def extract_tags(q: dict) -> list[str]:
    """Read tags, including malformed samples where the tag list became a column name."""
    tags = normalize_tags(parse_tags(q.get("tags", [])))
    if tags:
        return tags

    for key, value in q.items():
        key_tags = normalize_tags(parse_tags(key))
        value_tags = normalize_tags(parse_tags(value))
        if any(tag in KNOWN_TAGS for tag in key_tags):
            return key_tags
        if any(tag in KNOWN_TAGS for tag in value_tags):
            return value_tags
    return []


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


def load_samples(
    samples_file: str = "samples.json",
    start: int | None = None,
    end: int | None = None,
):
    """
    Load questions from samples.json and convert to Inspect AI Sample objects.

    start/end are Python slice indices applied to the final expanded sample list.
    If omitted, the module-level SAMPLE_START/SAMPLE_END (from --sample-range) are used.
    """
    with open(samples_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    questions = all_samples["all"]

    samples = []
    for q in questions:
        tags = extract_tags(q)
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

    _start = start if start is not None else SAMPLE_START
    _end = end if end is not None else SAMPLE_END
    if _start is not None or _end is not None:
        samples = samples[_start:_end]
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
        scorer=manta_per_turn_scorer()
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
        scorer=manta_per_turn_scorer()
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
        scorer=manta_per_turn_scorer()
    )


MODELS = [
    "anthropic/claude-opus-4-7",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.5",
    # "grok/grok-4.3",
    # "openai-api/deepseek/deepseek-v4-flash",
    # "mistral/mistral-small-2603",
    # "openrouter/meta-llama/llama-3.3-70b-instruct"
]
# MODELS = [
#     "google/gemini-3.1-flash-lite-preview",
#     "anthropic/claude-opus-4-7",
#     "anthropic/claude-sonnet-4-6",
#     "openai/gpt-5.5",
#     "grok/grok-4.3",
#     "openai-api/deepseek/deepseek-v4-flash",
#     "mistral/mistral-small-2603",
#     "openrouter/meta-llama/llama-3.3-70b-instruct"
# ]


def validate_environment(models: list[str]) -> None:
    """Fail fast for credentials required by the configured eval pipeline."""
    missing = []
    if any(model.startswith("anthropic/") for model in models) and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if any(model.startswith("grok/") for model in models) and not os.environ.get("XAI_API_KEY") and not os.environ.get("GROK_API_KEY"):
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

    if SAMPLE_START is not None and not os.environ.get("MANTA_USER") and not os.environ.get("MANTA_LOG_DIR"):
        confirm = input(
            "\n[MANTA] MANTA_USER is not set — logs will go to logs/.\n"
            "  Set it with: export MANTA_USER=YOUR_NAME && source ~/.zshrc\n"
            "  Proceed anyway? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        # clear_followup_store()  # re-enable if epoch_store=True

        for model in MODELS:
            print(f"\nRunning eval for model: {model}")
            eval(
                manta_5turn(),
                model=model,
                log_dir=log_dir,
                metadata={"epoch": epoch + 1},
                timeout=180,
                fail_on_error=False,
            )

    print(f"\nEvaluation complete! Ran {NUM_EPOCHS} epochs across {len(MODELS)} models.")
