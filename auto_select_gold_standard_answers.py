"""
Score generated MANTA GSA candidates and automatically select the best per sample.

Default ranking:
1. Highest AWVS (`3turn_mean`)
2. Ties broken by highest AWMS (`awms_score`)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset

from gold_standard import (
    DEFAULT_SELECTED_PATH,
    DEFAULT_VALIDATION_DIR,
    SCHEMA_VERSION,
    choose_best_candidates,
    load_selected,
    read_candidate_traces,
    validate_trace,
    write_json,
    write_selected,
)
from manta_scorer import manta_per_turn_scorer
from oracle_solver import gold_standard_oracle_solver

load_dotenv(dotenv_path=Path(".env"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate .eval or .jsonl file")
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED_PATH))
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--model", default="mockllm/model")
    parser.add_argument("--log-dir", default="logs/gsa_candidate_scoring")
    parser.add_argument("--output", default=None, help="Candidate scoring report JSON")
    parser.add_argument("--no-write-selected", action="store_true")
    return parser.parse_args()


def trace_input(trace: dict) -> str:
    for message in trace.get("messages", []):
        if message.get("turn") == 1 and message.get("role") == "user":
            return message["content"]
    return trace.get("question", "")


def trace_target(trace: dict) -> str:
    import json

    return json.dumps({"tags": trace.get("tags", [])})


def trace_metadata(trace: dict) -> dict:
    return {
        "tags": trace.get("tags", []),
        "pressure": trace.get("pressure", []),
        "candidate_id": trace["candidate_id"],
        "original_sample_id": str(trace["sample_id"]),
    }


def candidate_sample(trace: dict):
    from inspect_ai.dataset import Sample

    return Sample(
        input=trace_input(trace),
        target=trace_target(trace),
        id=trace["candidate_id"],
        metadata=trace_metadata(trace),
    )


def selected_for_candidates(candidates: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Temporary selected payload for scoring every generated GSA candidate. "
            "Keys are candidate ids so multiple candidates for one sample can be replayed."
        ),
        "selected": {
            candidate["candidate_id"]: candidate
            for candidate in candidates
        },
    }


def extract_score_metadata(sample) -> dict:
    score = next(iter((sample.scores or {}).values()), None)
    if score is None:
        return {}
    return {
        "score": score.value,
        "3turn_mean": score.metadata.get("3turn_mean"),
        "awms_score": score.metadata.get("awms_score"),
        "5turn_mean": score.metadata.get("5turn_mean"),
        "per_turn_scores": score.metadata.get("per_turn_scores"),
        "per_turn_explanations": score.metadata.get("per_turn_explanations"),
        "per_turn_judge": score.metadata.get("per_turn_judge"),
        "awms_judge": score.metadata.get("awms_judge"),
        "awms_explanation": score.metadata.get("awms_explanation"),
    }


def auto_select_candidates(
    candidates_path: str | Path,
    selected_path: str | Path = DEFAULT_SELECTED_PATH,
    turn_count: int = 5,
    model: str = "mockllm/model",
    log_dir: str = "logs/gsa_candidate_scoring",
    output: str | Path | None = None,
    update_selected: bool = True,
) -> dict:
    candidates = read_candidate_traces(candidates_path)
    if not candidates:
        raise ValueError(f"No candidates found in {candidates_path}")

    errors_by_candidate = {
        candidate["candidate_id"]: validate_trace(candidate, turn_count=turn_count)
        for candidate in candidates
    }
    errors_by_candidate = {k: v for k, v in errors_by_candidate.items() if v}
    if errors_by_candidate:
        raise ValueError(f"Invalid candidate traces: {errors_by_candidate}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_selected = Path(output).with_suffix(".selected.json") if output else (
        DEFAULT_VALIDATION_DIR / f"candidate_scoring_{timestamp}.selected.json"
    )
    report_path = Path(output) if output else (
        DEFAULT_VALIDATION_DIR / f"candidate_scoring_{timestamp}.json"
    )
    write_json(temp_selected, selected_for_candidates(candidates))

    task = Task(
        dataset=MemoryDataset(
            samples=[candidate_sample(candidate) for candidate in candidates],
            name="manta_gsa_candidate_scoring",
        ),
        solver=[
            gold_standard_oracle_solver(turn_count=turn_count, selected_path=str(temp_selected))
        ],
        scorer=manta_per_turn_scorer(),
    )
    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
        fail_on_error=False,
        display="none",
    )

    by_candidate = {candidate["candidate_id"]: candidate for candidate in candidates}
    scoring_rows = []
    for log in logs:
        for sample in log.samples or []:
            candidate_id = str(sample.id)
            candidate = by_candidate[candidate_id]
            validation = extract_score_metadata(sample)
            candidate["validation"] = validation
            scoring_rows.append({
                "candidate_id": candidate_id,
                "sample_id": str(candidate["sample_id"]),
                "source_model": candidate.get("source_model"),
                "validation": validation,
                "log": log.location,
            })

    winners = choose_best_candidates(candidates)
    winner_rows = []
    for sample_id, candidate in sorted(winners.items()):
        validation = candidate.get("validation", {})
        winner_rows.append({
            "sample_id": sample_id,
            "candidate_id": candidate["candidate_id"],
            "source_model": candidate.get("source_model"),
            "3turn_mean": validation.get("3turn_mean"),
            "awms_score": validation.get("awms_score"),
        })

    report = {
        "candidates": str(candidates_path),
        "ranking": "highest 3turn_mean/AWVS, tie-break highest awms_score",
        "scored": scoring_rows,
        "winners": winner_rows,
        "temp_selected": str(temp_selected),
    }
    write_json(report_path, report)

    if update_selected:
        selected = load_selected(selected_path)
        selected["selected"].update(winners)
        write_selected(selected_path, selected)

    print(f"Scored {len(scoring_rows)} candidate(s); report written to {report_path}")
    if not update_selected:
        print("Did not update selected GSAs because --no-write-selected was set")
    else:
        print(f"Selected {len(winner_rows)} winner(s) into {selected_path}")
    for row in winner_rows:
        print(
            f"{row['sample_id']}: {row['candidate_id']} "
            f"(AWVS={row['3turn_mean']}, AWMS={row['awms_score']})"
        )
    return report


def main() -> None:
    args = parse_args()
    try:
        auto_select_candidates(
            candidates_path=args.candidates,
            selected_path=args.selected,
            turn_count=args.turns,
            model=args.model,
            log_dir=args.log_dir,
            output=args.output,
            update_selected=not args.no_write_selected,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
