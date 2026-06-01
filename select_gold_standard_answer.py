"""
Review and select generated MANTA gold standard answer candidates.

Candidate files can be Inspect .eval logs or JSONL exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gold_standard import DEFAULT_SELECTED_PATH, load_selected, read_candidate_traces, validate_trace, write_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate .eval or .jsonl file")
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED_PATH))
    parser.add_argument("--list", action="store_true", help="List candidate summaries")
    parser.add_argument("--show", help="Show a candidate id in full")
    parser.add_argument("--select", nargs=2, metavar=("SAMPLE_ID", "CANDIDATE_ID"))
    return parser.parse_args()


def print_candidate_summary(candidate: dict) -> None:
    validation = candidate.get("validation") or {}
    score = validation.get("3turn_mean") or validation.get("score") or "unvalidated"
    print(
        f"{candidate['candidate_id']} | sample={candidate['sample_id']} | "
        f"model={candidate.get('source_model')} | score={score}"
    )


def print_candidate(candidate: dict) -> None:
    print_candidate_summary(candidate)
    print("")
    for message in candidate.get("messages", []):
        role = message["role"].upper()
        turn = message["turn"]
        pressure = message.get("pressure_type")
        pressure_suffix = f" [{pressure}]" if pressure else ""
        print(f"--- TURN {turn} {role}{pressure_suffix} ---")
        print(message["content"].strip())
        print("")


def main() -> None:
    args = parse_args()
    candidates = read_candidate_traces(args.candidates)
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    if args.list:
        for candidate in candidates:
            print_candidate_summary(candidate)

    if args.show:
        candidate = by_id.get(args.show)
        if candidate is None:
            raise SystemExit(f"Candidate not found: {args.show}")
        print_candidate(candidate)

    if args.select:
        sample_id, candidate_id = args.select
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise SystemExit(f"Candidate not found: {candidate_id}")
        if str(candidate.get("sample_id")) != sample_id:
            raise SystemExit(
                f"Candidate {candidate_id} belongs to sample {candidate.get('sample_id')}, not {sample_id}"
            )
        errors = validate_trace(candidate)
        if errors:
            raise SystemExit("Cannot select invalid candidate: " + "; ".join(errors))

        selected = load_selected(args.selected)
        selected["selected"][sample_id] = candidate
        write_selected(Path(args.selected), selected)
        print(f"Selected {candidate_id} for sample {sample_id} in {args.selected}")

    if not (args.list or args.show or args.select):
        raise SystemExit("Choose at least one action: --list, --show, or --select")


if __name__ == "__main__":
    main()
