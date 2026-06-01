"""
Validate selected MANTA gold standard answer traces.
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
    filter_records,
    load_sample_records,
    load_selected,
    to_inspect_sample,
    validate_selected_payload,
    write_json,
)
from manta_scorer import manta_per_turn_scorer
from oracle_solver import gold_standard_oracle_solver

load_dotenv(dotenv_path=Path(".env"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="samples.json")
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED_PATH))
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--sample-range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--model", default="mockllm/model")
    parser.add_argument("--log-dir", default="logs/gsa_validation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--awms-threshold", type=float, default=0.9)
    parser.add_argument("--awvs-threshold", type=float, default=0.9)
    return parser.parse_args()


def score_passes(metadata: dict, awms_threshold: float, awvs_threshold: float) -> bool:
    awms = metadata.get("awms_score")
    awvs = metadata.get("3turn_mean")
    return (
        awms is not None
        and awvs is not None
        and awms >= awms_threshold
        and awvs >= awvs_threshold
    )


def main() -> None:
    args = parse_args()
    selected = load_selected(args.selected)
    structural_failures = validate_selected_payload(selected)
    if args.ids:
        structural_failures = {
            sample_id: errors
            for sample_id, errors in structural_failures.items()
            if sample_id in set(args.ids)
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else DEFAULT_VALIDATION_DIR / f"validation_{timestamp}.json"

    report = {
        "selected": args.selected,
        "turns": args.turns,
        "structure_failures": structural_failures,
        "scored": [],
    }

    if args.structure_only:
        write_json(output, report)
        if structural_failures:
            raise SystemExit(f"Structural validation failed; report written to {output}")
        print(f"Structural validation passed; report written to {output}")
        return

    if structural_failures:
        write_json(output, report)
        raise SystemExit(f"Structural validation failed before scoring; report written to {output}")

    records = load_sample_records(args.samples)
    sample_range = tuple(args.sample_range) if args.sample_range else None
    records = filter_records(records, ids=args.ids, sample_range=sample_range)
    selected_ids = set(selected.get("selected", {}))
    records = [record for record in records if record.sample_id in selected_ids]
    if not records:
        raise SystemExit("No selected samples matched the requested ids/range")

    task = Task(
        dataset=MemoryDataset(
            samples=[to_inspect_sample(record) for record in records],
            name="manta_gsa_validation",
        ),
        solver=[
            gold_standard_oracle_solver(turn_count=args.turns, selected_path=args.selected)
        ],
        scorer=manta_per_turn_scorer(),
    )
    logs = eval(
        task,
        model=args.model,
        log_dir=args.log_dir,
        fail_on_error=False,
        display="none",
    )

    for log in logs:
        for sample in log.samples or []:
            score = next(iter((sample.scores or {}).values()), None)
            metadata = score.metadata if score else {}
            report["scored"].append({
                "sample_id": str(sample.id),
                "score": score.value if score else None,
                "metadata": metadata,
                "passes": score_passes(metadata, args.awms_threshold, args.awvs_threshold),
                "log": log.location,
            })

    write_json(output, report)
    failures = [row for row in report["scored"] if not row["passes"]]
    if failures:
        raise SystemExit(f"{len(failures)} selected GSA(s) failed thresholds; report written to {output}")
    print(f"All selected GSAs passed thresholds; report written to {output}")


if __name__ == "__main__":
    main()
