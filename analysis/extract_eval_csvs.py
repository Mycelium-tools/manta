"""
Extract per-sample scores + all metadata from .eval files into CSVs.
One CSV per .eval file, written alongside the .eval file.

Usage:
    python analysis/extract_eval_csvs.py                    # process all run dirs below
    python analysis/extract_eval_csvs.py --run-dir logs/Allen_May2026/sample_range_0_500_...
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

RUN_DIRS = [
    "logs/Allen_May2026/sample_range_0_500_per_turn_big3_2026-05-02_174537",
    "logs/Allen_May2026/sample_range_501_989_per_turn_big3_2026-05-02_215749",
    "logs/Allen_May2026/sample_range_0_500_per_turn_other4_2026-05-03_020440",
    "logs/Allen_May2026/sample_range_500_990_grokv2_2026-05-03_125850",
    "logs/Allen_May2026/sample_range_0_500_per_turn_deepMistralLlama_2026-05-03_161028",
    "logs/Allen_May2026/sample_range_500_990_per_turn_deepMistralLlama_2026-05-03_195936",
]


def _j(val):
    """Serialize a value to JSON string, or return None if falsy."""
    if val is None:
        return None
    return json.dumps(val, default=str)


def extract_sample_row(sample, log) -> dict:
    scorer = sample.scores.get("manta_per_turn_scorer") if sample.scores else None
    sm = scorer.metadata if scorer else {}
    pts = sm.get("per_turn_scores", {}) if sm else {}
    pte = sm.get("per_turn_explanations", {}) if sm else {}
    meta = sample.metadata or {}

    out = sample.output
    usage = out.usage if out else None
    stop_reason = None
    if out and out.choices:
        stop_reason = out.choices[-1].stop_reason if out.choices[-1].stop_reason else None

    # log-level fields (same across all samples in this file)
    row = {
        "log_file": None,           # filled in by caller
        "run_dir": None,            # filled in by caller
        "eval_id": log.eval.eval_id,
        "run_id": log.eval.run_id,
        "eval_created": log.eval.created,
        "task": log.eval.task,
        "model": log.eval.model,
        "dataset_name": log.eval.dataset.name if log.eval.dataset else None,
        "dataset_samples": log.eval.dataset.samples if log.eval.dataset else None,
        "epochs": log.eval.config.epochs if log.eval.config else None,
        "git_commit": log.eval.revision.commit if log.eval.revision else None,
        "git_dirty": log.eval.revision.dirty if log.eval.revision else None,

        # sample identity
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "uuid": sample.uuid,

        # input / target
        "input": sample.input if isinstance(sample.input, str) else _j(sample.input),
        "target": sample.target if isinstance(sample.target, str) else _j(sample.target),
        "choices": _j(sample.choices),

        # scores — top-level
        "overall_score": scorer.value if scorer else None,
        "score_explanation": scorer.explanation if scorer else None,

        # per-turn scores
        "turn1_score": pts.get("1"),
        "turn2_score": pts.get("2"),
        "turn3_score": pts.get("3"),
        "turn4_score": pts.get("4"),
        "turn5_score": pts.get("5"),

        # per-turn explanations
        "turn1_explanation": pte.get("1"),
        "turn2_explanation": pte.get("2"),
        "turn3_explanation": pte.get("3"),
        "turn4_explanation": pte.get("4"),
        "turn5_explanation": pte.get("5"),

        # aggregate stats from scorer metadata
        "3turn_mean": sm.get("3turn_mean"),
        "5turn_mean": sm.get("5turn_mean"),
        "3turn_slope": sm.get("3turn_slope"),
        "3turn_variance": sm.get("3turn_variance"),
        "5turn_slope": sm.get("5turn_slope"),
        "5turn_variance": sm.get("5turn_variance"),
        "pressure_types": _j(sm.get("pressure_types")),
        "per_turn_judge": sm.get("per_turn_judge"),

        # AWMS
        "awms_score": sm.get("awms_score"),
        "awms_explanation": sm.get("awms_explanation"),
        "awms_judge": sm.get("awms_judge"),

        # sample metadata fields
        "tags": _j(meta.get("tags")),
        "language": meta.get("language"),
        "pressure": _j(meta.get("pressure")),
        "reference_answer": meta.get("reference_answer"),
        "pressure_types_meta": _j(meta.get("pressure_types")),
        "dynamic_followups": meta.get("dynamic_followups"),
        "anchor_turn": meta.get("anchor_turn"),

        # output
        "output_model": out.model if out else None,
        "output_completion": out.completion if out else None,
        "output_stop_reason": stop_reason,
        "output_time": out.time if out else None,
        "output_error": str(out.error) if out and out.error else None,
        "output_input_tokens": usage.input_tokens if usage else None,
        "output_output_tokens": usage.output_tokens if usage else None,
        "output_total_tokens": usage.total_tokens if usage else None,
        "output_cache_write_tokens": usage.input_tokens_cache_write if usage else None,
        "output_cache_read_tokens": usage.input_tokens_cache_read if usage else None,
        "output_reasoning_tokens": usage.reasoning_tokens if usage else None,

        # full model usage breakdown (JSON dict keyed by model name)
        "model_usage": _j(
            {k: v.model_dump() for k, v in sample.model_usage.items()}
            if sample.model_usage else {}
        ),

        # full conversation messages (JSON array)
        "messages": _j(
            [m.model_dump() for m in sample.messages]
            if sample.messages else []
        ),

        # timing
        "started_at": sample.started_at,
        "completed_at": sample.completed_at,
        "total_time": sample.total_time,
        "working_time": sample.working_time,

        # errors
        "error": str(sample.error) if sample.error else None,
        "error_retries": len(sample.error_retries) if sample.error_retries else 0,
    }
    return row


def process_eval_file(eval_path: Path) -> int:
    print(f"  Reading {eval_path.name} ...", end=" ", flush=True)
    log = read_eval_log(str(eval_path))
    if not log.samples:
        print("no samples, skipping")
        return 0

    rows = []
    for s in log.samples:
        row = extract_sample_row(s, log)
        row["log_file"] = eval_path.name
        row["run_dir"] = eval_path.parent.name
        rows.append(row)

    out_path = eval_path.with_suffix(".csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} rows → {out_path.name}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Single run directory to process (overrides built-in list)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent  # MANTA/

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = [base / rd for rd in RUN_DIRS]

    total = 0
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[SKIP] {run_dir} not found")
            continue
        eval_files = sorted(run_dir.glob("*.eval"))
        if not eval_files:
            print(f"[SKIP] no .eval files in {run_dir}")
            continue
        print(f"\n{run_dir.name}/")
        for ef in eval_files:
            total += process_eval_file(ef)

    print(f"\nDone. {total} total rows written.")


if __name__ == "__main__":
    main()
