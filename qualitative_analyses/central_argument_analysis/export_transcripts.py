"""
Export conversation transcripts from a MANTA .eval file to CSV.

Usage:
    python analysis/export_transcripts.py <path/to/file.eval>
    python analysis/export_transcripts.py <path/to/file.eval> --output transcripts.csv

Output format (long): one row per message, columns:
    sample_id, epoch, turn_num, role, content,
    awvs_score, awms_score, pressure_types, awvs_explanation
"""

import argparse
import csv
import os
import sys
from inspect_ai.log import read_eval_log


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def main():
    parser = argparse.ArgumentParser(description="Export MANTA eval transcripts to CSV")
    parser.add_argument("eval_path", help="Path to the .eval file")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (default: <eval_name>_transcripts.csv)")
    args = parser.parse_args()

    eval_path = os.path.abspath(args.eval_path)
    output_path = args.output or eval_path.replace(".eval", "_transcripts.csv")

    print(f"Reading {eval_path} ...")
    log = read_eval_log(eval_path)
    print(f"  {len(log.samples)} samples found")

    rows = []
    for sample in log.samples:
        # Pull scores metadata
        scorer_result = sample.scores.get("manta_scorer")
        if scorer_result and scorer_result.metadata:
            meta = scorer_result.metadata
            awvs = meta.get("awvs_score")
            awms = meta.get("awms_score")
            pressure_types = ", ".join(meta.get("pressure_types") or [])
            dimensions = meta.get("dimensions", {})
            awvs_explanation = dimensions.get("Animal Welfare Value Stability", {}).get("explanation", "")
            awms_explanation = dimensions.get("Animal Welfare Moral Sensitivity", {}).get("explanation", "")
        else:
            awvs = awms = pressure_types = awvs_explanation = awms_explanation = ""

        overall = scorer_result.value if scorer_result else ""

        # Walk messages — filter to user/assistant only (skip system)
        turn_num = 0
        for msg in sample.messages:
            if msg.role not in ("user", "assistant"):
                continue
            if msg.role == "user":
                turn_num += 1
            rows.append({
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "turn_num": turn_num,
                "role": msg.role,
                "content": extract_text(msg.content),
                "overall_score": overall,
                "awvs_score": awvs,
                "awms_score": awms,
                "pressure_types": pressure_types,
                "awvs_explanation": awvs_explanation,
                "awms_explanation": awms_explanation,
            })

    fieldnames = [
        "sample_id", "epoch", "turn_num", "role", "content",
        "overall_score", "awvs_score", "awms_score", "pressure_types",
        "awvs_explanation", "awms_explanation",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {output_path}")


if __name__ == "__main__":
    main()
