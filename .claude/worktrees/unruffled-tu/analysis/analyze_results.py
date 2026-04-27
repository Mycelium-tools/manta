"""
Parse the latest manta-test5 eval log into a pandas DataFrame using inspect_ai.
"""

import glob
import os
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

# Find the latest manta-test5 log
log_files = sorted(glob.glob("logs/*manta-test5*.eval"))
if not log_files:
    raise FileNotFoundError("No manta-test5 log files found in logs/")

latest = log_files[-1]
print(f"Reading: {os.path.basename(latest)}\n")

log = read_eval_log(latest)
print(f"Status: {log.status} | Samples: {len(log.samples)}\n")

# Build rows
rows = []
for s in log.samples:
    sc = s.scores["ahb_multidimensional_scorer"]
    meta = sc.metadata

    row = {
        "sample_id": s.id,
        "input": s.input if isinstance(s.input, str) else s.input[0].get("content", ""),
        "overall_score": meta["overall_score"],
        "pressure_types": ", ".join(meta.get("pressure_types", [])),
        "overall_explanation": meta["overall_explanation"],
    }

    # Add each of the 13 dimension scores as columns
    for dim_key, dim_data in meta["dimensions"].items():
        row[dim_key] = dim_data["score"]

    rows.append(row)

df = pd.DataFrame(rows)

# ── Summary ──────────────────────────────────────────────────────────────────
dim_cols = list(log.samples[0].scores["ahb_multidimensional_scorer"].metadata["dimensions"].keys())

print("=" * 60)
print("PER-SAMPLE OVERALL SCORES")
print("=" * 60)
print(df[["sample_id", "overall_score", "pressure_types"]].to_string(index=False))

print(f"\n{'=' * 60}")
print("DIMENSION SCORES (per sample, -1 = N/A)")
print("=" * 60)
print(df[["sample_id"] + dim_cols].to_string(index=False))

print(f"\n{'=' * 60}")
print("MEAN SCORES ACROSS 5 SAMPLES (excluding N/A = -1.0)")
print("=" * 60)
# Replace -1.0 (N/A) with NaN so means ignore them
df_na = df[["overall_score"] + dim_cols].replace(-1.0, np.nan)
means = df_na.mean().sort_values()
for col, val in means.items():
    bar = "#" * int(val * 20) if not np.isnan(val) else "N/A"
    val_str = f"{val:.3f}" if not np.isnan(val) else " nan"
    print(f"  {col:<30} {val_str}  {bar}")

print(f"\nOverall mean: {df['overall_score'].mean():.3f}")
print(f"Overall std:  {df['overall_score'].std():.3f}")

print(f"\n{'=' * 60}")
print("OVERALL EXPLANATIONS")
print("=" * 60)
for _, row in df.iterrows():
    print(f"\n[Sample {row['sample_id']}] score={row['overall_score']:.2f}")
    print(f"  {row['overall_explanation']}")
