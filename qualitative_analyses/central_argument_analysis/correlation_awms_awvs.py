"""
AWMS x AWVS correlation analysis
=================================
Loads Inspect eval logs from a run directory, extracts per-sample
AWMS (Animal Welfare Moral Sensitivity) and AWVS (Animal Welfare Value Stability)
scores, and reports the recognition-vs-resilience correlation that is the
central scientific claim of MANTA.

Outputs:
  - analysis/awms_awvs_per_sample.csv           per (model, sample) row
  - analysis/figures/awms_x_awvs_scatter.png    pooled scatter, 4-quadrant overlay
  - analysis/figures/awms_x_awvs_by_model.png   per-model small-multiples
  - stdout: per-model + pooled Pearson / Spearman with bootstrap 95% CIs

Usage:
  python analysis/correlation_awms_awvs.py
  python analysis/correlation_awms_awvs.py --log-dir logs/Isabella_April2026/run_smoke_2026-04-28_120000/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inspect_ai.log import read_eval_log

try:
    from scipy import stats as _scipy_stats  # type: ignore
except ImportError:
    _scipy_stats = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "qualitative_analyses", "central_argument_analysis")
FIGURES_DIR = os.path.join(ANALYSIS_DIR, "figures")
CSV_PATH = os.path.join(ANALYSIS_DIR, "awms_awvs_per_sample.csv")


def _resolve_default_log_dir() -> str:
    """Pick the most recent run subdir under logs/<MANTA_USER>_<MonthYYYY>/."""
    base_candidates = []
    if os.environ.get("MANTA_LOG_DIR"):
        base_candidates.append(os.environ["MANTA_LOG_DIR"])
    if os.environ.get("MANTA_USER"):
        month_year = datetime.now().strftime("%B%Y")
        base_candidates.append(os.path.join(REPO_ROOT, "logs", f"{os.environ['MANTA_USER']}_{month_year}"))
    base_candidates.append(os.path.join(REPO_ROOT, "logs"))

    for base in base_candidates:
        if not os.path.isdir(base):
            continue
        run_subdirs = sorted(
            [p for p in glob.glob(os.path.join(base, "run_*")) if os.path.isdir(p)],
            key=os.path.getmtime,
        )
        if run_subdirs:
            return run_subdirs[-1]
        # No run_* subdirs — fall back to the base itself if it has .eval files
        if glob.glob(os.path.join(base, "*.eval")):
            return base

    raise FileNotFoundError(
        "Could not auto-resolve a log directory. Pass --log-dir explicitly."
    )


def _extract_rows(log_path: str) -> list[dict]:
    log = read_eval_log(log_path)
    if log.samples is None:
        return []
    model = log.eval.model if log.eval and log.eval.model else os.path.basename(log_path)
    rows = []
    for s in log.samples:
        score_obj = s.scores.get("manta_scorer") if s.scores else None
        if score_obj is None:
            continue
        meta = score_obj.metadata or {}
        awms = meta.get("awms_score")
        awvs = meta.get("awvs_score")
        rows.append({
            "model": model,
            "sample_id": str(s.id),
            "pressure_types": ", ".join(meta.get("pressure_types", []) or []),
            "awms_score": awms,
            "awms_anchor": meta.get("awms_anchor"),
            "awvs_score": awvs,
            "log_file": os.path.basename(log_path),
        })
    return rows


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, statistic, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(statistic(x[idx], y[idx]))
        except Exception:
            continue
    if not samples:
        return (np.nan, np.nan)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def _pearson(x, y) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.pearsonr(x, y).statistic)
    # Numpy fallback
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (handles ties) — matches scipy.stats.rankdata default method."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # Average ranks for ties
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # mean of ranks i+1..j+1
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x, y) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.spearmanr(x, y).statistic)
    rx = _rankdata(np.asarray(x, dtype=float))
    ry = _rankdata(np.asarray(y, dtype=float))
    return float(np.corrcoef(rx, ry)[0, 1])


def _pvalue_pearson(x, y, n: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.pearsonr(x, y).pvalue)
    # Two-sided t-test approximation: t = r * sqrt((n-2)/(1-r^2))
    if n < 3:
        return float("nan")
    r = _pearson(x, y)
    if abs(r) >= 1.0:
        return 0.0
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # Survival function of t with df = n-2; without scipy use a normal approximation (rough for small n)
    z = abs(t)
    # Standard-normal two-sided tail; flag in printout that this is an approximation
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(z / np.sqrt(2))))
    return float(p)


def _pvalue_spearman(x, y, n: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.spearmanr(x, y).pvalue)
    rx = _rankdata(np.asarray(x, dtype=float))
    ry = _rankdata(np.asarray(y, dtype=float))
    return _pvalue_pearson(rx, ry, n)


def _interpret(r: float) -> str:
    a = abs(r)
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "null"


def _scatter_pooled(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(42)
    for model, sub in df.groupby("model"):
        jitter = rng.normal(0, 0.02, size=len(sub))
        ax.scatter(sub["awms_score"] + jitter, sub["awvs_score"], label=model, alpha=0.7, s=50)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("AWMS — Animal Welfare Moral Sensitivity (Turn 1)")
    ax.set_ylabel("AWVS — Animal Welfare Value Stability (Turns 3-5)")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("AWMS x AWVS — recognition vs. resilience")

    # Quadrant counts
    q_tr = ((df["awms_score"] >= 0.5) & (df["awvs_score"] >= 0.5)).sum()
    q_tl = ((df["awms_score"] < 0.5) & (df["awvs_score"] >= 0.5)).sum()
    q_br = ((df["awms_score"] >= 0.5) & (df["awvs_score"] < 0.5)).sum()
    q_bl = ((df["awms_score"] < 0.5) & (df["awvs_score"] < 0.5)).sum()
    ax.text(0.85, 0.95, f"aware + robust\nn={q_tr}", ha="center", fontsize=9, color="darkgreen")
    ax.text(0.15, 0.95, f"misses cues / defends\nn={q_tl}", ha="center", fontsize=9, color="darkblue")
    ax.text(0.85, 0.05, f"aware but soft\nn={q_br}", ha="center", fontsize=9, color="darkorange")
    ax.text(0.15, 0.05, f"misses cues / folds\nn={q_bl}", ha="center", fontsize=9, color="darkred")

    ax.legend(loc="center right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _scatter_by_model(df: pd.DataFrame, out_path: str) -> None:
    models = sorted(df["model"].unique())
    n = len(models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    rng = np.random.default_rng(42)
    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = df[df["model"] == model]
        jitter = rng.normal(0, 0.02, size=len(sub))
        ax.scatter(sub["awms_score"] + jitter, sub["awvs_score"], alpha=0.7, s=50)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("AWMS")
        ax.set_ylabel("AWVS")
        if len(sub) >= 3:
            r = _pearson(sub["awms_score"].values, sub["awvs_score"].values)
            ax.set_title(f"{model}\nn={len(sub)}, r={r:.2f}")
        else:
            ax.set_title(f"{model}\nn={len(sub)} (too few)")
    # Hide unused panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AWMS x AWVS correlation analysis")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory containing .eval files. Defaults to the most recent run_* under logs/<MANTA_USER>_<Month><Year>/.",
    )
    parser.add_argument(
        "--csv-out",
        default=CSV_PATH,
        help=f"Path for per-sample CSV (default: {CSV_PATH})",
    )
    args = parser.parse_args(argv)

    log_dir = args.log_dir or _resolve_default_log_dir()
    print(f"Reading logs from: {log_dir}")

    eval_files = sorted(glob.glob(os.path.join(log_dir, "*.eval")))
    if not eval_files:
        print(f"No .eval files found in {log_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    skipped_files: list[str] = []
    for path in eval_files:
        try:
            file_rows = _extract_rows(path)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
            skipped_files.append(path)
            continue
        rows.extend(file_rows)

    if not rows:
        print("No samples extracted from any log.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    n_total = len(df)
    df_valid = df.dropna(subset=["awms_score", "awvs_score"])
    df_valid = df_valid[(df_valid["awms_score"] >= 0) & (df_valid["awvs_score"] >= 0)]
    n_dropped = n_total - len(df_valid)
    if n_dropped:
        print(f"Dropped {n_dropped} samples missing AWMS or AWVS (left: {len(df_valid)}).")

    # Write CSV (full df, including dropped rows for transparency)
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote per-sample CSV: {args.csv_out}")

    if df_valid.empty:
        print("No valid (AWMS, AWVS) pairs to correlate.")
        return 0

    # Per-model stats
    print()
    print("=" * 78)
    print(f"{'model':<48} {'n':>4} {'mAWMS':>6} {'mAWVS':>6} {'r':>6} {'rho':>6}")
    print("=" * 78)
    for model, sub in df_valid.groupby("model"):
        x = sub["awms_score"].to_numpy(dtype=float)
        y = sub["awvs_score"].to_numpy(dtype=float)
        n = len(x)
        mAWMS = x.mean()
        mAWVS = y.mean()
        if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r = _pearson(x, y)
            rho = _spearman(x, y)
        else:
            r = float("nan")
            rho = float("nan")
        print(f"{model:<48} {n:>4} {mAWMS:>6.3f} {mAWVS:>6.3f} {r:>6.2f} {rho:>6.2f}")

    # Pooled stats with bootstrap CIs
    x = df_valid["awms_score"].to_numpy(dtype=float)
    y = df_valid["awvs_score"].to_numpy(dtype=float)
    n = len(x)
    pooled_r = _pearson(x, y) if (n >= 3 and np.std(x) > 0 and np.std(y) > 0) else float("nan")
    pooled_rho = _spearman(x, y) if (n >= 3 and np.std(x) > 0 and np.std(y) > 0) else float("nan")
    p_r = _pvalue_pearson(x, y, n) if not np.isnan(pooled_r) else float("nan")
    p_rho = _pvalue_spearman(x, y, n) if not np.isnan(pooled_rho) else float("nan")
    r_ci = _bootstrap_ci(x, y, _pearson) if not np.isnan(pooled_r) else (np.nan, np.nan)
    rho_ci = _bootstrap_ci(x, y, _spearman) if not np.isnan(pooled_rho) else (np.nan, np.nan)

    print()
    print("Pooled across models:")
    print(f"  Pearson  r   = {pooled_r:.3f}  (p = {p_r:.4f})  95% CI [{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
    print(f"  Spearman rho = {pooled_rho:.3f}  (p = {p_rho:.4f})  95% CI [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")

    # Plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pooled_path = os.path.join(FIGURES_DIR, "awms_x_awvs_scatter.png")
    by_model_path = os.path.join(FIGURES_DIR, "awms_x_awvs_by_model.png")
    _scatter_pooled(df_valid, pooled_path)
    _scatter_by_model(df_valid, by_model_path)
    print(f"\nWrote pooled scatter:    {pooled_path}")
    print(f"Wrote per-model scatter: {by_model_path}")

    # One-line interpretation
    sig = _interpret(pooled_r)
    print()
    print(f"Pearson r = {pooled_r:.3f} (p = {p_r:.4f}), n = {n}. "
          f"Recognition vs resilience signal: {sig}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
