"""
MANTA Central Argument Evidence
================================
Generates all charts and statistics needed to support the three claims:

  H1  — Multi-turn surfaces failure modes single-turn benchmarks miss
  H2  — Species × pressure-type interaction matrix (novel contribution)
  H3  — AWMS → AWVS correlation (moral-recognition measures something stable)

Data sources
  manta_scorer runs  : AWMS + AWVS per sample (bulk end-of-conversation scorer)
  manta_per_turn_scorer runs : per-turn scores keyed '1'–'5', enabling pressure attribution

Outputs (all written to analysis/figures/):
  h1a_awms_vs_awvs_bars.png      — mean AWMS vs AWVS per model (bulk scorer)
  h1b_score_trajectory.png       — mean score by turn per model (T1–T5)
  h1c_aware_but_soft.png         — quadrant breakdown per model (bulk scorer)
  h1d_5turn_3turn_bars.png       — 5-turn mean vs pressured-turn mean per model
  h2_species_pressure_heatmap.png — species × pressure interaction
  h2b_pressure_type_bars.png     — mean score by pressure type
  h3a_awms_awvs_scatter.png      — pooled AWMS vs AWVS scatter
  h3b_correlation_by_model.png   — per-model Pearson r bar chart
  summary.txt                    — key statistics

Usage:
  uv run python analysis/central_argument_evidence.py
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from inspect_ai.log import read_eval_log

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(REPO, "qualitative_analyses", "central_argument_analysis", "figures")
os.makedirs(FIGS, exist_ok=True)

# Production run directories (manta_scorer — AWMS + AWVS)
MANTA_SCORER_DIRS = [
    "logs/Allen_April2026/sample_range_0_240_2026-04-30_205848",
    "logs/Allen_May2026/sample_range_241_500_2026-05-01_143210",
    "logs/Isabella_May2026/sample_range_501_751_2026-05-01_163900",
    "logs/Allen_May2026/sample_range_752_989_2026-05-01_202620",
]

# Per-turn run directories (manta_per_turn_scorer — per-turn scores)
# Full 7-model coverage across samples 0–1090
PER_TURN_DIRS = [
    # Opus, GPT-5.5, Gemini — samples 0–500 and 501–989
    "logs/Allen_May2026/sample_range_0_500_per_turn_big3_2026-05-02_174537",
    "logs/Allen_May2026/sample_range_501_989_per_turn_big3_2026-05-02_215749",
    # Grok — samples 0–500 and 500–990
    "logs/Allen_May2026/sample_range_0_500_per_turn_other4_2026-05-03_020440",
    "logs/Allen_May2026/sample_range_500_990_grokv2_2026-05-03_125850",
    # DeepSeek, Mistral, LLaMA — samples 0–500 and 500–990
    "logs/Allen_May2026/sample_range_0_500_per_turn_deepMistralLlama_2026-05-03_161028",
    "logs/Allen_May2026/sample_range_500_990_per_turn_deepMistralLlama_2026-05-03_195936",
    # All 7 models — samples 990–1090 (new questions)
    "logs/Allen_May2026/sample_range_990_1090_per_turn_all_models_2026-05-04_135307",
]

MODEL_LABELS = {
    "anthropic/claude-opus-4-7":                       "Claude Opus",
    "openrouter/anthropic/claude-opus-4-7":            "Claude Opus",   # openrouter alias
    "anthropic/claude-sonnet-4-6":                     "Claude Sonnet",
    "openai/gpt-5.5":                                  "GPT-5.5",
    "openrouter/openai/gpt-5.5":                       "GPT-5.5",       # openrouter alias
    "grok/grok-4.3":                                   "Grok 4.3",
    "google/gemini-3.1-flash-lite-preview":            "Gemini Flash",
    "openai-api/deepseek/deepseek-v4-flash":           "DeepSeek",
    "mistral/mistral-small-2603":                      "Mistral Small",
    "openrouter/meta-llama/llama-3.3-70b-instruct":   "LLaMA 3.3",
}

PRESSURE_ORDER = ["economic", "social", "pragmatic", "epistemic", "cultural"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def short_model(m: str) -> str:
    return MODEL_LABELS.get(m, m.split("/")[-1])


def extract_species(sample_id: str) -> str | None:
    """Return the animal part of e.g. '101_fish' → 'fish', else None."""
    s = str(sample_id)
    parts = s.split("_", 1)
    if len(parts) == 2 and not parts[1].isdigit():
        return parts[1].lower()
    return None


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i: j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_manta_scorer(dirs: list[str]) -> pd.DataFrame:
    """Load AWMS + AWVS from manta_scorer runs."""
    rows = []
    for d in dirs:
        full_d = os.path.join(REPO, d)
        for path in sorted(glob.glob(os.path.join(full_d, "*.eval"))):
            try:
                log = read_eval_log(path)
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
                continue
            if not log.samples:
                continue
            model = log.eval.model if log.eval else os.path.basename(path)
            for s in log.samples:
                sc = s.scores.get("manta_scorer") if s.scores else None
                if sc is None:
                    continue
                meta = sc.metadata or {}
                awms = meta.get("awms_score")
                awvs = meta.get("awvs_score")
                if awms is None or awvs is None:
                    continue
                species = extract_species(s.id)
                rows.append({
                    "model": model,
                    "label": short_model(model),
                    "sample_id": str(s.id),
                    "species": species,
                    "awms": float(awms),
                    "awvs": float(awvs),
                    "pressure_types": meta.get("pressure_types") or [],
                    "fold_detected": bool(meta.get("fold_detected", False)),
                    "fold_turn": meta.get("fold_turn"),
                })
    df = pd.DataFrame(rows)
    print(f"[manta_scorer] Loaded {len(df)} samples across {df['model'].nunique()} models")
    return df


def load_per_turn(dirs: list[str]) -> pd.DataFrame:
    """Load per-turn scores from manta_per_turn_scorer runs."""
    rows = []
    for d in dirs:
        full_d = os.path.join(REPO, d)
        for path in sorted(glob.glob(os.path.join(full_d, "*.eval"))):
            try:
                log = read_eval_log(path)
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
                continue
            if not log.samples:
                continue
            model = log.eval.model if log.eval else os.path.basename(path)
            for s in log.samples:
                sc = s.scores.get("manta_per_turn_scorer") if s.scores else None
                if sc is None:
                    continue
                meta = sc.metadata or {}
                pts = meta.get("per_turn_scores") or {}
                pressure_types = meta.get("pressure_types") or []
                awms = meta.get("awms_score")
                species = extract_species(s.id)
                row = {
                    "model": model,
                    "label": short_model(model),
                    "sample_id": str(s.id),
                    "species": species,
                    "awms": float(awms) if awms is not None else None,
                    "pressure_types": pressure_types,
                    "3turn_mean": meta.get("3turn_mean"),
                    "5turn_mean": meta.get("5turn_mean"),
                    "3turn_slope": meta.get("3turn_slope"),
                    "5turn_slope": meta.get("5turn_slope"),
                    "3turn_variance": meta.get("3turn_variance"),
                    "5turn_variance": meta.get("5turn_variance"),
                }
                for t in range(1, 6):
                    row[f"t{t}"] = float(pts.get(str(t), float("nan")))
                rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[per_turn] Loaded {len(df)} samples across {df['label'].nunique()} models")
    return df


# ── Chart functions ────────────────────────────────────────────────────────────

def plot_h1a_awms_vs_awvs_bars(pt_df: pd.DataFrame) -> None:
    """H1a: Grouped bar — mean AWMS vs mean AWVS (3-turn mean, T3–5) per model."""
    stats = pt_df.groupby("label").agg(
        awms_mean=("awms", "mean"),
        awms_sem=("awms", lambda x: x.sem()),
        awvs_mean=("3turn_mean", "mean"),
        awvs_sem=("3turn_mean", lambda x: x.sem()),
    ).reset_index()
    stats = stats.sort_values("awvs_mean")

    n = len(stats)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w / 2, stats["awms_mean"], w,
                   yerr=stats["awms_sem"], capsize=4,
                   label="AWMS (Turn 1 only)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w / 2, stats["awvs_mean"], w,
                   yerr=stats["awvs_sem"], capsize=4,
                   label="AWVS (Turns 3–5 under pressure)", color="#F44336", alpha=0.85)

    # Gap arrows / annotations
    for i, row in stats.reset_index().iterrows():
        gap = row["awms_mean"] - row["awvs_mean"]
        if gap > 0.05:
            ax.annotate(
                f"−{gap:.2f}",
                xy=(i, row["awvs_mean"] + 0.01),
                ha="center", va="bottom", fontsize=7.5, color="#555"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "H1 — Single-turn vs Multi-turn: AWMS (Turn 1) vs AWVS (Turns 3–5)\n"
        "Gap = welfare reasoning that survives only until pressure is applied",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIGS, "h1a_awms_vs_awvs_bars.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h1b_score_trajectory(pt_df: pd.DataFrame) -> None:
    """H1b: Mean score by turn (1–5), pooled and per model."""
    turn_cols = [f"t{i}" for i in range(1, 6)]
    turn_labels = ["T1\n(implicit)", "T2\n(anchor)", "T3\n(pressure 1)", "T4\n(pressure 2)", "T5\n(pressure 3)"]

    # Per model mean at each turn
    models = sorted(pt_df["label"].unique())
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per model with ±1 SEM shading
    for model, color in zip(models, palette):
        sub = pt_df[pt_df["label"] == model]
        means = np.array([sub[c].dropna().mean() for c in turn_cols])
        sems = np.array([
            sub[c].dropna().std() / np.sqrt(sub[c].dropna().count()) if sub[c].dropna().count() > 1 else 0.0
            for c in turn_cols
        ])
        xs = np.array(range(1, 6))
        ax1.fill_between(xs, means - sems, means + sems, alpha=0.10, color=color)
        ax1.plot(xs, means, marker="o", label=model, color=color, linewidth=1.8)

    ax1.axvspan(2.5, 5.5, alpha=0.07, color="red", label="Pressure zone")
    ax1.set_xticks(range(1, 6))
    ax1.set_xticklabels(turn_labels, fontsize=8)
    ax1.set_ylabel("Mean Per-Turn Score (0–1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Score Trajectory — Per Model", fontsize=10)
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)

    # Right: pooled with shaded CI
    pooled_means = []
    pooled_lo = []
    pooled_hi = []
    for c in turn_cols:
        vals = pt_df[c].dropna().values
        m = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        pooled_means.append(m)
        pooled_lo.append(m - 1.96 * se)
        pooled_hi.append(m + 1.96 * se)

    xs = np.array(range(1, 6))
    ax2.fill_between(xs, pooled_lo, pooled_hi, alpha=0.2, color="#1976D2")
    ax2.plot(xs, pooled_means, marker="o", color="#1976D2", linewidth=2.5, label="Pooled mean ± 95% CI")
    ax2.axvspan(2.5, 5.5, alpha=0.07, color="red")
    ax2.set_xticks(range(1, 6))
    ax2.set_xticklabels(turn_labels, fontsize=8)
    ax2.set_ylabel("Mean Per-Turn Score (0–1)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Score Trajectory — Pooled Across All Models", fontsize=10)
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "H1 — Score Degradation Under Multi-Turn Adversarial Pressure\n"
        "Turn 1 = single-turn proxy; Turns 3–5 = multi-turn pressure zone",
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(FIGS, "h1b_score_trajectory.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h1c_aware_but_soft(pt_df: pd.DataFrame) -> None:
    """H1c: Quadrant breakdown — what single-turn misses."""
    def quadrant(awms, awvs):
        hi_awms = awms >= 0.5
        hi_awvs = awvs >= 0.5
        if hi_awms and hi_awvs:
            return "aware + robust"
        if hi_awms and not hi_awvs:
            return "aware but soft\n(single-turn miss)"
        if not hi_awms and hi_awvs:
            return "misses cues / defends"
        return "misses cues / folds"

    df = pt_df.copy()
    df["quadrant"] = df.apply(lambda r: quadrant(r["awms"], r["3turn_mean"]), axis=1)
    order = ["aware + robust", "aware but soft\n(single-turn miss)", "misses cues / defends", "misses cues / folds"]
    colors = {"aware + robust": "#4CAF50", "aware but soft\n(single-turn miss)": "#FF9800",
              "misses cues / defends": "#2196F3", "misses cues / folds": "#F44336"}

    # Per-model stacked bar (proportion)
    models = df.groupby("label")["3turn_mean"].mean().sort_values().index.tolist()
    data = {}
    for q in order:
        vals = []
        for m in models:
            sub = df[df["label"] == m]
            vals.append((sub["quadrant"] == q).sum() / len(sub) * 100)
        data[q] = vals

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(models))
    for q in order:
        vals = np.array(data[q])
        ax.bar(models, vals, bottom=bottoms, label=q, color=colors[q], alpha=0.88)
        # Label if > 8%
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 8:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    ax.set_ylabel("% of Samples")
    ax.set_ylim(0, 102)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_title(
        'H1 — "Aware But Soft": Samples Single-Turn Would Pass But Multi-Turn Fails\n'
        'Orange = welfare-aware on T1 but folds under pressure (single-turn false negative)',
        fontsize=10
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(FIGS, "h1c_aware_but_soft.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h2_species_pressure_heatmap(pt_df: pd.DataFrame) -> None:
    """H2: Species × pressure interaction matrix using per-turn attribution."""
    # For each sample with animal variation:
    #   pressure_types[0] → turn 3 score (t3)
    #   pressure_types[1] → turn 4 score (t4)
    #   pressure_types[2] → turn 5 score (t5)
    species_df = pt_df[pt_df["species"].notna()].copy()
    if species_df.empty:
        print("  No animal-varied samples in per_turn data — skipping H2 heatmap")
        return

    records = []
    for _, row in species_df.iterrows():
        pt = row["pressure_types"]
        if not isinstance(pt, list) or len(pt) < 3:
            continue
        sp = row["species"]
        for turn_offset, pressure in enumerate(pt):
            turn = turn_offset + 3  # turns 3, 4, 5
            score = row.get(f"t{turn}")
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                records.append({"species": sp, "pressure": pressure.lower(), "score": float(score)})

    if not records:
        print("  Could not extract species×pressure records")
        return

    heat_df = pd.DataFrame(records)
    pivot = heat_df.groupby(["species", "pressure"])["score"].mean().unstack(fill_value=np.nan)

    # Only keep pressures in our standard order that appear in the data
    col_order = [p for p in PRESSURE_ORDER if p in pivot.columns]
    pivot = pivot[col_order]

    # Filter species with at least 5 data points total
    counts = heat_df.groupby("species")["score"].count()
    pivot = pivot.loc[counts[counts >= 5].index]

    # Sort species by overall mean score (ascending — most vulnerable first)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(max(8, len(col_order) * 1.4), max(6, len(pivot) * 0.35 + 2)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([p.capitalize() for p in col_order], fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.str.capitalize(), fontsize=8)
    ax.set_xlabel("Pressure Type Applied", fontsize=11)
    ax.set_ylabel("Animal Species", fontsize=11)
    ax.set_title(
        "H2 — Species × Pressure Interaction Matrix\n"
        "Mean per-turn welfare score when that pressure type was applied to questions about this animal\n"
        "(pooled across all models; red = low welfare robustness, green = high)",
        fontsize=10
    )

    # Annotate cells
    for i in range(len(pivot)):
        for j, col in enumerate(col_order):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Mean per-turn welfare score (0–1)", shrink=0.7)
    plt.tight_layout()
    out = os.path.join(FIGS, "h2_species_pressure_heatmap.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h3a_awms_awvs_scatter(pt_df: pd.DataFrame) -> None:
    """H3a: Pooled scatter AWMS vs AWVS (5-turn mean) with quadrant annotations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    models = sorted(pt_df["label"].unique())
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
    rng = np.random.default_rng(42)

    for model, color in zip(models, palette):
        sub = pt_df[pt_df["label"] == model]
        jitter = rng.normal(0, 0.015, size=len(sub))
        ax.scatter(sub["awms"] + jitter, sub["5turn_mean"], label=model, alpha=0.55, s=22, color=color)

    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("AWMS — Animal Welfare Moral Sensitivity (Turn 1)", fontsize=10)
    ax.set_ylabel("AWVS — Animal Welfare Value Stability (5-turn mean, T1–T5)", fontsize=10)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    q_tr = ((pt_df["awms"] >= 0.5) & (pt_df["5turn_mean"] >= 0.5)).sum()
    q_tl = ((pt_df["awms"] < 0.5) & (pt_df["5turn_mean"] >= 0.5)).sum()
    q_br = ((pt_df["awms"] >= 0.5) & (pt_df["5turn_mean"] < 0.5)).sum()
    q_bl = ((pt_df["awms"] < 0.5) & (pt_df["5turn_mean"] < 0.5)).sum()
    n = len(pt_df)
    ax.text(0.82, 0.96, f"aware+robust\n{q_tr/n*100:.0f}%", ha="center", fontsize=8.5, color="darkgreen",
            transform=ax.transAxes)
    ax.text(0.18, 0.96, f"misses/defends\n{q_tl/n*100:.0f}%", ha="center", fontsize=8.5, color="#1565C0",
            transform=ax.transAxes)
    ax.text(0.82, 0.04, f"aware-but-soft\n{q_br/n*100:.0f}%", ha="center", fontsize=8.5, color="darkorange",
            transform=ax.transAxes)
    ax.text(0.18, 0.04, f"misses/folds\n{q_bl/n*100:.0f}%", ha="center", fontsize=8.5, color="darkred",
            transform=ax.transAxes)

    # Pooled r
    valid = pt_df.dropna(subset=["awms", "5turn_mean"])
    x_all = valid["awms"].values
    y_all = valid["5turn_mean"].values
    r = _pearson(x_all, y_all)
    rho = _spearman(x_all, y_all)
    ax.set_title(
        f"H3 — AWMS × AWVS: Does moral recognition predict resilience?\n"
        f"Pooled Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}  (n = {n:,}, 7 models, per-turn scorer)",
        fontsize=10
    )
    ax.legend(loc="center right", fontsize=7.5, framealpha=0.85, markerscale=1.5)
    plt.tight_layout()
    out = os.path.join(FIGS, "h3a_awms_awvs_scatter.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h3b_correlation_by_model(pt_df: pd.DataFrame) -> None:
    """H3b: Per-model Pearson r (AWMS → per-turn AWVS 5-turn mean) bar chart."""
    stats = []
    for label, sub in pt_df.groupby("label"):
        sub_v = sub.dropna(subset=["awms", "5turn_mean"])
        x = sub_v["awms"].values
        y = sub_v["5turn_mean"].values
        r = _pearson(x, y)
        stats.append({"label": label, "r": r, "n": len(sub_v)})
    stats_df = pd.DataFrame(stats).sort_values("r")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors_bar = ["#F44336" if r < 0 else "#4CAF50" if r >= 0.3 else "#FFC107" for r in stats_df["r"]]
    bars = ax.barh(stats_df["label"], stats_df["r"], color=colors_bar, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3, color="grey", linestyle="--", linewidth=0.8, label="r = 0.3 (moderate)")

    for bar, (_, row) in zip(bars, stats_df.iterrows()):
        x_pos = row["r"] + (0.01 if row["r"] >= 0 else -0.01)
        ha = "left" if row["r"] >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'r={row["r"]:.2f}, n={row["n"]:,}', va="center", ha=ha, fontsize=8)

    ax.set_xlabel("Pearson r (AWMS → per-turn AWVS, 5-turn mean)", fontsize=10)
    ax.set_title(
        "H3 — Does Recognising Welfare (AWMS) Predict Defending It (AWVS)?\n"
        "Per-Model Pearson r — per-turn scorer, 7 models, 5-turn mean",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-0.6, 0.8)
    plt.tight_layout()
    out = os.path.join(FIGS, "h3b_correlation_by_model.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h1d_5turn_3turn_bars(pt_df: pd.DataFrame) -> None:
    """H1d: Grouped bars — 5turn_mean vs 3turn_mean (pressured turns) per model."""
    sub = pt_df.dropna(subset=["5turn_mean", "3turn_mean"])
    stats = sub.groupby("label").agg(
        mean_5turn=("5turn_mean", "mean"),
        sem_5turn=("5turn_mean", lambda x: x.sem()),
        mean_3turn=("3turn_mean", "mean"),
        sem_3turn=("3turn_mean", lambda x: x.sem()),
    ).reset_index()
    stats = stats.sort_values("mean_3turn", ascending=False)

    n = len(stats)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, stats["mean_5turn"], w,
           yerr=stats["sem_5turn"], capsize=4,
           label="5-turn mean (all turns T1–T5)", color="#5C6BC0", alpha=0.85)
    ax.bar(x + w / 2, stats["mean_3turn"], w,
           yerr=stats["sem_3turn"], capsize=4,
           label="3-turn mean (pressured turns T3–T5 only)", color="#EF5350", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "H1 — Full Conversation (5-turn) vs Pressure Phase Only (T3–T5) per Model\n"
        "Sorted by pressure-phase score (right bars); lower = more vulnerable under adversarial pressure",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIGS, "h1d_5turn_3turn_bars.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h2b_pressure_type_bars(pt_df: pd.DataFrame) -> None:
    """H2b: Mean per-turn score by pressure type (T3–T5) with error bars."""
    records = []
    for _, row in pt_df.iterrows():
        pt = row["pressure_types"]
        if not isinstance(pt, list) or len(pt) < 3:
            continue
        for i, pressure in enumerate(pt):
            turn = i + 3
            score = row.get(f"t{turn}")
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                records.append({"pressure": pressure.lower(), "score": float(score)})

    if not records:
        print("  No pressure records — skipping H2b")
        return

    press_df = pd.DataFrame(records)
    stats = press_df.groupby("pressure")["score"].agg(
        mean="mean",
        sem=lambda x: x.sem(),
        count="count",
    ).reset_index()

    order = [p for p in PRESSURE_ORDER if p in stats["pressure"].values]
    stats = stats.set_index("pressure").loc[order].reset_index()

    pooled_mean = press_df["score"].mean()

    color_map = {
        "economic": "#EF5350",
        "social": "#AB47BC",
        "pragmatic": "#FF7043",
        "epistemic": "#42A5F5",
        "cultural": "#66BB6A",
    }
    bar_colors = [color_map.get(p, "#90A4AE") for p in stats["pressure"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stats["pressure"].str.capitalize(), stats["mean"],
           yerr=stats["sem"], capsize=5,
           color=bar_colors, alpha=0.85)
    ax.axhline(pooled_mean, color="grey", linestyle="--", linewidth=1,
               label=f"Pooled mean = {pooled_mean:.3f}")

    for i, row in stats.iterrows():
        ax.text(i, row["mean"] + row["sem"] + 0.015,
                f'n={int(row["count"]):,}', ha="center", fontsize=8, color="#555")

    ax.set_ylabel("Mean Per-Turn Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "H2 — Which Pressure Types Best Erode Welfare Reasoning?\n"
        "Mean score at T3–T5 when each pressure type was applied (lower = more effective pressure)",
        fontsize=10
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, "h2b_pressure_type_bars.png")
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ── Summary stats ──────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, pt_df: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("MANTA CENTRAL ARGUMENT — KEY STATISTICS")
    lines.append("=" * 72)

    lines.append(f"\nBulk scorer dataset: {df['sample_id'].nunique()} unique sample-IDs, "
                 f"{df['label'].nunique()} models, {len(df):,} (model, sample) pairs")
    lines.append(f"Per-turn dataset: {pt_df['sample_id'].nunique()} unique IDs, "
                 f"{pt_df['label'].nunique()} models, {len(pt_df):,} pairs")

    if not pt_df.empty:
        lines.append("\n── H1: Per-turn scores by model (PRIMARY analysis) ────────────────")
        lines.append(f"{'Model':<22} {'AWMS (T1)':>10} {'5turn mean':>11} {'3turn mean':>11} {'Gap':>7} {'n':>6}")
        lines.append("-" * 75)
        for label, sub in pt_df.groupby("label"):
            awms_m = sub["awms"].dropna().mean()
            mean_5 = sub["5turn_mean"].dropna().mean()
            mean_3 = sub["3turn_mean"].dropna().mean()
            gap = awms_m - mean_3
            n = len(sub)
            lines.append(f"{label:<22} {awms_m:>10.3f} {mean_5:>11.3f} {mean_3:>11.3f} {gap:>7.3f} {n:>6,}")

    lines.append("\n── H1: Multi-turn surfaces failure modes (supplementary — bulk scorer) ──")
    lines.append(f"{'Model':<22} {'AWMS (T1)':>10} {'AWVS (T3-5)':>12} {'Gap':>7} {'Aware-soft%':>12}")
    lines.append("-" * 70)
    for label, sub in df.groupby("label"):
        awms_m = sub["awms"].mean()
        awvs_m = sub["awvs"].mean()
        gap = awms_m - awvs_m
        aware_soft = ((sub["awms"] >= 0.5) & (sub["awvs"] < 0.5)).mean() * 100
        lines.append(f"{label:<22} {awms_m:>10.3f} {awvs_m:>12.3f} {gap:>7.3f} {aware_soft:>11.1f}%")

    lines.append("\n── H2: Pressure type effectiveness (per-turn data, T3–T5) ───────────")
    if not pt_df.empty:
        all_pressure_records = []
        for _, row in pt_df.iterrows():
            pt = row["pressure_types"]
            if isinstance(pt, list) and len(pt) >= 3:
                for i, pressure in enumerate(pt):
                    t = i + 3
                    score = row.get(f"t{t}")
                    if score is not None and not (isinstance(score, float) and np.isnan(score)):
                        all_pressure_records.append({
                            "pressure": pressure.lower(), "score": float(score),
                            "label": row["label"],
                        })
        if all_pressure_records:
            pdf = pd.DataFrame(all_pressure_records)
            pooled = pdf.groupby("pressure")["score"].mean().sort_values()
            lines.append("Pooled mean per-turn score by pressure type (lower = harder to resist):")
            for ptype, mean_score in pooled.items():
                n_p = pdf[pdf["pressure"] == ptype]["score"].count()
                lines.append(f"  {ptype:<15} {mean_score:.3f}  (n={n_p:,})")

            lines.append("\nPer-model mean score by pressure type:")
            header = f"  {'Model':<22}" + "".join(f" {p[:6]:>8}" for p in PRESSURE_ORDER)
            lines.append(header)
            lines.append("  " + "-" * (22 + 8 * len(PRESSURE_ORDER)))
            for label, lsub in pdf.groupby("label"):
                row_str = f"  {label:<22}"
                for p in PRESSURE_ORDER:
                    vals = lsub[lsub["pressure"] == p]["score"]
                    row_str += f" {vals.mean():>8.3f}" if len(vals) > 0 else f" {'—':>8}"
                lines.append(row_str)

        species_rows = pt_df[pt_df["species"].notna()]
        if not species_rows.empty:
            records = []
            for _, row in species_rows.iterrows():
                pt = row["pressure_types"]
                if isinstance(pt, list) and len(pt) >= 3:
                    for i, pressure in enumerate(pt):
                        t = i + 3
                        score = row.get(f"t{t}")
                        if score is not None and not (isinstance(score, float) and np.isnan(score)):
                            records.append({"species": row["species"], "pressure": pressure.lower(), "score": float(score)})
            if records:
                hdf = pd.DataFrame(records)
                top_n = hdf.groupby("pressure")["score"].mean().sort_values()
                lines.append("\nSpecies-restricted mean per-turn score by pressure type:")
                for ptype, mean_score in top_n.items():
                    n = hdf[hdf["pressure"] == ptype]["score"].count()
                    lines.append(f"  {ptype:<15} {mean_score:.3f}  (n={n:,})")

    lines.append("\n── H3: AWMS → AWVS Correlation (per-turn scorer, 5-turn mean) ─────────")
    valid = pt_df.dropna(subset=["awms", "5turn_mean"])
    x_all = valid["awms"].values
    y_all = valid["5turn_mean"].values
    r = _pearson(x_all, y_all)
    rho = _spearman(x_all, y_all)
    lines.append(f"Pooled Pearson r  = {r:.4f}")
    lines.append(f"Pooled Spearman ρ = {rho:.4f}")
    lines.append(f"n = {len(valid):,}  (7 models, 5-turn mean)")
    lines.append("")
    lines.append("Per-model Pearson r:")
    for label, sub in pt_df.groupby("label"):
        sub_valid = sub.dropna(subset=["awms", "5turn_mean"])
        r_m = _pearson(sub_valid["awms"].values, sub_valid["5turn_mean"].values)
        lines.append(f"  {label:<22} r = {r_m:+.4f}  (n={len(sub_valid):,})")

    out = os.path.join(REPO, "qualitative_analyses", "central_argument_analysis", "central_argument_summary.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    df = load_manta_scorer(MANTA_SCORER_DIRS)
    pt_df = load_per_turn(PER_TURN_DIRS)

    if df.empty:
        print("ERROR: no manta_scorer data found", file=sys.stderr)
        sys.exit(1)

    print(f"\nGenerating charts → {FIGS}")

    print("\n[H1a] AWMS vs AWVS bars …")
    plot_h1a_awms_vs_awvs_bars(pt_df)

    if not pt_df.empty:
        print("\n[H1b] Score trajectory …")
        plot_h1b_score_trajectory(pt_df)

        print("\n[H1d] 5-turn vs 3-turn mean bars …")
        plot_h1d_5turn_3turn_bars(pt_df)

    print("\n[H1c] Aware-but-soft quadrant …")
    plot_h1c_aware_but_soft(pt_df)

    if not pt_df.empty:
        print("\n[H2] Species × pressure heatmap …")
        plot_h2_species_pressure_heatmap(pt_df)

        print("\n[H2b] Pressure type bars …")
        plot_h2b_pressure_type_bars(pt_df)

    print("\n[H3a] AWMS × AWVS scatter …")
    plot_h3a_awms_awvs_scatter(pt_df)

    print("\n[H3b] Correlation by model …")
    plot_h3b_correlation_by_model(pt_df)

    print("\n[Summary] Writing summary …")
    write_summary(df, pt_df)

    print("\nDone. All files written to analysis/figures/")


if __name__ == "__main__":
    main()
