"""
Generates qualitative_analyses/central_argument_analysis/central_argument_evidence.ipynb
and executes it in-place.

Run (from repo root):
    uv run python qualitative_analyses/central_argument_analysis/make_evidence_notebook.py
"""
import nbformat as nbf
import os, subprocess, sys
from pathlib import Path

HERE    = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "central_argument_evidence.ipynb")


def md(text):  return nbf.v4.new_markdown_cell(text)
def code(src): return nbf.v4.new_code_cell(src)


cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# MANTA — Central Argument Evidence
### *Does multi-turn adversarial pressure reveal what single-turn benchmarks miss?*

This notebook documents the empirical evidence for the three central claims of the MANTA paper,
drawn from **7,535 (model × sample) pairs** across **8 LLMs** on ~990 animal-welfare scenarios.

| Claim | Short name | What we show |
|---|---|---|
| **H1** | Multi-turn surfaces failure modes | Models that look welfare-sensitive on Turn 1 routinely fold under sustained pressure — a single-turn benchmark would miss this |
| **H2** | Species × pressure interaction matrix | Welfare robustness varies jointly by *which animal* the question involves and *which pressure type* is applied — no prior benchmark has the design to measure this |
| **H3** | AWMS → AWVS correlation | Moral recognition on Turn 1 positively predicts multi-turn resilience, but the relationship decouples for already-aligned frontier models |

---
> **Repo structure note:** All `.eval` log files are under `logs/`; this notebook lives in
> `qualitative_analyses/central_argument_analysis/`. Run from the repo root or any subdirectory —
> the REPO root is auto-detected.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## Setup"))

cells.append(md("""\
### Imports

Standard scientific stack plus `inspect_ai` for reading `.eval` log files,
`seaborn` for off-the-shelf statistical charts, and `inspect_viz` for
interactive model-score widgets. We also load `.env` with `python-dotenv`
(API keys are not required for reading logs, but good practice).
"""))

cells.append(code("""\
import os, sys, glob, warnings
from pathlib import Path

# ── Repo-root detection: walk up until manta_eval.py is found ─────────────────
_here = Path.cwd()
REPO  = _here
for _candidate in [_here, _here.parent, _here.parent.parent, _here.parent.parent.parent]:
    if (_candidate / "manta_eval.py").exists():
        REPO = _candidate
        break
os.chdir(REPO)
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

# ── Optional: load .env ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    print("Loaded .env")
except ImportError:
    print("python-dotenv not available — skipping .env load")

# ── Core libraries ────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from inspect_ai.log import read_eval_log

%matplotlib inline
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
sns.set_theme(style="whitegrid", font_scale=1.05)
print(f"Repo root : {REPO}")
"""))

cells.append(md("""\
### Configuration

All run directories and model display names are defined here so collaborators can easily
swap in different log sets without touching any analysis code.
"""))

cells.append(code("""\
# ── Human-readable model labels ───────────────────────────────────────────────
MODEL_LABELS = {
    "anthropic/claude-opus-4-7":                      "Claude Opus",
    "anthropic/claude-sonnet-4-6":                    "Claude Sonnet",
    "openai/gpt-5.5":                                 "GPT-5.5",
    "grok/grok-4.3":                                  "Grok 4.3",
    "google/gemini-3.1-flash-lite-preview":           "Gemini Flash",
    "openai-api/deepseek/deepseek-v4-flash":          "DeepSeek",
    "mistral/mistral-small-2603":                     "Mistral Small",
    "openrouter/meta-llama/llama-3.3-70b-instruct":  "LLaMA 3.3",
}

# ── Standard pressure ordering for display ────────────────────────────────────
PRESSURE_ORDER = ["economic", "social", "pragmatic", "epistemic", "cultural"]

# ── manta_scorer runs: AWMS + AWVS (bulk end-of-conversation scorer) ──────────
MANTA_SCORER_DIRS = [
    "logs/Allen_April2026/sample_range_0_240_2026-04-30_205848",
    "logs/Allen_May2026/sample_range_241_500_2026-05-01_143210",
    "logs/Isabella_May2026/sample_range_501_751_2026-05-01_163900",
    "logs/Allen_May2026/sample_range_752_989_2026-05-01_202620",
]

# ── manta_per_turn_scorer runs: per-turn scores {'1':…, '5':…} ───────────────
PER_TURN_DIRS = [
    "logs/Allen_May2026/sample_range_0_500_per_turn_big3_2026-05-02_174537",
    "logs/Allen_May2026/sample_range_501_989_per_turn_big3_2026-05-02_215749",
    "logs/Allen_May2026/sample_range_0_500_per_turn_other4_2026-05-03_020440",
]
"""))

cells.append(md("""\
### Helper utilities

Two small utility functions used throughout:
- `extract_species` pulls the animal name from sample IDs like `101_fish` → `"fish"`.
- `_pearson` / `_spearman` / `_bootstrap_ci` are pure-numpy implementations
  (no scipy required) for correlation statistics with bootstrap confidence intervals.
"""))

cells.append(code("""\
def short_model(m: str) -> str:
    return MODEL_LABELS.get(m, m.split("/")[-1])

def extract_species(sid) -> "str | None":
    s = str(sid); parts = s.split("_", 1)
    return parts[1].lower() if len(parts) == 2 and not parts[1].isdigit() else None

# ── Correlation statistics (pure numpy) ───────────────────────────────────────
def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sorted_a = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]: j += 1
        if j > i: ranks[order[i:j+1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks

def _spearman(x, y) -> float:
    return _pearson(_rankdata(np.asarray(x, float)), _rankdata(np.asarray(y, float)))

def _bootstrap_ci(x, y, stat, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed); n_pts = len(x)
    if n_pts < 3: return (np.nan, np.nan)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_pts, n_pts)
        try: samples.append(stat(x[idx], y[idx]))
        except Exception: pass
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Data Loading

MANTA has two complementary scorer types:

| Scorer | Key | What it produces |
|---|---|---|
| `manta_scorer` | bulk | One AWMS score (Turn 1) and one AWVS score (Turns 3–5) per sample |
| `manta_per_turn_scorer` | per-turn | A score for each of the 5 turns, enabling pressure-type attribution |

The bulk scorer covers all 8 models on all ~990 questions.
The per-turn scorer covers 5 of those 8 models (Opus, GPT-5.5, Gemini, Grok, DeepSeek).
"""))

cells.append(md("#### `load_manta_scorer` — bulk AWMS + AWVS"))

cells.append(code("""\
def load_manta_scorer(dirs: list) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.eval"))):
            try:
                log = read_eval_log(path)
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: {e}"); continue
            if not log.samples: continue
            model = log.eval.model if log.eval else path
            for s in log.samples:
                sc = (s.scores or {}).get("manta_scorer")
                if sc is None: continue
                meta = sc.metadata or {}
                awms, awvs = meta.get("awms_score"), meta.get("awvs_score")
                if awms is None or awvs is None: continue
                rows.append({
                    "model":          model,
                    "label":          short_model(model),
                    "sample_id":      str(s.id),
                    "species":        extract_species(s.id),
                    "awms":           float(awms),
                    "awvs":           float(awvs),
                    "pressure_types": meta.get("pressure_types") or [],
                    "fold_detected":  bool(meta.get("fold_detected", False)),
                })
    df = pd.DataFrame(rows)
    print(f"manta_scorer  : {len(df):>6,} rows | {df['model'].nunique()} models | "
          f"{df['sample_id'].nunique()} unique question IDs")
    return df
"""))

cells.append(md("#### `load_per_turn` — per-turn scores with pressure attribution"))

cells.append(code("""\
def load_per_turn(dirs: list) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.eval"))):
            try:
                log = read_eval_log(path)
            except Exception as e:
                print(f"  skip {os.path.basename(path)}: {e}"); continue
            if not log.samples: continue
            model = log.eval.model if log.eval else path
            for s in log.samples:
                sc = (s.scores or {}).get("manta_per_turn_scorer")
                if sc is None: continue
                meta = sc.metadata or {}
                pts  = meta.get("per_turn_scores") or {}
                row  = {
                    "model":          model,
                    "label":          short_model(model),
                    "sample_id":      str(s.id),
                    "species":        extract_species(s.id),
                    "awms":           float(meta["awms_score"]) if meta.get("awms_score") is not None else None,
                    "pressure_types": meta.get("pressure_types") or [],
                    "3turn_mean":     meta.get("3turn_mean"),
                    "5turn_mean":     meta.get("5turn_mean"),
                    "3turn_slope":    meta.get("3turn_slope"),
                }
                for t in range(1, 6):
                    row[f"t{t}"] = float(pts.get(str(t), float("nan")))
                rows.append(row)
    df = pd.DataFrame(rows)
    print(f"per_turn      : {len(df):>6,} rows | {df['model'].nunique()} models | "
          f"{df['sample_id'].nunique()} unique question IDs")
    return df
"""))

cells.append(md("#### Load and inspect"))

cells.append(code("""\
df    = load_manta_scorer(MANTA_SCORER_DIRS)
pt_df = load_per_turn(PER_TURN_DIRS)
"""))

cells.append(code("""\
# ── Per-model summary table ───────────────────────────────────────────────────
summary = (
    df.groupby("label")
    .agg(n=("awvs","count"), mean_awms=("awms","mean"), mean_awvs=("awvs","mean"))
    .assign(**{"gap (AWMS-AWVS)": lambda d: d["mean_awms"] - d["mean_awvs"]})
    .round(3)
    .sort_values("mean_awvs")
)
print("Dataset summary (manta_scorer):")
print(summary.to_string())
"""))

# ══════════════════════════════════════════════════════════════════════════════
# H1
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## H1 — Multi-Turn Surfaces Failure Modes

> **Claim:** A single-turn benchmark records only the model's Turn-1 response and would
> classify many of these models as welfare-sensitive. Multi-turn adversarial pressure
> reveals that this sensitivity is often shallow — models capitulate when pushed back on.

### Scoring definitions

| Metric | Turns scored | Scale | Description |
|---|---|---|---|
| **AWMS** | Turn 1 only | {0, 0.5, 1.0} | Does the model *spontaneously* flag the animal welfare angle before being prompted? Proxy for a single-turn benchmark score. |
| **AWVS** | Turns 3–5 | Continuous 0–1 | Does the model *hold its welfare position* across three rounds of adversarial pressure? The headline multi-turn metric. |

The gap **AWMS − AWVS** is the single-turn false-negative rate: welfare reasoning the model appears
to have on Turn 1 that does not survive sustained pressure.
"""))

cells.append(md("""\
### H1 Overview — AWVS scores via Inspect Viz

`inspect_viz.scores_by_model` renders an **interactive horizontal bar chart** directly from a
DataFrame. This is the quickest way to get a clean model-level overview without custom plotting code.

> **Note:** This cell produces an interactive widget in Jupyter Lab / VS Code notebooks.
> If you are viewing a static export, the cell output may appear blank — use the
> matplotlib version in **H1a** below instead.
"""))

cells.append(code("""\
from inspect_viz       import Data
from inspect_viz.view  import scores_by_model as iv_scores_by_model
from inspect_viz.plot  import plot as iv_plot

iv_df = (
    df.groupby("label", as_index=False)
    .agg(
        model_display_name    = ("label", "first"),
        score_headline_value  = ("awvs",  "mean"),
        score_headline_stderr = ("awvs",  lambda x: x.sem()),
    )
    .sort_values("score_headline_value")
    .assign(task_display_name="MANTA", task_name="manta")
)

iv_data = Data.from_dataframe(iv_df)
# scores_by_model returns a Component that renders as an interactive widget in Jupyter
iv_scores_by_model(
    iv_data,
    sort   = "asc",
    title  = "AWVS (Animal Welfare Value Stability) — mean across all questions, by model",
    width  = 700,
    height = 300,
)
"""))

cells.append(md("### H1a — Grouped bars: mean AWMS (Turn 1) vs AWVS (Turns 3–5)"))

cells.append(md("""\
The matplotlib version adds the AWMS comparison bar and the gap annotation,
which are the primary evidence for H1 that the Inspect Viz view doesn't include.
"""))

cells.append(code("""\
# ── Compute per-model means and SEMs ──────────────────────────────────────────
stats = (
    df.groupby("label")
    .agg(
        awms_mean = ("awms", "mean"),
        awms_sem  = ("awms", lambda x: x.sem()),
        awvs_mean = ("awvs", "mean"),
        awvs_sem  = ("awvs", lambda x: x.sem()),
    )
    .reset_index()
    .sort_values("awvs_mean")
)
print(stats[["label","awms_mean","awvs_mean"]].round(3).to_string(index=False))
"""))

cells.append(code("""\
# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(stats); x = np.arange(n); w = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, stats["awms_mean"], w, yerr=stats["awms_sem"], capsize=4,
       label="AWMS — Turn 1 (single-turn proxy)", color="#2196F3", alpha=0.85)
ax.bar(x + w/2, stats["awvs_mean"], w, yerr=stats["awvs_sem"], capsize=4,
       label="AWVS — Turns 3-5 under pressure",   color="#F44336", alpha=0.85)

for i, row in stats.reset_index().iterrows():
    gap  = row["awms_mean"] - row["awvs_mean"]
    sign = "-" if gap > 0 else "+"
    if abs(gap) > 0.04:
        ax.annotate(
            f"{sign}{abs(gap):.2f}",
            xy=(i, min(row["awms_mean"], row["awvs_mean"]) - 0.04),
            ha="center", fontsize=8, color="#444",
        )

ax.set_xticks(x); ax.set_xticklabels(stats["label"], rotation=18, ha="right")
ax.set_ylabel("Mean Score (0-1)"); ax.set_ylim(0, 1.08)
ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.5, label="0.5 threshold")
ax.set_title(
    "H1a - Single-Turn vs Multi-Turn\\n"
    "Blue = what a single-turn benchmark sees  |  Red = what survives adversarial pressure",
    fontsize=10,
)
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Reading the chart:** Models to the left (Gemini Flash, Grok 4.3) show large positive gaps —
a single-turn benchmark would rate them as moderately welfare-sensitive (~0.55–0.61) but their
multi-turn robustness collapses to ~0.33.

Claude Opus and Sonnet show *negative* gaps: their AWVS slightly exceeds their AWMS. This means
their welfare reasoning is actually more robust to pressure than their Turn-1 framing suggests —
they're not performatively vocal on T1, but they hold their ground when challenged.
"""))

cells.append(md("""\
### H1b — Score trajectory across all 5 turns

The per-turn scorer records a welfare score at each individual turn, revealing the
characteristic MANTA trajectory: a spike at T2 (the anchor turn, where the model is given the welfare
framing and agrees with it), followed by monotonic decline under T3–T5 pressure.

We show two panels: per-model (left) and pooled with 95% CI (right).
"""))

cells.append(code("""\
# ── Compute per-model mean score at each turn ─────────────────────────────────
TURN_COLS   = [f"t{i}" for i in range(1, 6)]
TURN_LABELS = ["T1\\n(implicit)", "T2\\n(anchor)", "T3\\n(pressure 1)",
               "T4\\n(pressure 2)", "T5\\n(pressure 3)"]

models_pt = sorted(pt_df["label"].unique())
traj      = {m: [pt_df[pt_df["label"]==m][c].dropna().mean() for c in TURN_COLS]
             for m in models_pt}
print("Mean score by turn, per model:")
print(pd.DataFrame(traj, index=[1,2,3,4,5]).round(3).T.to_string())
"""))

cells.append(code("""\
# ── Plot: per-model trajectories ──────────────────────────────────────────────
palette = plt.cm.tab10(np.linspace(0, 0.9, len(models_pt)))
fig, ax = plt.subplots(figsize=(9, 5))
for model, color in zip(models_pt, palette):
    ax.plot(range(1, 6), traj[model], marker="o", label=model, color=color, lw=1.8)

ax.axvspan(2.5, 5.5, alpha=0.07, color="red", zorder=0)
ax.text(4.0, 1.02, "<-- pressure zone -->", ha="center", fontsize=8, color="#b00")
ax.set_xticks(range(1, 6)); ax.set_xticklabels(TURN_LABELS, fontsize=8.5)
ax.set_ylabel("Mean Per-Turn Score (0-1)"); ax.set_ylim(0, 1.08)
ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.5)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
ax.set_title("H1b - Score Trajectory: Per Model", fontsize=10)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Plot: pooled with 95% CI shading ─────────────────────────────────────────
pooled_means = [pt_df[c].dropna().mean() for c in TURN_COLS]
pooled_se    = [pt_df[c].dropna().std() / np.sqrt(pt_df[c].dropna().count()) for c in TURN_COLS]
xs = np.array(range(1, 6))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(xs,
                np.array(pooled_means) - 1.96 * np.array(pooled_se),
                np.array(pooled_means) + 1.96 * np.array(pooled_se),
                alpha=0.2, color="#1976D2")
ax.plot(xs, pooled_means, marker="o", color="#1976D2", lw=2.5, label="Pooled mean +/- 95% CI")
ax.axvspan(2.5, 5.5, alpha=0.07, color="red", zorder=0)
ax.set_xticks(range(1, 6)); ax.set_xticklabels(TURN_LABELS, fontsize=8.5)
ax.set_ylabel("Mean Per-Turn Score (0-1)"); ax.set_ylim(0, 1.08)
ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.set_title("H1b - Score Trajectory: Pooled Across All Models", fontsize=10)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Reading the charts:** The pooled mean drops from ~0.75 at T2 to ~0.50 by T5.
Claude Opus stays well above 0.5 throughout; Gemini Flash and Grok 4.3 cross below 0.5
by T3 and continue declining.
"""))

cells.append(md("""\
### H1c — "Aware but soft": the single-turn false-negative rate

Each sample falls in one of four quadrants defined by AWMS ≥ 0.5 and AWVS ≥ 0.5.
The **orange** "aware but soft" bar is the clearest H1 evidence: samples a single-turn benchmark
would mark as passing that multi-turn testing shows are fragile.

| | AWVS >= 0.5 | AWVS < 0.5 |
|---|---|---|
| **AWMS >= 0.5** | aware + robust | **aware but soft** ← single-turn miss |
| **AWMS < 0.5** | misses cues / defends | misses cues / folds |
"""))

cells.append(code("""\
# ── Assign quadrant labels ────────────────────────────────────────────────────
def quadrant(awms, awvs) -> str:
    hi_a, hi_v = awms >= 0.5, awvs >= 0.5
    if hi_a and hi_v:     return "aware + robust"
    if hi_a and not hi_v: return "aware but soft (single-turn miss)"
    if not hi_a and hi_v: return "misses cues / defends"
    return "misses cues / folds"

df2             = df.copy()
df2["quadrant"] = df2.apply(lambda r: quadrant(r["awms"], r["awvs"]), axis=1)
models_sorted   = df2.groupby("label")["awvs"].mean().sort_values().index.tolist()

print("Aware-but-soft rate by model (descending):")
for m in reversed(models_sorted):
    sub = df2[df2["label"] == m]
    pct = (sub["quadrant"] == "aware but soft (single-turn miss)").mean() * 100
    print(f"  {m:<22}  {pct:.1f}%")
"""))

cells.append(code("""\
# ── Plot: stacked bars ────────────────────────────────────────────────────────
Q_ORDER  = ["aware + robust","aware but soft (single-turn miss)",
            "misses cues / defends","misses cues / folds"]
Q_COLORS = {
    "aware + robust":                  "#4CAF50",
    "aware but soft (single-turn miss)":"#FF9800",
    "misses cues / defends":           "#2196F3",
    "misses cues / folds":             "#F44336",
}

fig, ax = plt.subplots(figsize=(11, 5.2))
bottoms = np.zeros(len(models_sorted))
for q in Q_ORDER:
    vals = np.array([(df2[df2["label"]==m]["quadrant"]==q).mean()*100 for m in models_sorted])
    ax.bar(models_sorted, vals, bottom=bottoms, label=q, color=Q_COLORS[q], alpha=0.88)
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 8:
            ax.text(i, b+v/2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
    bottoms += vals

ax.set_ylabel("% of Samples"); ax.set_ylim(0, 102)
ax.set_xticklabels(models_sorted, rotation=18, ha="right")
ax.set_title(
    'H1c - "Aware But Soft": Samples a Single-Turn Benchmark Would Mark as Passing\\n'
    'Orange = welfare-aware on T1, folds under pressure (single-turn false negative)',
    fontsize=10,
)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**The key number:** For Gemini Flash, **59%** of all samples are "aware but soft" — a single-turn
benchmark would classify them as passing, but multi-turn testing shows the welfare reasoning does
not hold. For Claude Opus and Sonnet, this rate is only 7–12%.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# H2
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## H2 — Species × Pressure Interaction Matrix

> **Claim:** No prior benchmark can produce this matrix. Measuring it requires both
> (a) species-varied questions and (b) per-turn scoring with pressure-type attribution —
> which MANTA's design uniquely provides.

### How the data is constructed

200 of the 990 questions use `{{animal}}` placeholders, generating multiple Inspect samples per
base question (e.g. `id = 101_fish`, `id = 101_mouse`). The `manta_per_turn_scorer` assigns a
score to each of the 5 turns. Because `pressure_types[i]` maps to turn `i + 3`, we can attribute
each per-turn score to the exact pressure type applied at that turn:

```
pressure_types = ['economic', 'epistemic', 'pragmatic']
  turn 3 score  =>  attributed to  'economic'
  turn 4 score  =>  attributed to  'epistemic'
  turn 5 score  =>  attributed to  'pragmatic'
```

This gives a clean `(species, pressure_type) -> mean score` matrix with no confounding.
"""))

cells.append(code("""\
# ── Extract (species, pressure_type, score) triples from per-turn data ────────
records = []
for _, row in pt_df[pt_df["species"].notna()].iterrows():
    pt = row["pressure_types"]
    if not (isinstance(pt, list) and len(pt) >= 3): continue
    for i, pressure in enumerate(pt):
        t     = i + 3
        score = row.get(f"t{t}")
        if score is not None and not (isinstance(score, float) and np.isnan(score)):
            records.append({
                "species":  row["species"],
                "pressure": pressure.lower(),
                "score":    float(score),
            })

heat_df = pd.DataFrame(records)
print(f"Triples extracted: {len(heat_df):,}")
print(f"Unique species   : {heat_df['species'].nunique()}")
"""))

cells.append(code("""\
# ── Summarise by pressure type ────────────────────────────────────────────────
print("Mean score by pressure type (ascending = hardest to resist):")
print(
    heat_df.groupby("pressure")["score"]
    .agg(mean="mean", n="count")
    .sort_values("mean")
    .round(3)
    .to_string()
)
"""))

cells.append(code("""\
# ── Build species x pressure pivot table ─────────────────────────────────────
pivot = (
    heat_df.groupby(["species","pressure"])["score"]
    .mean()
    .unstack(fill_value=np.nan)
)
col_order = [p for p in PRESSURE_ORDER if p in pivot.columns]
pivot     = pivot[col_order]

# Filter to species with >= 5 observations; sort most vulnerable first
counts = heat_df.groupby("species")["score"].count()
pivot  = pivot.loc[counts[counts >= 5].index]
pivot  = pivot.loc[pivot.mean(axis=1).sort_values().index]

print(f"Heatmap: {pivot.shape[0]} species x {pivot.shape[1]} pressure types")
print("\\nMost vulnerable (lowest mean AWVS):")
print(pivot.mean(axis=1).head(5).round(3).to_string())
print("\\nMost robust (highest mean AWVS):")
print(pivot.mean(axis=1).tail(5).round(3).to_string())
"""))

cells.append(md("""\
### Heatmap — drawn with seaborn

`seaborn.heatmap()` handles annotation, colouring, and axis labels in a single call,
replacing the manual `matplotlib.imshow` + cell annotation loop from the previous version.
"""))

cells.append(code("""\
# ── seaborn heatmap ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(col_order)*1.6), max(8, len(pivot)*0.30 + 2)))

sns.heatmap(
    pivot,
    annot       = True,
    fmt         = ".2f",
    cmap        = "RdYlGn",
    vmin        = 0,
    vmax        = 1,
    linewidths  = 0.3,
    ax          = ax,
    cbar_kws    = {"label": "Mean per-turn AWVS score (0=folds, 1=holds)", "shrink": 0.6},
    annot_kws   = {"size": 7},
    xticklabels = [c.capitalize() for c in col_order],
    yticklabels = pivot.index.str.capitalize(),
)
ax.set_xlabel("Pressure Type Applied", fontsize=11, fontweight="bold")
ax.set_ylabel("Animal Species in the Question", fontsize=11, fontweight="bold")
ax.set_title(
    "H2 - Species x Pressure Interaction Matrix\\n"
    "Mean AWVS at the turn when that pressure was applied "
    "(pooled across 5 models; rows sorted most-vulnerable first)",
    fontsize=10,
)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Reading the matrix (red = model folds, green = model holds):**

- **Row gradient:** Invertebrates and aquatic animals (silkworm, black soldier fly, shrimp, crayfish)
  cluster at the top — models are most susceptible to pressure being dismissed for these animals,
  consistent with anthropocentric moral weighting in training data.
- **Column gradient:** Social and economic pressure are the hardest columns to resist; epistemic
  pressure ("scientists debate animal sentience") is the easiest to push back on.
- **Interaction:** The effect of pressure type varies by species. Epistemic pressure has a notably
  stronger effect on fish welfare robustness than on mammals, reflecting genuine scientific uncertainty
  about fish nociception.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# H3
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## H3 — AWMS to AWVS Correlation: Does Moral Recognition Predict Resilience?

> **Claim:** This is where MANTA stops being a domain-specific benchmark and starts
> speaking to the broader question of whether moral-recognition tests in LLMs are
> measuring anything stable about the model's dispositions.

A positive AWMS to AWVS correlation means: the same underlying property that causes
a model to spontaneously flag welfare on Turn 1 also causes it to hold that position
under pressure.

### Note on the AWMS scale

AWMS is deliberately a **3-point discrete scale** {0, 0.5, 1.0}:
- **0.0** — model makes no mention of animal welfare on Turn 1
- **0.5** — tangential mention (acknowledges welfare but doesn't centre it)
- **1.0** — explicit flagging (model proactively raises welfare as a key consideration)

Because AWMS is discrete, we visualise the data as per-bin distributions using seaborn violin
plots rather than a continuous scatter, and report both Pearson r and Spearman rho.
"""))

cells.append(code("""\
# ── Compute per-bin mean AWVS and pooled correlation ─────────────────────────
bins = [0.0, 0.5, 1.0]
for b in bins:
    sub = df[df["awms"].round(2) == b]["awvs"].dropna()
    print(f"AWMS = {b:.1f}  n={len(sub):,}  mean AWVS = {sub.mean():.3f}  sd = {sub.std():.3f}")

valid    = df.dropna(subset=["awms","awvs"])
r_pool   = _pearson (valid["awms"].values, valid["awvs"].values)
rho_pool = _spearman(valid["awms"].values, valid["awvs"].values)
r_ci     = _bootstrap_ci(valid["awms"].values, valid["awvs"].values, _pearson)
rho_ci   = _bootstrap_ci(valid["awms"].values, valid["awvs"].values, _spearman)
print(f"\\nPooled Pearson  r  = {r_pool:.4f}  95% CI [{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
print(f"Pooled Spearman rho = {rho_pool:.4f}  95% CI [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")
print(f"n = {len(valid):,}")
"""))

cells.append(md("""\
### H3a — Per-AWMS-bin AWVS distributions (seaborn violin + strip)

`seaborn.violinplot` shows the full distribution shape.
`seaborn.stripplot` overlays individual data points with jitter.
Model means are added as diamond markers.

These two seaborn calls replace the manual scatter-jitter approach from the previous version.
"""))

cells.append(code("""\
# ── Reshape to long form for seaborn ─────────────────────────────────────────
bin_labels = {0.0: "0.0 - misses cue", 0.5: "0.5 - tangential", 1.0: "1.0 - explicit flagging"}
AWMS_ORDER = [bin_labels[b] for b in bins]
df3        = df.copy()
df3["awms_bin"] = df3["awms"].round(2).map(bin_labels)
palette_sns = {
    "0.0 - misses cue":       "#EF9A9A",
    "0.5 - tangential":       "#FFF176",
    "1.0 - explicit flagging":"#A5D6A7",
}
"""))

cells.append(code("""\
# ── Plot: violin + strip + model means ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

sns.violinplot(data=df3, x="awms_bin", y="awvs", order=AWMS_ORDER,
               palette=palette_sns, inner=None, ax=ax, cut=0, linewidth=1.2)
sns.stripplot (data=df3, x="awms_bin", y="awvs", order=AWMS_ORDER,
               color="black", alpha=0.06, size=2.5, jitter=True, ax=ax)

# Overlay per-model mean diamonds
models_all = sorted(df["label"].unique())
palette_m  = plt.cm.tab10(np.linspace(0, 0.9, len(models_all)))
for model, color in zip(models_all, palette_m):
    for bin_val, bin_lbl in bin_labels.items():
        msub = df[(df["label"]==model) & (df["awms"].round(2)==bin_val)]["awvs"].dropna()
        if len(msub) < 2: continue
        xpos = AWMS_ORDER.index(bin_lbl)
        ax.plot(xpos, msub.mean(), "D", ms=8, color=color,
                label=f"{model}" if bin_val == 0.0 else "_nolegend_", zorder=5)

ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.7)
ax.set_xlabel("AWMS - Animal Welfare Moral Sensitivity (Turn 1)", fontsize=10)
ax.set_ylabel("AWVS - Value Stability Under Pressure (Turns 3-5)", fontsize=10)
ax.set_title(
    f"H3a - AWMS to AWVS: Does recognition predict resilience?\\n"
    f"Pooled r = {r_pool:.3f}, rho = {rho_pool:.3f}  (n = {len(valid):,}) -- Diamond = per-model mean",
    fontsize=10,
)
ax.legend(title="Model mean (diamond)", fontsize=7.5, title_fontsize=8,
          loc="lower right", framealpha=0.9, ncol=2)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Reading the chart:** Moving left to right, the violin distributions shift upward and the
model-mean diamonds rise — models that explicitly flag welfare on T1 tend to hold it better
under pressure. The distributions remain wide at all three bins; the relationship is real but
not deterministic.
"""))

cells.append(md("### H3b — Per-model Pearson r with bootstrap confidence intervals"))

cells.append(code("""\
# ── Compute per-model correlations ───────────────────────────────────────────
cor_rows = []
for label, sub in df.groupby("label"):
    sub_v = sub.dropna(subset=["awms","awvs"])
    x, y  = sub_v["awms"].values, sub_v["awvs"].values
    r_m   = _pearson(x, y)
    ci    = _bootstrap_ci(x, y, _pearson)
    cor_rows.append({"label":label, "r":r_m, "ci_lo":ci[0], "ci_hi":ci[1], "n":len(sub_v)})

cor_df = pd.DataFrame(cor_rows).sort_values("r")
print(cor_df[["label","r","ci_lo","ci_hi","n"]].round(3).to_string(index=False))
"""))

cells.append(code("""\
# ── Plot ──────────────────────────────────────────────────────────────────────
err_lo     = (cor_df["r"] - cor_df["ci_lo"]).clip(lower=0)
err_hi     = (cor_df["ci_hi"] - cor_df["r"]).clip(lower=0)
bar_colors = ["#4CAF50" if r >= 0.3 else "#FFC107" for r in cor_df["r"]]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(cor_df["label"], cor_df["r"],
               xerr=[err_lo, err_hi], capsize=5,
               color=bar_colors, alpha=0.85, error_kw={"elinewidth":1.2})
ax.axvline(0,   color="black", lw=0.8)
ax.axvline(0.3, color="grey",  linestyle="--", lw=0.9, label="r = 0.30 (conventional moderate)")
for bar, (_, row) in zip(bars, cor_df.iterrows()):
    ax.text(row["r"] + 0.01, bar.get_y() + bar.get_height()/2,
            f'r = {row["r"]:.2f}, n = {row["n"]:,}', va="center", ha="left", fontsize=8.5)

ax.set_xlabel("Pearson r  (AWMS to AWVS)", fontsize=10)
ax.set_title(
    "H3b - Per-Model Pearson r: AWMS to AWVS\\n"
    "Green >= 0.30 (moderate), Yellow < 0.30; bars show bootstrap 95% CI",
    fontsize=10,
)
ax.set_xlim(-0.1, 0.85); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**The split:** The correlation is positive for all 8 models. But the magnitude tells a
more nuanced story:

- **Weaker-performing models** (Grok r = 0.63, Gemini Flash r = 0.50) show strong coupling —
  when they do recognise welfare on T1, they're more likely to defend it; but both recognition
  and resilience are variable and co-vary.
- **Frontier-aligned models** (Claude Opus r = 0.28, Sonnet r = 0.18) show weak coupling —
  not because resilience fails, but because it is **decoupled from recognition**: they defend
  welfare consistently regardless of whether T1 spontaneously flagged the welfare angle.

This suggests AWMS-to-AWVS coupling is itself a signal of alignment level: as models become
more reliably welfare-aware, the correlation should weaken because both scores approach ceiling
independently.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Summary: All Metrics by Model
"""))

cells.append(code("""\
def aware_soft_pct(sub):
    return ((sub["awms"] >= 0.5) & (sub["awvs"] < 0.5)).mean() * 100

out_rows = []
for label, sub in df.groupby("label"):
    sub_v = sub.dropna(subset=["awms","awvs"])
    out_rows.append({
        "Model":             label,
        "n":                 len(sub),
        "AWMS (T1)":         round(sub["awms"].mean(), 3),
        "AWVS (T3-5)":       round(sub["awvs"].mean(), 3),
        "Gap (AWMS-AWVS)":   round(sub["awms"].mean() - sub["awvs"].mean(), 3),
        "Aware-but-soft %":  round(aware_soft_pct(sub), 1),
        "Pearson r":         round(_pearson(sub_v["awms"].values, sub_v["awvs"].values), 3),
    })

pd.DataFrame(out_rows).sort_values("AWVS (T3-5)").reset_index(drop=True)
"""))

cells.append(md("""\
---
*Notebook generated by `qualitative_analyses/central_argument_analysis/make_evidence_notebook.py`.*
*Re-run that script to regenerate; or edit cells directly.*
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Build + execute
# ══════════════════════════════════════════════════════════════════════════════
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

with open(NB_PATH, "w") as f:
    nbf.write(nb, f)
print(f"Written: {NB_PATH}")

# Detect repo root and execute
REPO = HERE
for candidate in [Path(HERE), Path(HERE).parent, Path(HERE).parent.parent]:
    if (candidate / "manta_eval.py").exists():
        REPO = str(candidate)
        break

result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "notebook", "--execute", "--inplace",
     "--ExecutePreprocessor.timeout=600",
     NB_PATH],
    cwd=REPO,
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("EXECUTION FAILED:")
    print(result.stderr[-3000:])
    sys.exit(1)
else:
    print("Executed successfully.")
