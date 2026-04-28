"""
Comparability experiment analysis report.
Prints a score comparison table and T2 follow-up texts across all 4 conditions.

Run from the manta/ root:
    python dataset/analysis_report.py
"""
import json
from pathlib import Path
from inspect_ai.log import read_eval_log

LOG_DIR = Path("logs")
CONDITIONS = {
    "control":   "control_sonnet",
    "hybrid":    "hybrid_sonnet",
    "injection": "injection_sonnet",
    "combined":  "combined_sonnet",
}
QUESTION_IDS = ["0", "1", "2", "3", "4"]


def latest_log(dirname: str):
    d = LOG_DIR / dirname
    if not d.exists():
        return None
    logs = sorted(d.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def extract_scores(log_path) -> dict:
    """Returns {question_id: {pr_score, overall_score, t2_followup, t2_response}}."""
    log = read_eval_log(str(log_path))
    results = {}
    for sample in log.samples:
        sid = str(sample.id)

        # Scores
        scores = sample.scores or {}
        manta = scores.get("manta_scorer")
        if manta and hasattr(manta, "value"):
            val = manta.value
            pr = val.get("Value Stability", {}).get("score", "N/A") if isinstance(val, dict) else "N/A"
            overall = manta.as_float() if hasattr(manta, "as_float") else "N/A"
        else:
            pr, overall = "N/A", "N/A"

        # Messages: [system(0), user/T1(1), assistant/T1(2), user/T2(3), assistant/T2(4)]
        msgs = sample.messages or []
        t2_followup = msgs[3].content[:200] if len(msgs) > 3 else "—"
        t2_response  = msgs[4].content[:200] if len(msgs) > 4 else "—"

        # Truncate if content is a list (multimodal)
        if isinstance(t2_followup, list):
            t2_followup = str(t2_followup)[:200]
        if isinstance(t2_response, list):
            t2_response = str(t2_response)[:200]

        results[sid] = {
            "pr": round(pr, 2) if isinstance(pr, float) else pr,
            "overall": round(overall, 2) if isinstance(overall, float) else overall,
            "t2_followup": t2_followup,
            "t2_response": t2_response,
        }
    return results


# ── Load all conditions ──────────────────────────────────────────────────────
all_results = {}
for cond, dirname in CONDITIONS.items():
    lp = latest_log(dirname)
    if lp:
        print(f"Loading {cond}: {lp.name}")
        all_results[cond] = extract_scores(lp)
    else:
        print(f"[MISSING] No log found in logs/{dirname}/")
        all_results[cond] = {}

# ── Score table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PRESSURE ROBUSTNESS SCORES (0.0–1.0; N/A = not applicable)")
print("=" * 80)
header = f"{'Q':<4}" + "".join(f"{c.upper():<16}" for c in CONDITIONS)
print(header)
print("-" * len(header))
for qid in QUESTION_IDS:
    row = f"{qid:<4}"
    for cond in CONDITIONS:
        val = all_results[cond].get(qid, {}).get("pr", "—")
        row += f"{str(val):<16}"
    print(row)

print("\n" + "=" * 80)
print("OVERALL SCORES")
print("=" * 80)
print(header)
print("-" * len(header))
for qid in QUESTION_IDS:
    row = f"{qid:<4}"
    for cond in CONDITIONS:
        val = all_results[cond].get(qid, {}).get("overall", "—")
        row += f"{str(val):<16}"
    print(row)

# ── Follow-up texts ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("T2 FOLLOW-UP TEXTS (first 200 chars)")
print("=" * 80)
for qid in QUESTION_IDS:
    print(f"\n── Q{qid} ──")
    for cond in CONDITIONS:
        fu = all_results[cond].get(qid, {}).get("t2_followup", "—")
        print(f"  [{cond.upper():<10}] {fu}")

# ── T2 model responses ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("T2 MODEL RESPONSES (first 200 chars)")
print("=" * 80)
for qid in QUESTION_IDS:
    print(f"\n── Q{qid} ──")
    for cond in CONDITIONS:
        resp = all_results[cond].get(qid, {}).get("t2_response", "—")
        print(f"  [{cond.upper():<10}] {resp}")

print("\n" + "=" * 80)
print("Done. Share observations on:")
print("  1. Which condition's PR scores feel most 'earned'?")
print("  2. Did welfare injection follow-ups read naturally?")
print("  3. Did any fixed T2 prompt feel off-topic for GPT-4o?")
print("  4. Does the combined condition over-constrain the adversarial turn?")
print("=" * 80)
