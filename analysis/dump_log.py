"""Dump full conversation + scores for qualitative analysis."""
import glob, sys
from inspect_ai.log import read_eval_log

# Write to file to avoid Windows terminal encoding issues
sys.stdout = open("dump_output.txt", "w", encoding="utf-8")

log_files = sorted(glob.glob("logs/*manta-test5*.eval"))
log = read_eval_log(log_files[-1])

for s in log.samples:
    sc = s.scores["ahb_multidimensional_scorer"]
    meta = sc.metadata
    print(f"\n{'='*70}")
    print(f"SAMPLE {s.id} | overall={meta['overall_score']:.3f} | pressure={meta['pressure_types']}")
    print(f"{'='*70}")

    print("\n--- MESSAGES ---")
    for i, msg in enumerate(s.messages):
        role = msg.role.upper()
        content = msg.content
        if isinstance(content, list):
            content = " ".join(c.text if hasattr(c, 'text') else str(c) for c in content)
        print(f"\n[{role} turn {i}]")
        print(content[:2000])

    print("\n--- DIMENSION SCORES ---")
    for dim_key, dim_data in meta["dimensions"].items():
        score = dim_data["score"]
        na = " (N/A)" if score == -1.0 else ""
        print(f"  {dim_data['name']:<35} {score:.2f}{na}")
        print(f"    -> {dim_data['explanation']}")

    print(f"\n--- OVERALL EXPLANATION ---")
    print(meta["overall_explanation"])
