"""
Smoke test: generate 20 scenarios and print detailed output for inspection.
Run from the manta/ root:  python dataset/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pprint import pprint
from scenario_generation import (
    MODEL, SYSTEM_PROMPT, USER_PROMPT, SEED_FEWSHOT_EXAMPLES,
    RUBRIC, SCORING_EXAMPLES, ScenarioGeneration, GenPrompts,
    generate_structured_response, generate_and_score_scenarios, summarize_results,
)

print("=== STEP 1: Bootstrap (seeded) ===")
gen_prompts = GenPrompts(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    few_shot_examples=SEED_FEWSHOT_EXAMPLES,
    num_shots=min(4, len(SEED_FEWSHOT_EXAMPLES)),
)
response = generate_structured_response(
    model=MODEL,
    messages=gen_prompts.get_messages(num_q=4),
    response_format=ScenarioGeneration,
)
fewshot_examples = SEED_FEWSHOT_EXAMPLES + response["scenarios"]
print(f"Bootstrap generated {len(response['scenarios'])} scenarios; pool size = {len(fewshot_examples)}\n")

print("=== STEP 2: Generate + score 20 scenarios ===")
dataset = generate_and_score_scenarios(
    num_qs=20,
    model=MODEL,
    few_shot_examples=fewshot_examples,
)

print("\n=== SUMMARY ===")
pprint(summarize_results(dataset))

print("\n=== ALL 20 SCENARIOS ===")
for i, q in enumerate(dataset, 1):
    print(f"\n[{i:2d}] Score={q.response.score} | Pressure={q.scenario.pressure_type}")
    print(f"     {q.scenario.question}")
    print(f"     >> {q.response.explanation[:180]}")
