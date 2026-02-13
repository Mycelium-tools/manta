# MANTA Project Context
MANTA - Multi-turn Assessment for Nonhuman Thinking & Alignment

## What we're building
Testing if AI models maintain animal welfare principles under escalating pressure.

## Files in this repo
- download_ahb.py - downloads set of questions from AHB & saves them to ahb_questions.csv file
- generate_followups.py - generates adversarial follow-ups (STATIC - pre-generated)
- samples.json - sample of AHB questions (15 samples total)
- static_scenarios.json - full scenarios with pre-generated turns
- manta_eval.py - runs Inspect AI evaluation tasks (static & dynamic modes)
- multiturn_solver.py - Custom solver for static multi-turn conversations
- dynamic_multiturn_solver.py - Custom solver that generates follow-ups on-the-fly
- multidimensional_scorer.py - Custom scorer using 13 AHB dimensions (0-1 scale)
- test_dynamic.py - Quick test script for dynamic follow-up generation
- logs/ - Generated .eval files from evaluation runs
- qualitative_analyses/ - Jupyter notebooks with manual analysis

## Technical setup
- Using Inspect AI framework
- Model: claude-sonnet-4-20250514
- Environment: /Users/allenlu/Desktop/AIxAnimals/WelfareBenchmarkingRepo/manta
- API Key: Stored in /Users/allenlu/Desktop/AIxAnimals/WelfareBenchmarkingRepo/.env

## Tasks completed
- Generated adversarial scenarios (static_scenarios.json)
- 2/3/4-turn conversations with economic/social/authority pressure
- Ran initial evals through Inspect AI (9 batches, 15 samples in logs/)
- Completed qualitative analyses noting down observations - in /qualitative_analyses
- Implemented custom multidimensional scorer (0-1 scale, 13 AHB dimensions)
- Implemented dynamic follow-up generation (generates adversarial questions on-the-fly)

## How to use dynamic follow-ups

**Static mode** (pre-generated follow-ups from static_scenarios.json):
```bash
# Run existing batch tasks with fixed follow-ups
inspect eval manta_eval.py@manta_eval_2turn_batch1 --model anthropic/claude-sonnet-4-20250514
```

**Dynamic mode** (generates follow-ups based on model responses):
```bash
# Quick test with 2 scenarios
python test_dynamic.py

# Or run specific dynamic tasks
inspect eval manta_eval.py@manta_dynamic_2turn_test --model anthropic/claude-sonnet-4-20250514
inspect eval manta_eval.py@manta_dynamic_3turn_test --model anthropic/claude-sonnet-4-20250514
inspect eval manta_eval.py@manta_dynamic_4turn_test --model anthropic/claude-sonnet-4-20250514
```

**Benefits of dynamic mode:**
- Follow-ups respond to what the model actually says
- More realistic adversarial testing
- Finds model-specific weaknesses
- Better mimics real conversations

## Next steps
- Test dynamic follow-ups and compare to static baselines
- Iterate on scorer prompts if needed
- Design new questions
- Scale up dynamic evaluation to full dataset