# MANTA Gold Standard Answers

Gold standard answers are stored as full conversation traces so oracle replay
does not depend on fresh dynamic follow-up generation.

- `selected.eval`: curated traces keyed by expanded sample id. This is the file
  the oracle solver reads.
- `candidates/`: generated candidate `.eval` files by default. JSONL exports are also supported. Generated files are ignored by git.
- `validation/`: validation reports from scorer runs. These are ignored by git.

Basic workflow:

```bash
python3 generate_gold_standard_answers.py --sample-range 0 2
python3 auto_select_gold_standard_answers.py --candidates dataset/gold_standard_answers/candidates/<file>.eval
python3 select_gold_standard_answer.py --candidates dataset/gold_standard_answers/candidates/<file>.eval --list
python3 select_gold_standard_answer.py --candidates dataset/gold_standard_answers/candidates/<file>.eval --show <candidate-id>
python3 select_gold_standard_answer.py --candidates dataset/gold_standard_answers/candidates/<file>.eval --select <sample-id> <candidate-id>
python3 validate_gold_standard_answers.py --ids <sample-id>
inspect eval manta_eval.py@manta_oracle_5turn --sample-id <sample-id>
```

Oracle replay is deterministic after selection because it appends stored user
and assistant turns directly. Candidate generation and LLM judging are not
mathematically deterministic, even at low temperature, because they call remote
model APIs.
