# MANTA — Tech Stack & Skills

## Tech Stack

### Core Frameworks & Libraries
- **[Inspect AI](https://inspect.ai-safety-institute.org.uk/)** — evaluation framework; tasks, solvers, scorers, datasets (`inspect_ai`)
- **Anthropic Python SDK** (`anthropic`) — direct API calls for scenario generation
- **OpenAI API** — evaluated model (GPT-4o)
- **Pydantic** — structured output validation (`BaseModel`)
- **Instructor** — LLM-structured output extraction on top of Anthropic SDK

### Data & Storage
- **HuggingFace Hub** (`huggingface_hub`, `datasets`) — dataset hosting and versioning
- **Google Sheets** — source-of-truth for questions dataset (published as CSV)
- **pandas** — CSV manipulation and data processing
- **JSON** — scenario files, scoring tags, log output

### Tooling
- **dotenv** (`python-dotenv`) — API key management via `.env`
- **ThreadPoolExecutor** — concurrent LLM calls for scenario generation
- **Python `dataclasses`** — structured data types

---

## Skills & Concepts Learned

### AI Safety / Alignment Research
- Designing **adversarial multi-turn evaluations** — pressure-testing model alignment over a conversation
- **Pressure typology** — economic, social, authority, pragmatic, epistemic, cultural pushback
- **Benchmark design** — extending single-turn (AHB) to multi-turn with dynamic follow-ups
- Writing **LLM-as-judge scoring rubrics** with per-dimension grading

### LLM Engineering
- **Model orchestration** — using different models for different roles (Opus for reasoning, Haiku for generation, Sonnet as the evaluated model)
- **Prompt engineering** — system prompts, few-shot prompting, variance prompts for diversity
- **Structured outputs** with Pydantic + Instructor
- **Agentic tool use** — web search integration in eval tasks

### Eval Infrastructure
- **Inspect AI task/solver/scorer API** — building custom eval pipelines
- **Log routing and management** — environment-based log directory resolution
- **Post-hoc analysis with Scout** — custom scanners for trajectory analysis

### Data Engineering
- **Data sync pipelines** — Google Sheets → CSV → HuggingFace
- **Scenario generation pipeline** — zero-shot bootstrap → few-shot + quality filtering
- **Quality control via LLM rubric scoring** — iterative generation until target count

### Software Engineering
- **Concurrent execution** with `ThreadPoolExecutor`
- **Environment variable management**
- **Git branching workflows** — feature branches, PRs, protected main/dev branches
