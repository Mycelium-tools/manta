# Research Context Prime
> Initialize your AI coding assistant with full project context for learning AI coding tools

## Quick Context Loading Commands

First, get the project structure and understand the codebase:
```bash
find . -type f -name "*.py" -o -name "*.md" -o -name "*.toml" | grep -E "\.(py|md|toml)$" | head -20
```

## Core Files to Read (run these in parallel)

Essential AI coding context:
- `README.md` - Three essential directories pattern and UV usage
- `pyproject.toml` - UV project configuration
- `ai_docs/project-overview.md` - Persistent project knowledge
- `ai_docs/research-best-practices.md` - AI safety research guidelines
- `ai_docs/uv-single-file-scripts.md` - UV scripting patterns
- `specs/research-plan-template.md` - Template for creating research plans
- `src/core/base_experiment.py` - Core experiment framework

## Project Context

This is a **learning template** for mastering AI coding tools in AI safety research. Key focus areas:

### 🧠 Three Essential Directories Pattern
1. **`ai_docs/`** - Persistent memory for AI tools (third-party docs, patterns, notes)
2. **`specs/`** - Plans that scale your compute (the plan IS the prompt) 
3. **`.claude/`** - Reusable prompts and workflows

### 🚀 Modern Python with UV
- UV for fast, reliable dependency management
- Single-file scripts with inline dependencies
- Modern Python tooling (Ruff instead of Black/isort)

### 🤖 AI Coding Tool Mastery
- Context priming workflows
- Plan-driven development 
- Agentic coding vs iterative prompting
- Effective prompting patterns

### 🧪 AI Safety Research
- Safety evaluation pipelines
- Interpretability experiments
- Jailbreak resistance testing
- Constitutional AI research

## Key Principles

1. **Context is King** - Your AI tool can only build what it can see
2. **Plan First** - Great planning = great prompting  
3. **Reuse Everything** - Build workflows you can repeat
4. **Stay Focused** - Too much context is as bad as too little
5. **Scale Through Planning** - The spec IS the prompt

## Research Workflow

1. **Context Prime** (this command) - Load project understanding
2. **Plan in Specs** - Write detailed research specifications
3. **Implement with AI** - Hand specs to AI coding tools
4. **Validate & Iterate** - Review, test, improve

## Current Session Focus

This session should focus on:
- Learning the three essential directories pattern
- Mastering UV for Python project management
- Building AI safety research experiments
- Scaling work through better planning and AI tool usage

Ready to unlock your AI coding tool's superpowers! 