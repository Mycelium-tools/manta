# AI Safety Research Template - Project Overview

> Teaching AI safety researchers how to fully leverage AI coding tools

## Purpose

This template is designed to teach AI safety researchers how to **master AI coding tools** like Claude Code, Cursor, and Windsurf. The goal is not just to build research tools, but to learn the **three essential directories pattern** that unlocks your AI coding tool's superpowers.

## Core Philosophy: Plan-Driven Development

Traditional AI coding: You prompt back and forth iteratively
**Our approach**: You write comprehensive plans that become prompts

**Key insight**: Great planning = great prompting

## The Three Essential Directories Pattern

### 🤖 `ai_docs/` - Persistent Memory for AI Tools
- **Purpose**: Documentation your AI tool can instantly access
- **Contents**: Third-party API docs, patterns, integration notes
- **Why**: AI tools read files 100x faster than humans read READMEs
- **Example**: Instead of explaining OpenAI API each session, store docs here

### 📋 `specs/` - Plans That Scale Your Compute  
- **Purpose**: Detailed specifications that become prompts
- **Contents**: Research plans, feature specifications, implementation details
- **Why**: The plan IS the prompt - comprehensive planning yields better results
- **Example**: Write complete experiment spec, hand to AI tool for implementation

### ⚡ `.claude/` - Reusable Prompts & Workflows
- **Purpose**: Tested prompts you can reuse across sessions
- **Contents**: Context priming, experiment setup, debugging workflows
- **Why**: Don't waste tokens re-explaining your codebase every session
- **Example**: `/prime` command instantly loads project context

## Technology Stack

### Modern Python with UV
- **UV**: Fast, reliable Python package management (10-100x faster than pip)
- **Single-file scripts**: Dependencies declared inline with scripts
- **Ruff**: Modern linting and formatting (replaces Black/isort/flake8)

### AI Safety Research Focus
- **Core Areas**: Interpretability, alignment, safety evaluation, jailbreaking
- **Framework**: Async-first base experiment class with safety features
- **APIs**: OpenAI, Anthropic with built-in safety filtering and rate limiting

### Development Philosophy  
- **Context is King**: AI tools can only build what they can see
- **Stay Focused**: Too much context is as bad as too little
- **Reuse Everything**: Build workflows you can repeat
- **Scale Through Planning**: Investment in specs pays exponential dividends

## Workflow Example

### Traditional Approach (Inefficient)
1. Prompt: "Help me build a safety evaluation"
2. AI asks clarifying questions
3. Back and forth for 10+ messages
4. Still missing key details

### Our Approach (Efficient)
1. Write `specs/safety-evaluation.md` with complete plan
2. Prompt: "Implement everything in specs/safety-evaluation.md"
3. AI builds complete, working experiment
4. Review and iterate on results, not implementation

## Success Metrics

You know you've mastered AI coding tools when:
- [ ] Context priming takes <30 seconds for any codebase
- [ ] Your specs generate working code in one shot
- [ ] You spend more time reviewing than prompting
- [ ] Your AI tool understands your codebase better than you do
- [ ] You're building features faster than ever before

## Key Files to Understand

### Framework Core
- `src/core/base_experiment.py` - Base class for all research experiments
- `src/utils/inference.py` - LLM API wrapper with safety features
- `src/utils/experiment_utils.py` - Common research utilities

### AI Coding Integration
- `.claude/commands/research-prime.md` - Context loading command
- `.claude/commands/experiment-setup.md` - New experiment workflow
- `ai_docs/research-best-practices.md` - AI safety research guidelines
- `ai_docs/uv-single-file-scripts.md` - UV scripting patterns

### Examples
- `experiments/examples/hello_world/` - Basic framework demonstration
- `experiments/examples/safety_eval/` - Safety evaluation pipeline

## Research Areas Enabled

1. **Safety Evaluation**: Automated testing of model safety
2. **Interpretability**: Attention analysis, mechanistic interpretability  
3. **Alignment**: Constitutional AI, RLHF, value learning
4. **Jailbreaking**: Prompt-based attacks and defenses
5. **Robustness**: Distribution shift and adversarial testing

## UV Integration Benefits

- **Fast dependency resolution**: No waiting for environment setup
- **Single-file experiments**: Self-contained research scripts
- **Reproducible environments**: Consistent dependencies across machines
- **Modern tooling**: Built for the AI coding era

## Agentic vs AI Coding

**AI Coding (old way)**: Human-driven iteration with AI assistance
**Agentic Coding (new way)**: AI-driven implementation from human plans

This template teaches agentic coding:
- AI tools handle entire development lifecycles
- Humans focus on curation, planning, and validation
- Self-validating loops reduce manual intervention

## Learning Path

### Week 1: Foundation
- Master the three essential directories
- Learn context priming workflows  
- Run example experiments with UV

### Week 2: Plan-Driven Development
- Write detailed research specifications
- Practice handing complete plans to AI tools
- Build experiments from specs alone

### Week 3: Scale and Optimize
- Create custom reusable prompts
- Build agentic workflows with self-validation
- Optimize through better planning

## Template Evolution

This template is designed to evolve with the AI coding community:
- Patterns that work get added to `ai_docs/`
- Successful research specs get shared in `specs/examples/`
- Reusable workflows get added to `.claude/commands/`

The goal is building a comprehensive resource for AI safety researchers learning to leverage AI coding tools effectively.

---

**Ready to unlock your AI coding tool's superpowers?**
Start with context priming, plan with specs, and scale with reusable workflows. 