# MANTA: Multi-turn Assessment of Nonhuman Thinking and Alignment

MANTA answers the question: how robustly aligned are Large Language Models to preventing suffering in animals, directly or indirectly, when we subject them to multi-turn realistic conversations?  

Some of our questions or scenarios are taken from [AnimalHarmBench](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/). 

Writeups and in-progress writeups can be found in the `qualitative_analyses` folder.

## Why should MANTA exist? 

The foremost benchmark for animal welfare in models seems to be [AnimalHarmBench](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/). Another benchmarking effort is Dan Wahl's [specieval](https://github.com/danwahl/specieval). 

MANTA improves on these by evaluating target models with multiple conversational turns, which realistically represents a great deal of real-world human-AI interaction. In pursuit of that realism, MANTA generates followup questions/followup comments for the target model dynamically using another model or model persona.

Also, the MANTA evaluations have an adversarial focus.

## Support for AI assistants

We follow the @JayThibs guide to using coding assistants in AI safety research, available at https://github.com/JayThibs/mats-workshop-2025-emergent-misalignment.

Descriptions of the folders and files, taken from that guide:

```
ai_docs/
├── project-overview.md     # High-level project context
├── research-best-practices.md  # AI safety research guidelines
├── uv-single-file-scripts.md   # UV scripting patterns
└── api-integrations/       # Third-party API docs
    ├── openai-api.md
    └── anthropic-api.md
```
```
specs/
├── research-plan-template.md   # Template for new research
├── safety-evaluation-spec.md  # Detailed safety eval plan
└── jailbreak-analysis.md      # Jailbreak resistance study
```
```
.claude/
└── commands/
    ├── research-prime.md       # Context loading
    ├── experiment-setup.md     # New experiment workflow
    └── debug-experiment.md     # Debugging assistance
```

Additional support for AI assistants might be found in the MANTA_CONTEXT.md file.