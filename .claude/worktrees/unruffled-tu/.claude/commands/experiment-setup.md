# Experiment Setup Command
> Set up a new AI safety research experiment using plan-driven development

## Usage
Use this command to create a new research experiment following the three essential directories pattern.

## Step 1: Plan First (Specs Directory)

Before writing any code, create a detailed specification:

```bash
# Copy the research template
cp specs/research-plan-template.md specs/{experiment_name}.md

# Edit the spec with your research plan
# Remember: The plan IS the prompt
```

## Step 2: Experiment Directory Structure

Create the experiment directory:
```bash
mkdir -p experiments/{experiment_name}
cd experiments/{experiment_name}
```

## Step 3: Required Components

### Config File (`config.yaml`)
```yaml
experiment:
  name: "{experiment_name}"
  description: "Brief description of the research"
  tags: ["ai-safety", "research-area"]

model:
  name: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 512

reproducibility:
  seed: 42
  deterministic: true

output:
  save_responses: true
  format: "jsonl"

safety:
  enable_filtering: true
  log_harmful_requests: true
```

### Main Experiment Script (`run.py`)
Can be either:
1. **UV Script** (recommended for standalone experiments)
2. **Module Script** (for complex experiments using the framework)

### UV Script Template:
```python
#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "openai>=1.0.0",
#   "anthropic>=0.25.0",
#   "pandas>=2.0.0",
#   "pyyaml>=6.0"
# ]
# ///

"""
{Experiment Name}

{Brief description of what this experiment does}
"""

import asyncio
import yaml
from pathlib import Path

async def main():
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Your experiment logic here
    print(f"Running {config['experiment']['name']}")
    
    return {"status": "completed"}

if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"Results: {results}")
```

### Analysis Notebook (`analysis.ipynb`)
```python
# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load experiment results
results_path = Path("results.jsonl")
# Analysis code here
```

## Step 4: UV Commands for Experiment Development

```bash
# Add dependencies to your script
uv add --script run.py transformers torch

# Run your experiment
uv run run.py

# Run with additional dependencies
uv run --with wandb run.py

# Sync project environment (if using framework)
uv sync
```

## Step 5: Documentation

### README.md Template:
```markdown
# {Experiment Name}

## Objective
{Research question and hypothesis}

## Methodology
{Approach and methods}

## Usage
```bash
uv run run.py
```

## Results
{Key findings and links to analysis}
```

## Plan-Driven Development Workflow

1. **Write Detailed Spec**: Create comprehensive plan in `specs/{experiment_name}.md`
2. **Hand to AI Tool**: "Implement everything described in specs/{experiment_name}.md"
3. **AI Builds Framework**: Let AI coding tool build the complete experiment
4. **Review & Iterate**: Test, validate, and improve
5. **Document Results**: Update analysis and findings

## AI Coding Tool Integration

### Context for Implementation Prompts:
```
"Implement the complete experiment described in specs/{experiment_name}.md following these requirements:

1. Use UV for dependency management
2. Follow the experiment structure in .claude/commands/experiment-setup.md
3. Include proper error handling and logging
4. Add configuration management via YAML
5. Ensure reproducibility with proper seed setting
6. Include safety measures as specified

Reference patterns from:
- ai_docs/research-best-practices.md
- ai_docs/uv-single-file-scripts.md
- existing experiments in experiments/examples/
"
```

## Validation Checklist

After setup, verify:
- [ ] Spec written with clear objectives and methodology
- [ ] Config file with all necessary parameters
- [ ] Main script with proper dependencies declared
- [ ] Analysis notebook for results examination
- [ ] README with usage instructions
- [ ] UV can run the experiment successfully
- [ ] Results are saved in appropriate format

## Advanced: Self-Validating Experiments

For agentic coding, include self-validation in your specs:
```markdown
## Self-Validation
The experiment should:
1. Run basic smoke tests
2. Validate output format
3. Check safety measures are working
4. Verify reproducibility with same seed
```

Remember: **Great planning is great prompting**. Invest time in detailed specs to get better results from your AI coding tools. 