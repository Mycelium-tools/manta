# Research Project Overview

## Project Purpose
This is a template codebase designed specifically for researchers conducting AI safety research. The template optimizes for rapid experimentation, reproducibility, and effective collaboration with AI coding assistants.

## Architecture Philosophy

### Research-First Design
- **Rapid Prototyping**: Quick experiment setup and iteration
- **Hypothesis Testing**: Clear structure for testing research hypotheses  
- **Results Tracking**: Systematic logging and analysis of experimental outcomes
- **Reproducibility**: Version control for code, data, and experimental conditions

### AI-Assisted Development
- **Context Priming**: `.claude/commands/` for reusable AI assistant workflows
- **Persistent Knowledge**: `.ai-docs/` maintains project context across sessions
- **Clear Specifications**: `specs/` directory provides detailed requirements
- **Structured Prompting**: Templates optimize AI coding assistant effectiveness

## Key Components

### Core Framework (`src/core/`)
- `base_experiment.py`: Abstract base class for all experiments
- `config.py`: Configuration management and validation
- `metrics.py`: Standardized evaluation metrics for AI safety research

### Utilities (`src/utils/`)
- `visualization.py`: Publication-ready plotting and analysis
- `data_processing.py`: Common data manipulation patterns
- `model_utils.py`: Model loading, saving, and manipulation helpers

### Experiment Management (`experiments/`)
- Each experiment gets its own directory
- Standardized config.yaml for hyperparameters
- Integrated results tracking and analysis

## Research Workflow

### 1. Planning Phase
```
specs/research-plan.md → Define objectives and hypotheses
specs/experiment-design.md → Design methodology  
specs/success-criteria.md → Set evaluation metrics
```

### 2. Implementation Phase  
```
src/experiments/{name}.py → Implement experiment logic
experiments/{name}/config.yaml → Set hyperparameters
tests/test_{name}.py → Write validation tests
```

### 3. Execution Phase
```
experiments/{name}/run.py → Execute experiment
results/{name}/ → Collect outputs and logs
notebooks/{name}_analysis.ipynb → Analyze results
```

### 4. Analysis Phase
```
docs/findings/{name}.md → Document insights
specs/next-experiments.md → Plan follow-up research
```

## AI Safety Research Patterns

### Common Research Areas Supported
- **Interpretability**: Transformer attention analysis, feature visualization
- **Alignment**: Reward modeling, value learning, constitutional AI
- **Robustness**: Adversarial testing, distribution shift analysis  
- **Scalable Oversight**: AI-assisted evaluation, debate, amplification

### Experiment Types
- **Baseline Studies**: Reproducing existing results
- **Ablation Studies**: Isolating causal factors
- **Novel Methods**: Testing new approaches
- **Comparative Analysis**: Benchmarking across methods

## Technology Stack

### Core Dependencies
- **Python 3.9+**: Primary development language
- **PyTorch**: Deep learning framework
- **Transformers**: Pre-trained model access
- **Datasets**: Data loading and processing
- **Weights & Biases**: Experiment tracking

### Research Tools
- **Jupyter Notebooks**: Interactive analysis
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Linting and code quality
- **mypy**: Type checking

## Best Practices

### Code Organization
- Keep experiments modular and focused
- Use type hints for better AI assistant understanding
- Document assumptions and design decisions
- Maintain clean separation between research logic and infrastructure

### Reproducibility
- Version control all code and configurations
- Set and document random seeds
- Track compute environment and dependencies
- Save intermediate results for debugging

### Collaboration
- Write clear specifications before implementation
- Use descriptive commit messages
- Document experimental hypotheses and outcomes
- Share reusable components in src/utils/

## Integration with AI Coding Tools

### Effective Prompting Strategies
- Reference specific files and line numbers
- Provide clear context about research objectives
- Break complex tasks into smaller, testable components
- Validate AI-generated code with existing tests

### Context Management
- Use `.claude/commands/research-prime.md` at session start
- Keep `.ai-docs/` updated with project evolution
- Reference `specs/` for detailed requirements
- Maintain experiment logs for historical context

---

**Last Updated**: [Date]
**Primary Researcher**: [Name]
**Project Phase**: [Planning/Implementation/Analysis] 