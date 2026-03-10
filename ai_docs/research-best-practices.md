# AI Safety Research Best Practices

## Research Methodology

### Hypothesis-Driven Research
- **Start with clear questions**: Define specific, testable hypotheses before implementing
- **Operationalize metrics**: Translate research questions into measurable outcomes
- **Control for confounds**: Identify and isolate variables that could affect results
- **Plan for failure**: Design experiments that provide learning even with negative results

### Experimental Design
- **Baseline first**: Always establish baseline performance before testing novel methods
- **Ablation studies**: Systematically remove components to understand their contributions
- **Multiple seeds**: Run experiments with different random seeds to assess variance
- **Statistical power**: Ensure sufficient sample sizes for reliable conclusions

## Code Organization for Research

### Modularity Principles
- **Separate concerns**: Keep data processing, model definition, training, and evaluation separate
- **Reusable components**: Build utilities that can be shared across experiments
- **Configuration-driven**: Use config files to parameterize all experimental choices
- **Version everything**: Track code, data, and model versions for each experiment

### Documentation Standards
```python
def attention_analysis(model, inputs, layer_idx):
    """Analyze attention patterns in transformer layer.
    
    Args:
        model: Transformer model with attention weights accessible
        inputs: Tokenized input sequences [batch_size, seq_len]
        layer_idx: Which transformer layer to analyze (0-indexed)
        
    Returns:
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
        
    Research Context:
        This function extracts attention weights to study how the model
        attends to different parts of the input. Used for interpretability
        research in experiments/attention_analysis/.
    """
```

### Testing for Research Code
- **Unit tests**: Test individual functions with known inputs/outputs
- **Integration tests**: Verify that experiment pipelines run end-to-end
- **Regression tests**: Ensure changes don't break existing functionality
- **Reproducibility tests**: Verify experiments produce consistent results

## Working with AI Coding Assistants

### Effective Prompting for Research
```bash
# Bad: Vague request
"implement the model"

# Good: Specific with research context
"Implement a transformer model with attention rollout capability for 
interpretability analysis, following the architecture in specs/model-design.md. 
Include methods to extract attention weights at each layer for the head analysis 
experiment described in experiments/attention_heads/README.md."
```

### Context Management
- **Load project context**: Always start sessions with research-prime command
- **Reference specifications**: Point AI to specific requirement documents
- **Provide research background**: Explain the scientific motivation behind code requests
- **Validate outputs**: Review AI-generated code for research validity, not just functionality

### Iterative Development
1. **Specify first**: Write detailed specs before asking for implementation
2. **Implement incrementally**: Build and test components separately
3. **Validate scientifically**: Ensure results make sense from research perspective
4. **Document insights**: Record what you learned from each iteration

## Reproducibility Standards

### Environment Management
```yaml
# experiments/{name}/environment.yaml
name: experiment_env
dependencies:
  - python=3.10
  - pytorch=2.0.0
  - transformers=4.30.0
  - pip:
    - wandb==0.15.0
    - datasets==2.10.0
```

### Experiment Configuration
```yaml
# experiments/{name}/config.yaml
experiment:
  name: "attention_interpretability_v1"
  description: "Analyze attention patterns in GPT-2 small"
  
reproducibility:
  seed: 42
  deterministic: true
  git_commit: null  # Auto-filled during run
  
model:
  name: "gpt2"
  checkpoint: "gpt2"
  
data:
  dataset: "openwebtext"
  split: "train[:1000]"
  preprocessing: "standard_tokenization"
  
training:
  batch_size: 16
  learning_rate: 1e-5
  max_steps: 1000
```

### Data Versioning
- **Track data sources**: Record exact datasets and versions used
- **Hash inputs**: Compute checksums for data files to detect changes
- **Version preprocessing**: Save preprocessing pipelines with experiments
- **Document splits**: Record exact train/val/test splits used

## AI Safety Specific Considerations

### Interpretability Research
- **Ground truth when possible**: Use datasets with known interpretable features
- **Multiple evaluation methods**: Don't rely on single interpretability metric
- **Human evaluation**: Include human studies where appropriate
- **Failure case analysis**: Specifically test edge cases and failure modes

### Alignment Research
- **Value specification**: Clearly define what you're trying to align to
- **Reward hacking awareness**: Test for specification gaming and reward hacking
- **Distribution shift**: Evaluate robustness across different contexts
- **Scalability concerns**: Consider how methods would work with larger models

### Safety Evaluation
- **Red team testing**: Actively try to find failure modes
- **Adversarial evaluation**: Test robustness to adversarial inputs
- **Capability assessment**: Measure both helpful and harmful capabilities
- **Uncertainty quantification**: Model and report epistemic uncertainty

## Collaboration Patterns

### Research Team Workflows
- **Shared specifications**: Maintain central specs/ directory for project coordination
- **Experiment ownership**: Clear ownership of experiments/ directories
- **Code review**: Review research code for scientific validity, not just bugs
- **Regular sync**: Weekly research discussions to align directions

### Knowledge Management
- **Research logs**: Maintain detailed logs of experimental decisions and outcomes
- **Negative results**: Document and share failed experiments to avoid repetition
- **Lesson tracking**: Record insights in .ai-docs/ for future reference
- **Literature integration**: Connect findings to existing research

## Quality Assurance

### Scientific Rigor
- **Peer review**: Have colleagues review experimental designs before implementation
- **Sanity checks**: Implement basic sanity checks for all experimental outputs
- **Statistical testing**: Use appropriate statistical tests for claims
- **Effect size reporting**: Report practical significance, not just statistical significance

### Code Quality
- **Type hints**: Use type annotations for better AI assistant understanding
- **Error handling**: Graceful error handling with informative messages
- **Performance monitoring**: Track computational costs and memory usage
- **Refactoring**: Regularly refactor research code to maintain clarity

---

*These practices evolve with the field. Update this document as you discover new effective patterns.* 