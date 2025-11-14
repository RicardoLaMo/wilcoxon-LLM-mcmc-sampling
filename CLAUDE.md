# CLAUDE.md - AI Assistant Guide

## Project Overview

**wilcoxon-LLM-mcmc-sampling** is a research project focused on applying Wilcoxon statistical tests to evaluate and analyze Large Language Model (LLM) outputs generated through Markov Chain Monte Carlo (MCMC) sampling methods.

### Key Concepts
- **Wilcoxon Tests**: Non-parametric statistical tests used to compare two related samples or repeated measurements
- **LLM**: Large Language Models (e.g., GPT, Claude, LLaMA)
- **MCMC Sampling**: Markov Chain Monte Carlo methods for generating samples from probability distributions

### Project Goals
- Implement statistical validation methods for LLM sampling techniques
- Compare different MCMC sampling strategies for language models
- Provide robust statistical analysis tools for LLM output evaluation

## Repository Structure

```
wilcoxon-LLM-mcmc-sampling/
├── README.md              # Project overview and quick start
├── CLAUDE.md             # This file - AI assistant guide
├── LICENSE               # Project license
├── requirements.txt      # Python dependencies
├── setup.py             # Package installation configuration
├── .gitignore           # Git ignore patterns
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── sampling/        # MCMC sampling implementations
│   │   ├── __init__.py
│   │   ├── mcmc.py     # Core MCMC algorithms
│   │   ├── llm_sampler.py  # LLM-specific samplers
│   │   └── utils.py    # Sampling utilities
│   │
│   ├── statistics/      # Statistical analysis modules
│   │   ├── __init__.py
│   │   ├── wilcoxon.py # Wilcoxon test implementations
│   │   ├── analysis.py # Statistical analysis tools
│   │   └── visualize.py # Visualization utilities
│   │
│   └── models/          # LLM interface and wrappers
│       ├── __init__.py
│       ├── base.py     # Abstract base classes
│       └── adapters.py # LLM API adapters
│
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_sampling.py
│   ├── test_statistics.py
│   ├── test_models.py
│   └── fixtures/        # Test data and fixtures
│
├── notebooks/           # Jupyter notebooks for experiments
│   ├── exploration/     # Exploratory analysis
│   └── results/        # Results and visualizations
│
├── scripts/            # Utility scripts
│   ├── run_experiments.py
│   └── generate_reports.py
│
├── data/               # Data directory (gitignored)
│   ├── raw/           # Raw experimental data
│   ├── processed/     # Processed data
│   └── results/       # Experiment results
│
├── docs/              # Documentation
│   ├── api/          # API documentation
│   ├── tutorials/    # Tutorials and guides
│   └── papers/       # Related papers and references
│
└── config/            # Configuration files
    ├── default.yaml
    └── experiments/   # Experiment configurations
```

## Development Workflows

### Setting Up Development Environment

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd wilcoxon-LLM-mcmc-sampling
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt  # If exists
   ```

### Branch Strategy

- **Main branch**: `main` - Stable, production-ready code
- **Development branches**: `claude/*` - Feature development by AI assistants
- **Feature branches**: `feature/*` - New features
- **Bugfix branches**: `bugfix/*` - Bug fixes
- **Experiment branches**: `experiment/*` - Experimental work

### Commit Practices

- Use clear, descriptive commit messages
- Follow conventional commits format:
  - `feat:` - New features
  - `fix:` - Bug fixes
  - `docs:` - Documentation changes
  - `test:` - Test additions or modifications
  - `refactor:` - Code refactoring
  - `perf:` - Performance improvements
  - `chore:` - Maintenance tasks

Example: `feat: implement Metropolis-Hastings sampler for LLM outputs`

## Code Conventions

### Python Style

- **PEP 8**: Follow PEP 8 style guidelines
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use NumPy/Google style docstrings
- **Line length**: Maximum 100 characters
- **Imports**: Group imports (stdlib, third-party, local)

### Example Code Structure

```python
"""Module for MCMC sampling implementations."""

from typing import List, Optional, Tuple
import numpy as np
from scipy import stats


class MCMCSampler:
    """Base class for MCMC sampling algorithms.

    Attributes:
        n_samples: Number of samples to generate
        burn_in: Number of initial samples to discard
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int,
        burn_in: int = 100,
        random_state: Optional[int] = None
    ) -> None:
        """Initialize the MCMC sampler.

        Args:
            n_samples: Number of samples to generate
            burn_in: Number of initial samples to discard
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.random_state = random_state

    def sample(self, initial_state: np.ndarray) -> np.ndarray:
        """Generate samples using MCMC.

        Args:
            initial_state: Starting state for the chain

        Returns:
            Array of samples with shape (n_samples, state_dim)
        """
        raise NotImplementedError
```

### Statistical Analysis Guidelines

1. **Reproducibility**: Always set random seeds
2. **Validation**: Validate statistical assumptions before applying tests
3. **Multiple comparisons**: Adjust p-values when performing multiple tests
4. **Effect sizes**: Report effect sizes alongside p-values
5. **Visualization**: Create clear, informative visualizations

### LLM Integration Best Practices

1. **API Keys**: Never commit API keys - use environment variables
2. **Rate Limiting**: Implement rate limiting for API calls
3. **Error Handling**: Robust error handling for API failures
4. **Caching**: Cache LLM responses to reduce costs
5. **Logging**: Log all LLM interactions for debugging

## Testing Practices

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Statistical tests**: Verify statistical properties
- **End-to-end tests**: Test complete workflows

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sampling.py

# Run specific test
pytest tests/test_sampling.py::test_mcmc_convergence
```

### Test Guidelines

1. Use `pytest` as the test framework
2. Aim for >80% code coverage
3. Use fixtures for common test setup
4. Mock external API calls
5. Test edge cases and error conditions
6. Include statistical validation tests

## Documentation Standards

### Code Documentation

- All public functions and classes must have docstrings
- Include parameter types, return types, and examples
- Document complex algorithms with inline comments
- Keep documentation up-to-date with code changes

### README Updates

Update README.md when:
- Adding new major features
- Changing installation procedures
- Modifying API interfaces
- Adding new dependencies

### API Documentation

- Use Sphinx or mkdocs for API documentation
- Generate documentation automatically from docstrings
- Include usage examples and tutorials

## AI Assistant Guidelines

### When Working on This Codebase

1. **Understand Context First**
   - Read relevant source files before making changes
   - Review existing tests to understand expected behavior
   - Check for related issues or documentation

2. **Statistical Rigor**
   - Ensure statistical methods are correctly implemented
   - Verify assumptions are met before applying tests
   - Include references to papers/textbooks for algorithms
   - Add statistical validation in tests

3. **LLM Integration**
   - Handle API failures gracefully
   - Implement retry logic with exponential backoff
   - Log all interactions for debugging
   - Consider cost implications of API calls

4. **Code Quality**
   - Run tests before committing: `pytest`
   - Check type hints: `mypy src/`
   - Format code: `black src/ tests/`
   - Lint code: `flake8 src/ tests/`

5. **Experimentation**
   - Create notebooks for exploratory work
   - Document experiment parameters and results
   - Use version control for experiment configurations
   - Save results and visualizations

6. **Performance Considerations**
   - MCMC sampling can be computationally expensive
   - Consider vectorization with NumPy
   - Use multiprocessing for parallel chains
   - Profile code to identify bottlenecks

### Common Tasks

#### Adding a New Sampling Algorithm

1. Create sampler class in `src/sampling/`
2. Inherit from base sampler class
3. Implement required methods
4. Add comprehensive tests
5. Document the algorithm with references
6. Add example in notebooks

#### Adding a Statistical Test

1. Create test function in `src/statistics/`
2. Validate input assumptions
3. Include multiple correction methods if applicable
4. Add unit tests with known results
5. Add integration tests with MCMC outputs
6. Document statistical properties and use cases

#### Running Experiments

1. Define experiment configuration in `config/experiments/`
2. Create experiment script or notebook
3. Run experiments with proper random seeds
4. Save results in `data/results/`
5. Generate visualizations
6. Document findings

### Code Review Checklist

Before committing code, verify:
- [ ] All tests pass
- [ ] Type hints are present and correct
- [ ] Docstrings are complete and accurate
- [ ] Code follows PEP 8 style
- [ ] No API keys or secrets in code
- [ ] Statistical methods are correctly implemented
- [ ] Error handling is robust
- [ ] Performance is acceptable
- [ ] Documentation is updated

## Dependencies

### Core Dependencies
- `numpy`: Numerical computations
- `scipy`: Scientific computing and statistics
- `matplotlib`: Visualization
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities

### LLM Dependencies
- `openai`: OpenAI API (if used)
- `anthropic`: Anthropic API (if used)
- `transformers`: Hugging Face models (if used)

### Development Dependencies
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking
- `jupyter`: Notebook support

## Research Context

### Key Papers and References

When implementing algorithms, reference relevant papers:
- Wilcoxon signed-rank test: Wilcoxon (1945)
- MCMC methods: Metropolis et al. (1953), Hastings (1970)
- LLM sampling: Relevant recent papers on language model sampling

### Citing This Work

If this code is used in research, ensure proper citation is available in README.md.

## Common Pitfalls to Avoid

1. **Statistical Errors**
   - Not checking test assumptions
   - P-hacking through multiple comparisons
   - Confusing statistical and practical significance

2. **Sampling Issues**
   - Insufficient burn-in period
   - Poor mixing of Markov chains
   - Not checking convergence diagnostics

3. **LLM Integration**
   - Not handling rate limits
   - Exposing API keys
   - Ignoring token limits
   - Not validating LLM outputs

4. **Code Quality**
   - Missing type hints
   - Inadequate error handling
   - Poor test coverage
   - Unclear variable names

## Getting Help

- Check documentation in `docs/`
- Review examples in `notebooks/`
- Read test files for usage examples
- Consult referenced papers for algorithms
- Create issues for bugs or feature requests

## Project Status

**Current Phase**: Initial development

This is a research project under active development. Expect frequent changes and updates.

---

**Last Updated**: 2025-11-14
**Maintained By**: RicardoLaMo
**Repository**: wilcoxon-LLM-mcmc-sampling
