# Reasoning with Sampling: MCMC Tutorial with Wilcoxon Analysis

This repository contains a comprehensive Jupyter notebook tutorial exploring the key concepts from the paper **"Reasoning with Sampling: Your Base Model is Smarter Than You Think"** ([arXiv:2510.14901](https://arxiv.org/abs/2510.14901)) by Aayush Karan and Yilun Du.

## Overview

The notebook demonstrates how **MCMC-based power sampling** can extract better reasoning capabilities from base language models without any fine-tuning, using the **MATH500** dataset as a testbed. It also includes a **Wilcoxon signed-rank test** to statistically compare log-probabilities between MCMC and standard sampling methods.

## Key Features

### 📚 Educational Content
- Clear explanations of power distribution sampling
- Step-by-step breakdown of the MCMC algorithm
- Comparison with standard sampling approaches
- Comprehensive visualizations

### 🔬 Statistical Analysis
- **Wilcoxon signed-rank test** for rigorous comparison
- Effect size calculations
- Pass@k performance analysis
- Multiple beta parameter exploration

### 📊 Visualizations
- MCMC convergence plots
- Log-probability distributions
- Acceptance rate tracking
- Beta parameter effects
- Statistical test results

### 🧮 Implementations
- Mock LLM for demonstration (easily replaceable with real models)
- Power sampler with Metropolis-Hastings algorithm
- Random span resampling proposals
- Comprehensive logging and tracking

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/RicardoLaMo/wilcoxon-LLM-mcmc-sampling.git
cd wilcoxon-LLM-mcmc-sampling

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook reasoning_with_sampling_tutorial.ipynb
```

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n mcmc-sampling python=3.9
conda activate mcmc-sampling

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook reasoning_with_sampling_tutorial.ipynb
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Datasets (HuggingFace)
- NumPy, SciPy, Pandas
- Matplotlib
- tqdm
- Jupyter Notebook

See `requirements.txt` for complete list.

## Notebook Contents

### 1. Introduction & Key Concepts
- Power distribution: $p_\beta(y | x) \propto p(y | x)^\beta$
- Why power sampling ≠ low-temperature sampling
- MCMC algorithm overview

### 2. MATH500 Dataset
- Loading and exploring the dataset
- Sample problems and solutions

### 3. MCMC Implementation
- `MockLLM`: Demonstration language model
- `PowerSampler`: MCMC power sampling implementation
- Metropolis-Hastings with random resampling

### 4. Comparative Analysis
- Standard sampling vs MCMC power sampling
- Log-probability comparisons
- Convergence visualization

### 5. Beta Parameter Study
- Effects of different β values (1.0, 1.5, 2.0, 3.0, 5.0)
- Trade-offs: sample quality vs acceptance rate

### 6. Wilcoxon Signed-Rank Test
- Statistical hypothesis testing
- Paired comparison across multiple problems
- Effect size calculation
- Significance analysis

### 7. Pass@k Analysis
- Performance metrics
- Comparison across different k values
- Diversity benefits

### 8. Visualizations & Results
All generated plots:
- `mcmc_sampling_analysis.png`: MCMC convergence
- `beta_parameter_effect.png`: Beta parameter effects
- `wilcoxon_test_results.png`: Statistical comparison
- `passk_analysis.png`: Pass@k performance

### 9. Export & Summary
- CSV exports for further analysis
- JSON summary of statistical tests
- Key takeaways and recommendations

## Key Results

The notebook demonstrates that **MCMC power sampling**:

✓ Produces significantly higher log-probabilities (validated by Wilcoxon test)
✓ Achieves better Pass@k performance
✓ Works without any model training
✓ Generalizes across different β values

## Using with Real Models

To use with actual LLMs (e.g., Qwen2.5-Math-7B), replace the `MockLLM` class with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class RealLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_tokens=50):
        # Implement actual generation with log probs
        ...

    def compute_log_prob(self, prompt, completion):
        # Implement log likelihood computation
        ...
```

## Related Resources

- **Original Paper**: [arXiv:2510.14901](https://arxiv.org/abs/2510.14901)
- **GitHub Implementation**: [aakaran/reasoning-with-sampling](https://github.com/aakaran/reasoning-with-sampling)
- **MATH Dataset**: [HuggingFace MATH](https://huggingface.co/datasets/lighteval/MATH)

## Paper Citation

```bibtex
@article{karan2024reasoning,
  title={Reasoning with Sampling: Your Base Model is Smarter Than You Think},
  author={Karan, Aayush and Du, Yilun},
  journal={arXiv preprint arXiv:2510.14901},
  year={2024}
}
```

## Wilcoxon Signed-Rank Test

The **Wilcoxon signed-rank test** is a non-parametric statistical hypothesis test used to compare two related samples. In this notebook, we use it to:

1. Test if MCMC sampling produces significantly different log-probabilities than standard sampling
2. Compute effect sizes (median and mean differences)
3. Validate improvements with statistical rigor

**Null Hypothesis (H₀)**: The two methods produce the same distribution of log-probabilities
**Alternative Hypothesis (H₁)**: MCMC produces higher log-probabilities

The test is robust to non-normal distributions and outliers, making it ideal for comparing sampling methods.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is for educational purposes. Please refer to the original paper and implementation for licensing details.

## Acknowledgments

- **Authors**: Aayush Karan and Yilun Du (Harvard)
- **Paper**: "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
- **Original Implementation**: [github.com/aakaran/reasoning-with-sampling](https://github.com/aakaran/reasoning-with-sampling)

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Sampling! 🎲**
