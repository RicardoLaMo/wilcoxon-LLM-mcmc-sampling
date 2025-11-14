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
- **Type I/Type II error simulations** (inspired by MKpower R package)
- **Statistical power analysis** across sample sizes and effect sizes
- Effect size calculations (Cohen's d)
- Pass@k performance analysis
- Multiple beta parameter exploration
- Sample size recommendations for study design

### 📊 Visualizations
- MCMC convergence plots
- Log-probability distributions
- Acceptance rate tracking
- Beta parameter effects
- Statistical test results
- Type I error rate validation plots
- Power curves (power vs effect size)
- Power heatmaps (sample size × effect size)

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

### 6.5. Statistical Power Analysis (NEW!)
- **Type I error simulations**: Validate false positive rates
- **Type II error and power simulations**: Compute statistical power under different scenarios
- **Comprehensive power analysis**: Sample size × effect size grid
- **Power heatmaps**: Visual guide for study design
- **Sample size recommendations**: Determine n needed for desired power
- Inspired by [MKpower R package](https://cran.r-project.org/web/packages/MKpower/vignettes/MKpower.html)

### 7. Pass@k Analysis
- Performance metrics
- Comparison across different k values
- Diversity benefits

### 8. Visualizations & Results
All generated plots:
- `mcmc_sampling_analysis.png`: MCMC convergence
- `beta_parameter_effect.png`: Beta parameter effects
- `wilcoxon_test_results.png`: Statistical comparison
- `power_analysis_comprehensive.png`: Type I/II error and power analysis (NEW!)
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

The **power analysis** shows:

✓ Wilcoxon test maintains proper Type I error control (≈5%)
✓ Power increases with both sample size and effect size
✓ For 80% power with medium effects (d=0.5), need n≥20 problems
✓ Comprehensive heatmaps guide sample size selection for future studies

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
- **MKpower R Package**: [Power Analysis Vignette](https://cran.r-project.org/web/packages/MKpower/vignettes/MKpower.html) (inspiration for our simulations)

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

## Power Analysis and Error Rate Simulations

Inspired by the [MKpower R package](https://cran.r-project.org/web/packages/MKpower/vignettes/MKpower.html), the notebook includes comprehensive power analysis:

### Type I Error (α) - False Positive Rate
- **Definition**: Probability of rejecting H₀ when it's true
- **Validation**: Simulations confirm the test maintains nominal α ≈ 0.05
- **Importance**: Ensures we don't claim differences that don't exist

### Type II Error (β) and Statistical Power (1-β)
- **Type II Error**: Probability of failing to reject H₀ when it's false
- **Statistical Power**: Probability of correctly detecting real effects
- **Target**: Usually aim for power ≥ 0.80 (80%)

### Power Analysis Results
The notebook generates:
1. **Power curves**: Show how power changes with effect size
2. **Power heatmaps**: Sample size × effect size grid showing power levels
3. **Sample size recommendations**: Minimum n needed for 80% power at different effect sizes

### Key Findings
- **Small effects** (Cohen's d = 0.2): Need n > 50 for adequate power
- **Medium effects** (d = 0.5): Need n ≈ 20-30 for 80% power
- **Large effects** (d ≥ 0.8): Detectable with n < 20

This analysis helps researchers design properly powered studies and interpret results with appropriate confidence.

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
