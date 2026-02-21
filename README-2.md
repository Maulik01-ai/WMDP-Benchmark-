# WMDP-Benchmark
wmdp-main: contains the benchmark data and code. 
# Improving Machine Unlearning Precision for Biosecurity Knowledge in LLMs

**Course**: Responsible AI Engineering  
**Semester**: Fall 2024

## Team Members
- Zahra Rahmani - [GitHub: @zazmir] - [zxr81@case.edu]
- Lyra Bhatnagar - [GitHub: @username] - [lxb414@case.edu]
- Maulik Moghe - [GitHub: @username] - [Email]

---

## Project Overview

This project addresses the precision-utility tradeoff in machine unlearning for biosecurity applications. Building on the WMDP benchmark, we evaluate how domain-specific retain sets improve the selective removal of hazardous knowledge while preserving beneficial biological capabilities in large language models.

### Responsible AI Properties
- **Safety**: Removing hazardous biosecurity knowledge that could enable bioweapon development
- **Robustness**: Ensuring unlearned knowledge remains inaccessible even under adversarial attacks

### Research Question
**Can domain-specific retain sets improve unlearning precision, reducing hazardous biosecurity knowledge while maintaining general biology and defensive virology capabilities?**

---

## Project Scope (Revised per Professor Feedback)

### Focused Main Contribution
**Evaluating the impact of domain-specific retain sets on unlearning precision**, comparing:
1. Generic retain set (Wikitext - baseline)
2. Domain-general biology (introductory textbooks)
3. Domain-specific filtered biology (defensive virology content)

### Why This Scope?
- **Feasible for one semester**: Builds on established RMU method rather than new architecture
- **High impact**: Addresses critical precision problem identified in WMDP paper
- **Measurable outcomes**: Clear quantitative metrics for success
- **Robust experimentation**: 18 experimental configurations across 2 models

---

## Data Characteristics

### Forget Set (Hazardous Knowledge)
- **Source**: WMDP-Bio benchmark + PubMed corpus
- **Size**: 1,273 questions, ~500K tokens of text
- **Content**: Dual-use virology, bioweapons, reverse genetics, enhanced pathogens
- **Format**: Multiple-choice questions + unstructured text passages
- **Access**: Publicly available at https://wmdp.ai

### Retain Set (Beneficial Knowledge) - 3 Variants

**Variant 1: Generic Control**
- **Source**: Wikitext-103
- **Size**: 100M tokens
- **Purpose**: Baseline (as used in original WMDP)

**Variant 2: Domain-General Biology**
- **Sources**: 
  - OpenStax Biology (CC-BY, undergraduate level)
  - Khan Academy biology content
- **Size**: ~2M tokens
- **Content**: Cell biology, genetics, ecology (excluding virology)
- **Filtering**: Automated keyword removal + manual review

**Variant 3: Domain-Specific Filtered Biology**
- **Sources**:
  - Introductory virology textbook excerpts
  - PubMed abstracts (2020-2024)
- **Size**: ~1M tokens
- **Content**: Vaccine development, antivirals, diagnostics
- **Exclusions**: Gain-of-function, weaponization, enhanced virulence
- **Validation**: Manual review by biology domain experts

### Evaluation Datasets
- **WMDP-Bio**: 1,273 biosecurity questions (hazard measure)
- **MMLU Biology**: 144 questions (general knowledge retention)
- **MMLU Virology**: 166 questions (domain-specific retention)
- **Custom Boundary Set**: 100 questions (precision measure)

---

## ML Models

### Primary Models
1. **Zephyr-7B-Beta**
   - Size: 7 billion parameters
   - Type: Instruction-tuned (Mistral-7B base)
   - Baseline WMDP-Bio: 63.7%
   - Access: `HuggingFaceH4/zephyr-7b-beta`

2. **Yi-34B-Chat**
   - Size: 34 billion parameters
   - Type: Chat-optimized
   - Baseline WMDP-Bio: 75.3%
   - Access: `01-ai/Yi-34B-Chat`

### Model Selection Rationale
- **Scale diversity**: 7B and 34B parameters test generalization
- **Different training**: Instruction-tuned vs. chat-optimized
- **Established baselines**: Direct comparison with WMDP paper results
- **Computational feasibility**: Both models run on single A100 GPU

---

## Experimental Design

### Main Experiment: Retain Set Comparison

**Independent Variables**:
- Retain set type (3 variants)
- Model size (7B, 34B)
- Hyperparameters (layer, α, c)

**Dependent Variables**:
- WMDP-Bio accuracy (hazard reduction)
- MMLU Biology accuracy (general retention)
- MMLU Virology accuracy (domain retention)

**Controlled Variables**:
- Unlearning method (RMU)
- Forget set (WMDP-Bio corpus)
- Number of training steps (300)
- Batch size, learning rate

### Experimental Matrix
```
3 retain variants × 2 models × 3 hyperparameter configs = 18 runs
Each run repeated with 3 random seeds = 54 total evaluations
```

### Success Criteria
| Metric | Target | Baseline (RMU) | Improvement |
|--------|--------|----------------|-------------|
| WMDP-Bio | <30% | 31.2% | Maintain |
| MMLU Biology | >80% | 63.2% | +17% |
| MMLU Virology | >70% | 25.9% | +44% |

---

## Repository Structure

```
wmdp-unlearning-project/
├── README.md                    # This file
├── PROPOSAL.md                  # Full project proposal (ACM format)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
│
├── data/                        # Datasets
│   ├── wmdp_benchmark/          # WMDP questions
│   │   ├── wmdp_bio.json
│   │   ├── wmdp_cyber.json
│   │   └── wmdp_chem.json
│   ├── mmlu/                    # MMLU evaluation sets
│   └── retain_sets/             # Our curated retain sets
│       ├── generic/             # Wikitext
│       ├── domain_general/      # Biology textbooks
│       └── domain_specific/     # Filtered virology
│
├── src/                         # Source code
│   ├── data/
│   │   ├── load_wmdp.py        # WMDP data loader
│   │   ├── load_mmlu.py        # MMLU data loader
│   │   └── prepare_retain.py   # Retain set preparation
│   ├── unlearning/
│   │   └── rmu.py              # RMU implementation
│   ├── evaluation/
│   │   ├── benchmark.py        # Evaluation harness
│   │   ├── metrics.py          # Precision metrics
│   │   └── adversarial.py      # GCG attack testing
│   └── utils/
│       ├── model_utils.py      # Model loading helpers
│       └── visualization.py    # Result plotting
│
├── scripts/                     # Executable scripts
│   ├── download_data.sh        # Download all datasets
│   ├── download_models.sh      # Download HF models
│   ├── run_baseline.py         # Baseline evaluation
│   ├── run_unlearning.py       # Run RMU experiments
│   └── analyze_results.py      # Generate analysis
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_retain_set_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/                       # Unit tests
│   ├── test_data_loading.py
│   ├── test_unlearning.py
│   └── test_evaluation.py
│
├── results/                     # Experimental results
│   ├── baselines/              # Baseline model results
│   ├── experiments/            # Unlearning experiment results
│   └── figures/                # Generated plots
│
└── docs/                        # Documentation
    ├── installation.md          # Setup instructions
    ├── data_sources.md          # Dataset documentation
    ├── experiments.md           # Experimental protocol
    └── results_analysis.md      # Results interpretation
```

---

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/zazmir/WMDP-Benchmark
cd wmdp-unlearning-project

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data & Models

## Dataset Setup

This project builds on the official [WMDP Benchmark](https://huggingface.co/datasets/centerforaisafety/wmdp).

To download datasets:

```bash
pip install datasets
python - <<'PY'
from datasets import load_dataset

splits = ["bio", "chem", "cyber"]
for split in splits:
    ds = load_dataset("centerforaisafety/wmdp", split=split)
    ds.to_json(f"data/wmdp_benchmark/wmdp_{split}.json")
PY


### 3. Run Baseline Evaluation
```bash
# Evaluate Zephyr-7B on WMDP-Bio
python scripts/run_baseline.py --model zephyr-7b --dataset wmdp-bio

# Expected output: ~63.7% accuracy
```

### 4. Run Unlearning Experiment
```bash
# Run RMU with generic retain set
python scripts/run_unlearning.py \
  --model zephyr-7b \
  --retain-set generic \
  --output results/exp_001/
```

---

## Compute Requirements

### Minimum Setup
- **GPU**: NVIDIA GPU with 16GB VRAM (RTX 4090, A4000)
- **RAM**: 32GB
- **Storage**: 50GB
- **Time**: ~2 hours per experimental run

### Recommended Setup
- **GPU**: NVIDIA A100 (40GB) or A6000 (48GB)
- **RAM**: 64GB
- **Storage**: 100GB
- **Time**: ~1 hour per experimental run

### Optimization Strategies
- 8-bit quantization for memory-constrained environments
- Gradient checkpointing to reduce memory usage
- Mixed precision training (bfloat16)

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1-2 | Setup, baseline replication | Working codebase, baseline results |
| 3-4 | Retain set curation | 3 validated retain set variants |
| 5-7 | Core experiments | 18 experimental runs completed |
| 8-9 | Analysis & evaluation | Metrics, figures, statistical tests |
| 10 | Final report & demo | Report, presentation, code release |

---

## Evaluation Metrics

### Primary Metrics
1. **Hazard Reduction**: WMDP-Bio accuracy ↓
2. **General Knowledge**: MMLU Biology accuracy (maintain)
3. **Domain Precision**: MMLU Virology accuracy ↑

### Secondary Metrics
4. **Boundary Precision**: Custom boundary set accuracy
5. **Semantic Distance**: Embedding similarity analysis
6. **Adversarial Robustness**: GCG attack success rate

### Statistical Analysis
- Paired t-tests between retain set variants
- Effect size calculations (Cohen's d)
- Confidence intervals (95%) for all metrics
- Multiple comparison correction (Bonferroni)

---

## Key Results (To Be Updated)

*Results will be populated as experiments complete*

### Baseline Performance
| Model | WMDP-Bio | MMLU Bio | MMLU Virology |
|-------|----------|----------|---------------|
| Zephyr-7B | 63.7% | 68.1% | 52.4% |
| Yi-34B | 75.3% | 88.9% | 57.2% |

### After RMU (Original Paper)
| Model | WMDP-Bio | MMLU Bio | MMLU Virology |
|-------|----------|----------|---------------|
| Zephyr-7B | 31.2% | 63.2% | 25.9% |
| Yi-34B | 30.7% | 84.0% | 22.3% |

### Our Results (Domain-Specific Retain Set)
*To be completed*

---

## Citation

If you use this code or build upon our work:

```bibtex
@misc{wmdp2024unlearning,
  title={Improving Machine Unlearning Precision for Biosecurity Knowledge in LLMs},
  author={[Your Names]},
  year={2025},
  note={Course Project, Responsible AI Engineering}
}
```

Original WMDP Benchmark:
```bibtex
@article{li2024wmdp,
  title={The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning},
  author={Li, Nathaniel and Pan, Alexander and others},
  journal={arXiv preprint arXiv:2403.03218},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/zazmir/WMDP-Benchmark)

---

## Acknowledgments

- Professor Biswas and TA Panimalar for valuable feedback
- Original WMDP authors for open-sourcing benchmark
- Center for AI Safety for hosting WMDP resources
- HuggingFace for model infrastructure

---

*Last Updated: [Oct 13, 2025]*
