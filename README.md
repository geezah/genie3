# GENIE3: Gene Regulatory Network Inference with Ensemble of Trees

![uv-badge](https://img.shields.io/badge/managed%20by-uv-purple)

An implementation of GENIE3 (GEne Network Inference with Ensemble of trees) with optional GPU acceleration and federated learning capabilities.

## Overview

GENIE3 is an algorithmic framework for inferring gene regulatory networks (GRNs) from gene expression data using tree-based ensemble methods. This repository provides a modern, efficient implementation with support for:

- **GPU Acceleration**: Optional CUDA support for faster computation using cuDF, cuML, and CuPy
- **Federated Learning**: Privacy-preserving GRN inference across distributed data sites using FeatureCloud
- **Multiple Regressors**: Support for various tree-based models (Random Forest, Extra Trees, Gradient Boosting, LightGBM)
- **Comprehensive Evaluation**: Built-in metrics and visualization tools for network quality assessment

## Project Structure

This project is organized as a workspace managed with [uv](https://docs.astral.sh/uv/) containing two core packages:

```
genie3/
├── packages/
│   ├── core/          # Core GENIE3 implementation with GPU support
│   └── federated/     # Federated GENIE3 for FeatureCloud platform
├── configs/           # Example configuration files
├── scripts/           # Utility scripts
└── pyproject.toml     # Workspace configuration
```

### Packages

#### 1. **Core Package** (`packages/core/`)

📖 [Core Package Documentation](packages/core/README.md)

#### 2. **Federated Package** (`packages/federated/`)

📖 [Federated Package Documentation](packages/federated/README.md)

## Installation

This project requires Python 3.13 and the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.

### CPU-only Installation (default)
```bash
# Clone the repository
git clone https://github.com/geezah/genie3.git
cd genie

# Install with uv
uv sync --all-packages --extra cpu
```

### Optional GPU installation

For CUDA 13.x:
```bash
uv sync --all-packages --extra cu13
```

For CUDA 12.x:
```bash
uv sync --all-packages --extra cu12
```

**Note**: GPU support requires the NVIDIA CUDA Toolkit and compatible GPU drivers to be installed on your system.

**Note**: Currently, GPU acceleration is only available for local computations done with the `core` package. The `federated` application does not support GPU acceleration as of now.

### Running Tests

Currently, tests are implemented for the core package only.

```bash
# Run all tests
uv run pytest
```

## Citation

If you use this implementation in your research, please cite the original GENIE3 publication:

```bibtex
@article{huynh-thuInferringRegulatoryNetworks2010,
  title = {Inferring {{Regulatory Networks}} from {{Expression Data Using Tree-Based Methods}}},
  author = {{Huynh-Thu}, V{\^a}n Anh and Irrthum, Alexandre and Wehenkel, Louis and Geurts, Pierre},
  editor = {Isalan, Mark},
  year = {2010},
  month = sep,
  journal = {PLoS ONE},
  volume = {5},
  number = {9},
  pages = {e12776},
  issn = {1932-6203},
  doi = {10.1371/journal.pone.0012776},
  urldate = {2024-09-23},
  langid = {english}
}
```

## License

This project is licensed under the [Unlicense](LICENSE).

## Acknowledgments

This implementation is based on the original GENIE3 algorithm by Huynh-Thu et al. (2010). The federated learning extension enables privacy-preserving collaborative research across institutions.
