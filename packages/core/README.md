# GENIE3: Gene Regulatory Network Inference with Ensemble of Trees

GENIE3 (GEne Network Inference with Ensemble of trees) is an algorithmic framework for inferring gene regulatory networks from gene expression data with tree-based ensemble methods. This implementation provides a flexible, efficient, and user-friendly way to reconstruct gene regulatory networks with optional GPU acceleration.

## Overview

GENIE3 treats the problem of network inference as a feature selection problem: for each gene in the network, it tries to predict its expression level based on the expression levels of all other genes, particularly focusing on transcription factors. The importance of each input gene in the prediction is then used as an estimate of the strength of the regulatory link.

## Features

- **GPU Acceleration**: Optional CUDA support for faster computation with cuDF, cuML, and CuPy (CUDA 12.x or 13.x)

- **Multiple Regressor Support**: Includes implementations for various tree-based ensemble methods:
  - Random Forest (CPU & GPU)
  - Extra Trees (CPU only)

- **Comprehensive Evaluation**: Built-in metrics for evaluating inferred networks:
  - AUROC (Area Under the Receiver Operating Characteristic curve)
  - AUPRC (Area Under the Precision-Recall Curve)
  - Visualization tools for ROC and PR curves

- **Rigorous Data Handling**: Rigorous data validation via Pydantic

- **Experiment Tracking**: Tools for logging experiments and results

## Installation

This package is part of the GENIE3 monorepo. See the [main repository README](../../README.md) for installation instructions.

## Basic Usage

The following example shows how to use the package to infer a gene regulatory network from gene expression data and transcription factors.

```python
import pandas as pd
from genie3.data import GRNDataset
from genie3.config import RegressorConfig
from genie3.genie3 import run
from genie3.eval import prepare_evaluation, run_evaluation, Results

# Load your data
gene_expressions = pd.DataFrame(...)  # Your gene expression data
tf_names = pd.Series(...)  # Your transcription factor names

# Create a dataset
dataset = GRNDataset(
    gene_expressions=gene_expressions,
    transcription_factor_names=tf_names
)

# Configure the regressor
config = RegressorConfig(
    name="ExtraTreesRegressor",
    init_params={"n_estimators": 100, "random_state": 42}
)

# Run GENIE3
predicted_network = run(dataset, config)

# The result is a DataFrame with columns: transcription_factor, target_gene, importance
print(predicted_network.head())

# If you have a reference network, you can evaluate the performance of the inferred network
y_preds, y_true = prepare_evaluation(
        predicted_network, grn_dataset.reference_network
    )
results : Results = run_evaluation(y_preds, y_true)
```
You can also run GENIE3 using the [provided CLI](../../scripts/local_app.py)
This CLI can be used for inference only as well as for evaluation against a reference network.
From the repository's root directory, executing the following command runs GENIE3 with the configuration specified in [`randomforest.yaml`](../../configs/randomforest.yaml):

```bash
uv run python scripts/local_app.py configs/randomforest.yaml
```

## Configuration Format

The configuration is expected to adhere to the following structure:

```yaml
data:
  gene_expressions_path: "data/gene_expression_data.tsv" # Required
  transcription_factors_path: "data/transcription_factors.tsv" # If not provided, will use all genes as transcription factors
  reference_network_path: "data/reference_network_data.tsv" # Can be empty if you want to run inference only

regressor:
  name: "ExtraTreesRegressor" # Supported regressors: ['ExtraTreesRegressor', RandomForestRegressor', 'CuRandomForestRegressor']
  init_params:
    n_estimators: 100
    random_state: 42
    max_features: 0.1
  fit_params:
```

## Data Format

The expected format of the data files. Note that the header rows are expected to be present in the respective files.

### Gene Expression Data

A tab-separated file with genes as columns, samples as rows, and gene expression values as entries:

```tsv
Gene1   Gene2   Gene3   ... # Header row ()
0.5     1.2     0.8     ...
0.7     0.9     1.1     ...
...
```

### Transcription Factor List

A tab-separated file with one column containing transcription factor names. The transcription factors are expected to be present in the columns of the gene expression data.

```tsv
transcription_factor # Header row
Gene1
Gene2
...
```

### Reference Network (for evaluation)

A tab-separated file with columns for transcription factors, target genes, and binary labels, indicating the presence of an edge between the transcription factor and the target gene:

```tsv
transcription_factor  target_gene  label # Header row
Gene1                 Gene2        1
Gene1                 Gene3        0
...
```

## Advanced Usage

### Implementing Custom Regressors

You can extend GENIE3 with custom regressors by implementing the `RegressorProtocol` interface:

```python
from genie3.regressor import RegressorFactory, RegressorProtocol

class MyCustomRegressor(RegressorProtocol):
    def __init__(self, **init_params):
        # Initialize your regressor
        pass

    def fit(self, X, y, **fit_params):
        # Fit your regressor
        # Set feature_importances_ property
        self.feature_importances_ = ...

# Register your regressor
RegressorFactory.register("MyCustomRegressor", MyCustomRegressor)
```

### Using a Custom Regressor

Once you've implemented and registered your custom regressor, you can use it like any other regressor:

```python
from genie3.config import RegressorConfig
from genie3.genie3 import run

# Load your data
gene_expressions = pd.DataFrame(...)  # Your gene expression data
tf_names = pd.Series(...)  # Your transcription factor names

# Create a dataset
dataset = GRNDataset(
    gene_expressions=gene_expressions,
    transcription_factor_names=tf_names
)
# Configure with your custom regressor
config = RegressorConfig(
    name="MyCustomRegressor",
    init_params={"my_param": 42},
    fit_params={"another_param": True}
)

# Run GENIE3 with your custom regressor
predicted_network = run(dataset, config)
```

## Package Structure

```
packages/core/
├── src/core/
│   ├── __init__.py
│   ├── config.py        # Configuration models
│   ├── data.py          # Dataset handling and validation
│   ├── eval.py          # Evaluation metrics and visualization
│   ├── genie3.py        # Main GENIE3 algorithm
│   ├── plot.py          # Plotting utilities
│   ├── regressor.py     # Regressor registry and protocols
│   └── utils.py         # Utility functions
├── tests/               # Unit tests
├── pyproject.toml       # Package configuration
└── README.md           # This file
```

## Related Resources

- [Main Repository README](../../README.md)
- [Federated GENIE3 Package](../federated/README.md) - For privacy-preserving federated learning

## Citation

If you use this implementation in your research, please cite the original GENIE3 paper:

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
  langid = {english},
  keywords = {Gene Regulatory Networks}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
