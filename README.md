# GENIE3: Gene Regulatory Network Inference with Ensemble of Trees

GENIE3 (GEne Network Inference with Ensemble of trees) is an algorithmic framework for inferring gene regulatory networks from gene expression data with tree-based ensemble methods. This implementation provides a flexible, efficient, and user-friendly way to reconstruct gene regulatory networks.

## Overview

GENIE3 treats the problem of network inference as a feature selection problem: for each gene in the network, it tries to predict its expression level based on the expression levels of all other genes, particularly focusing on transcription factors. The importance of each input gene in the prediction is then used as an estimate of the strength of the regulatory link.

## Features

- **Multiple Regressor Support**: Includes implementations for various tree-based ensemble methods:
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - LightGBM

- **Comprehensive Evaluation**: Built-in metrics for evaluating inferred networks:
  - AUROC (Area Under the Receiver Operating Characteristic curve)
  - AUPRC (Area Under the Precision-Recall Curve)
  - Visualization tools for ROC and PR curves

- **Rigorous Data Handling**: Rigorous data validation via pydantic

- **Experiment Tracking**: Tools for logging experiments and results 

## Installation

### Standard Installation

This package requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed. When this is done, the package and the environment can be set up with the following commands:

```bash
# Clone the repository
git clone https://github.com/geezah/genie3.git
cd genie3

# Install the package
uv sync
```

### Basic Usage

The following example shows how to use the package to infer a gene regulatory network from gene expression data and transcription factors.

```python
import pandas as pd
from genie3.data.dataset import GRNDataset
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

### Using the CLI

You can also run GENIE3 using the provided CLI. This CLI can be used for inference only as well as for evaluation against a reference network. The following command will run GENIE3 with the configuration file `configs/extratrees.yaml`:

```bash
python scripts/local_app.py configs/extratrees.yaml
```

Example configuration file (`configs/extratrees.yaml`):

```yaml
data:
  gene_expressions_path: "data/gene_expression_data.tsv"
  transcription_factors_path: "data/transcription_factors.tsv" # If not provided, will use all genes as transcription factors
  reference_network_path: "data/reference_network_data.tsv" # Can be omitted for inference

regressor:
  name: "ExtraTreesRegressor" # Can be any of the supported regressors
  init_params: 
    n_estimators: 100
    random_state: 42
    max_features: 0.1
    n_jobs: 8
  fit_params:
```

## Data Format

### Gene Expression Data

A tab-separated file with genes as columns, samples as rows, and gene expression values as entries:

```csv
        Gene1   Gene2   Gene3   ...
Sample1 0.5     1.2     0.8     ...
Sample2 0.7     0.9     1.1     ...
...
```

### Transcription Factor List

A tab-separated file with one column containing transcription factor names. The transcription factors are expected to be present in the columns of the gene expression data.

```csv
TF
Gene1
Gene2
...
```

### Reference Network (for evaluation)

A tab-separated file with columns for transcription factors, target genes, and binary labels, indicating the presence of an edge between the transcription factor and the target gene:

```csv
transcription_factor  target_gene  label
Gene1                 Gene2        1
Gene1                 Gene3        0
...
```

## Advanced Usage

### Custom Regressors

You can extend GENIE3 with custom regressors by implementing the `RegressorProtocol` interface:

```python
from genie3.regressor import RegressorFactory, RegressorProtocol

class MyCustomRegressor(RegressorProtocol):
    def __init__(self, **init_params):
        # Initialize your regressor
        pass
        
    def fit(self, X, y, **fit_params):
        # Fit your regressor
        # Set feature_importances property
        self.feature_importances = ...

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.