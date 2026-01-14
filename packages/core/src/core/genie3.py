from typing import List, Tuple

import cupy as cp
import numpy as np
from cupy.typing import ArrayLike
from tqdm.auto import tqdm

from .config import RegressorConfig
from .data import GRNDataset

from .regressor import RegressorRegistry, CUDA_AVAILABLE


if CUDA_AVAILABLE:
    try:
        from cudf.pandas import install
        install()
        import cupy as cp
        xp = cp
    except ImportError:
        pass
    xp = np
else:
    xp = np
import pandas as pd  # noqa : F401


def run(dataset: GRNDataset, regressor_config: RegressorConfig) -> pd.DataFrame:
    importance_scores = calculate_importances(
        dataset,
        regressor_config,
    )
    predicted_network = rank_genes_by_importance(
        dataset,
        importance_scores,
    )
    return predicted_network


def calculate_importances(
    dataset: GRNDataset,
    regressor_config: RegressorConfig,
) -> ArrayLike:
    gene_expressions: ArrayLike = xp.asarray(dataset.gene_expressions.values)

    transcription_factor_indices: List = dataset._transcription_factor_indices
    transcription_factor_indices: ArrayLike = xp.asarray(
        transcription_factor_indices, dtype=xp.uint32
    )

    # Get the number of genes and transcription factors
    num_genes = gene_expressions.shape[1]
    num_transcription_factors = len(transcription_factor_indices)

    # Initialize the importance matrix
    importance_matrix = xp.zeros(
        (num_genes, num_transcription_factors), dtype=xp.float32
    )

    # Standardize data
    gene_expressions = (gene_expressions - gene_expressions.mean(axis=0)) / (
        gene_expressions.std() + 1e-8
    )

    gene_expressions, transcription_factor_indices, importance_matrix = (
        RegressorRegistry.get(regressor_config.name).convert_inputs(
            gene_expressions, transcription_factor_indices, importance_matrix
        )
    )

    progress_bar = tqdm(
        range(num_genes),
        total=num_genes,
        desc="Computing importances",
        unit="gene",
    )
    for target_gene in progress_bar:
        regressor = RegressorRegistry.get(regressor_config.name)(
            regressor_config.init_params
        )
        X, y, input_genes = partition_data(
            gene_expressions,
            transcription_factor_indices,
            target_gene,
        )
        regressor.fit(X, y, regressor_config.fit_params)
        importance_matrix[target_gene, input_genes] = regressor.feature_importances_

    importance_matrix = xp.asarray(importance_matrix)
    return importance_matrix


def rank_genes_by_importance(
    dataset: GRNDataset,
    importance_matrix: ArrayLike,
) -> pd.DataFrame:
    """
    Ranks genes by their importance scores and returns a DataFrame with gene names.

    Args:
        importance_matrix: Matrix of importance scores
        transcription_factor_indices: List of TF indices
        gene_names: List of gene names

    Returns:
        pandas DataFrame with columns (transcription_factor, target_gene, importance)
        where transcription_factor and target_gene are gene names
    """
    rows = []
    transcription_factor_indices: List = dataset._transcription_factor_indices
    gene_names: List = dataset._gene_names

    num_genes, num_transcription_factors = importance_matrix.shape

    for i in range(num_genes):
        for j in range(num_transcription_factors):
            tf_idx = transcription_factor_indices[j]
            target_idx = i
            importance = importance_matrix[i, j]

            # Use gene names instead of indices
            tf_name = gene_names[tf_idx]
            target_name = gene_names[target_idx]

            rows.append(
                {
                    "transcription_factor": tf_name,
                    "target_gene": target_name,
                    "importance": float(importance),
                }
            )
    # Convert to DataFrame and sort by importance
    predicted_network = pd.DataFrame(rows)
    predicted_network = predicted_network.sort_values(by="importance", ascending=False)
    return predicted_network


def partition_data(
    gene_expressions: ArrayLike,
    transcription_factor_indices: ArrayLike,
    target_gene: int,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    # Create a mask for transcription factors that are not the target gene
    mask = transcription_factor_indices != target_gene

    # Apply the mask to get input genes
    input_genes = transcription_factor_indices[mask]

    # Extract input features and target variable
    X = gene_expressions[:, input_genes]
    y = gene_expressions[:, target_gene]

    return X, y, input_genes
