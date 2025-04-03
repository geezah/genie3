from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from genie3.config import RegressorConfig
from genie3.data import GRNDataset

from .regressor import (
    RegressorFactory,
)


def run(
    dataset: GRNDataset, regressor_config: RegressorConfig
) -> pd.DataFrame:
    importance_scores = calculate_importances(
        dataset.gene_expressions.values,
        dataset._transcription_factor_indices,
        regressor_config
    )
    predicted_network = rank_genes_by_importance(
        importance_scores,
        dataset._transcription_factor_indices,
        dataset._gene_names,
    )
    return predicted_network


def partition_data(
    gene_expressions: NDArray,
    transcription_factor_indices: List[int],
    target_gene: int,
) -> Tuple[NDArray, NDArray, List[int]]:
    # Remove target gene from regulator list and gene expression matrix
    input_genes = [i for i in transcription_factor_indices if i != target_gene]
    X = gene_expressions[:, input_genes]
    y = gene_expressions[:, target_gene]
    return X, y, input_genes


def calculate_importances(
    gene_expressions: NDArray,
    transcription_factor_indices: List[int],
    regressor_config : RegressorConfig
) -> NDArray:
    # Get the number of genes and transcription factors
    num_genes = gene_expressions.shape[1]
    num_transcription_factors = len(transcription_factor_indices)

    # Standardize data
    gene_expressions = (
        gene_expressions - gene_expressions.mean(axis=0)
    ) / gene_expressions.std()

    # Initialize importance matrix
    importance_matrix = np.zeros(
        (num_genes, num_transcription_factors), dtype=np.float32
    )

    progress_bar = tqdm(
        range(num_genes),
        total=num_genes,
        desc="Computing importances",
        unit="gene",
        miniters=num_genes // 100,
    )
    for target_gene in progress_bar:
        regressor = RegressorFactory.get(regressor_config.name)(
            regressor_config.init_params
        )
        X, y, input_genes = partition_data(
            gene_expressions,
            transcription_factor_indices,
            target_gene,
        )
        regressor.fit(X, y, regressor_config.fit_params)
        importance_matrix[target_gene, input_genes] = (
            regressor.feature_importances
        )
    return importance_matrix


def rank_genes_by_importance(
    importance_matrix: NDArray,
    transcription_factor_indices: List[int],
    gene_names: List[str],
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
    num_genes, num_transcription_factors = importance_matrix.shape

    # Create list of regulator-target pairs with importance scores
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
    predicted_network = predicted_network.sort_values(
        by="importance", ascending=False
    )

    return predicted_network
