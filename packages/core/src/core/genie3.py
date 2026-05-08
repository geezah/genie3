from typing import Tuple

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from .config import RegressorConfig
from .data import GRNDataset
from .regressor import RegressorRegistry


def run(dataset: GRNDataset, regressor_config: RegressorConfig) -> pl.LazyFrame:
    importance_scores = calculate_importances(dataset, regressor_config)
    return rank_genes_by_importance(dataset, importance_scores)


def preprocess_data(dataset: GRNDataset, regressor_config: RegressorConfig):
    gene_expressions_np = dataset.gene_expressions.collect().to_numpy()
    num_genes = gene_expressions_np.shape[1]
    num_tfs = len(dataset._transcription_factor_indices)
    importance_matrix = np.zeros((num_genes, num_tfs), dtype=np.float32)
    return RegressorRegistry.get(regressor_config.name).convert_inputs(
        gene_expressions_np,
        dataset._transcription_factor_indices,
        importance_matrix,
    )


def calculate_importances(
    dataset: GRNDataset,
    regressor_config: RegressorConfig,
) -> ArrayLike:
    gene_expressions, transcription_factor_indices, importance_matrix = preprocess_data(
        dataset, regressor_config
    )
    num_genes = gene_expressions.shape[1]
    regressor_cls = RegressorRegistry.get(regressor_config.name)
    for target_gene in tqdm(
        range(num_genes),
        total=num_genes,
        desc="Computing importances",
        unit="gene",
    ):
        regressor = regressor_cls(regressor_config.init_params)
        X, y, input_genes = partition_data(
            gene_expressions, transcription_factor_indices, target_gene
        )
        regressor.fit(X, y, regressor_config.fit_params)
        importance_matrix[target_gene, input_genes] = regressor.feature_importances_
    return importance_matrix


def rank_genes_by_importance(
    dataset: GRNDataset,
    importance_matrix: ArrayLike,
) -> pl.LazyFrame:
    """
    Ranks genes by their importance scores and returns a LazyFrame with gene names.

    Args:
        dataset (GRNDataset): The GRN dataset containing gene and TF metadata.
        importance_matrix (ArrayLike): Matrix of shape (num_genes, num_tfs) containing
            importance scores.

    Returns:
        pl.LazyFrame: LazyFrame with columns (transcription_factor, target_gene, importance),
            sorted by importance in descending order.
    """
    gene_names: list[str] = dataset._gene_names
    tf_indices: ArrayLike = np.asarray(dataset._transcription_factor_indices)
    num_genes, num_tfs = importance_matrix.shape

    # Build index arrays for all (target_gene, tf) pairs via broadcasting
    target_indices = np.repeat(
        np.arange(num_genes), num_tfs
    )  # shape: (num_genes * num_tfs,)
    tf_col_indices = np.tile(
        np.arange(num_tfs), num_genes
    )  # shape: (num_genes * num_tfs,)
    tf_gene_indices = tf_indices[tf_col_indices]  # map TF column → gene index

    return pl.LazyFrame(
        {
            "transcription_factor": [gene_names[i] for i in tf_gene_indices],
            "target_gene": [gene_names[i] for i in target_indices],
            "importance": importance_matrix.ravel().astype(np.float32),
        }
    ).sort("importance", descending=True)


def partition_data(
    gene_expressions: ArrayLike,
    transcription_factor_indices: ArrayLike,
    target_gene: int,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    mask = transcription_factor_indices != target_gene
    input_genes = transcription_factor_indices[mask]
    X = gene_expressions[:, input_genes]
    y = gene_expressions[:, target_gene]
    return X, y, input_genes
