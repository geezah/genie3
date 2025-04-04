from pathlib import Path
from typing import Optional

import pandas as pd

from .dataset import GRNDataset
from .loading import (
    load_gene_expression_data,
    load_reference_network_data,
    load_transcription_factor_data,
)


def init_grn_dataset(
    gene_expressions_path: Path,
    transcription_factor_path: Optional[Path],
    reference_network_path: Optional[Path],
) -> GRNDataset:
    gene_expressions: pd.DataFrame = load_gene_expression_data(
        gene_expressions_path
    )
    transcription_factor_names = None
    if transcription_factor_path is not None:
        transcription_factor_names: pd.Series = load_transcription_factor_data(
            transcription_factor_path
        )

    reference_network = None
    if reference_network_path is not None:
        reference_network: pd.DataFrame = load_reference_network_data(
            reference_network_path
        )
    return GRNDataset(
        gene_expressions=gene_expressions,
        transcription_factor_names=transcription_factor_names,
        reference_network=reference_network,
    )


__all__ = [
    "GRNDataset",
    "init_grn_dataset",
    "load_gene_expression_data",
    "load_transcription_factor_data",
    "load_reference_network_data",
]
