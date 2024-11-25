from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


def _map_gene_ids_to_names_for_expression_data(
    gene_expressions: pd.DataFrame,
    gene_ids: Dict[str, str],
) -> pd.DataFrame:
    gene_expressions.columns = gene_expressions.columns.map(gene_ids)
    return gene_expressions


def _map_gene_ids_to_names_for_network_data(
    network: pd.DataFrame, gene_ids: Dict[str, str]
) -> pd.DataFrame:
    network["transcription_factor"] = network["transcription_factor"].map(
        gene_ids
    )
    network["target_gene"] = network["target_gene"].map(gene_ids)
    return network


def _map_gene_ids_to_names_for_transcription_factors(
    transcription_factors: pd.Series, gene_ids: Dict[str, str]
) -> pd.Series:
    transcription_factors = transcription_factors.map(gene_ids)
    return transcription_factors


def init_dream_five_dataset(root: Path, network_id: Literal[1, 3, 4]) -> None:
    network_id_to_directory_name = {
        1: Path("Network 1 - in silico"),
        # 2: Path("Network 2 - S. aureus"), # Not used for evaluation
        3: Path("Network 3 - E. coli"),
        4: Path("Network 4 - S. cerevisiae"),
    }
    network_dir = network_id_to_directory_name[network_id]

    root: Path = root
    training_data_dir = root / "training data"
    reference_network_dir = root / "test data"

    gene_expression_path = (
        training_data_dir
        / network_dir
        / f"net{network_id}_expression_data.tsv"
    )
    id_to_name_path = (
        training_data_dir / network_dir / f"net{network_id}_gene_ids.tsv"
    )
    transcription_factors_path = (
        training_data_dir
        / network_dir
        / f"net{network_id}_transcription_factors.tsv"
    )
    network_data_path = (
        reference_network_dir
        / f"DREAM5_NetworkInference_GoldStandard_{str(network_dir).replace(f'Network {network_id}', f'Network{network_id}')}.tsv"
    )

    gene_expression_data = pd.read_csv(
        gene_expression_path,
        sep="\t",
        dtype=np.float32,
    )
    id_to_name_mapping = pd.read_csv(
        id_to_name_path,
        sep="\t",
        dtype=str,
    )
    id_to_name_mapping = dict(id_to_name_mapping.values)
    transcription_factor_data = pd.read_csv(
        transcription_factors_path,
        sep="\t",
        header=None,
        dtype=str,
    )

    reference_network_data = pd.read_csv(
        network_data_path,
        sep="\t",
        header=None,
        dtype={
            0: str,
            1: str,
            2: float,
        },
    )

    reference_network_data.columns = [
        "transcription_factor",
        "target_gene",
        "label",
    ]
    transcription_factor_data = transcription_factor_data[0]
    transcription_factor_data.name = "transcription_factor"

    gene_expression_data = _map_gene_ids_to_names_for_expression_data(
        gene_expression_data, id_to_name_mapping
    )
    reference_network_data = _map_gene_ids_to_names_for_network_data(
        reference_network_data, id_to_name_mapping
    )
    transcription_factor_data = (
        _map_gene_ids_to_names_for_transcription_factors(
            transcription_factor_data, id_to_name_mapping
        )
    )

    return (
        gene_expression_data,
        reference_network_data,
        transcription_factor_data,
    )


def preprocess_dream_five(
    raw_data_root: Path = Path("local_data/raw/syn2787209/Gene Network Inference"),
    processed_data_root: Path = Path("local_data/processed/dream_five"),
):
    print(
        f"Processing DREAM5 data from {raw_data_root} to {processed_data_root}"
    )
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    progress_bar = tqdm(
        net_id_to_net_name.items(),
        desc="Processing DREAM5 data",
        unit="network",
    )
    for net_id, net_name in progress_bar:
        # Create directory for network
        net_dir = processed_data_root / f"net{net_id}_{net_name}"
        net_dir.mkdir(exist_ok=True, parents=True)
        # Load data
        (
            gene_expression_data,
            reference_network_data,
            transcription_factor_data,
        ) = init_dream_five_dataset(raw_data_root, net_id)
        # Save data
        gene_expression_data.to_csv(
            net_dir / "gene_expression_data.tsv",
            index=False,
            sep="\t",
        )
        reference_network_data.to_csv(
            net_dir / "reference_network_data.tsv",
            index=False,
            sep="\t",
        )
        transcription_factor_data.to_csv(
            net_dir / "transcription_factors.tsv",
            index=False,
            sep="\t",
        )