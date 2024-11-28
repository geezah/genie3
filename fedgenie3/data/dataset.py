from pathlib import Path
from typing import Optional

import pandas as pd


class GRNDataset:
    def __init__(
        self,
        gene_expression_path: Path,
        transcription_factor_path: Optional[Path] = None,
        reference_network_path: Optional[Path] = None,
    ):
        self.gene_expression_path = gene_expression_path
        self.transcription_factor_path = transcription_factor_path
        self.reference_network_path = reference_network_path

        self.gene_expression_data = self._load_gene_expression_data()
        self.transcription_factor_data = (
            self._load_transcription_factor_data()
            if transcription_factor_path
            else None
        )
        self.reference_network_data = (
            self._load_reference_network_data()
            if reference_network_path
            else None
        )

    def _load_gene_expression_data(self) -> pd.Series:
        df = pd.read_csv(self.gene_expression_path, sep="\t")
        return df

    def _load_transcription_factor_data(self) -> pd.DataFrame:
        series = pd.read_csv(
            self.transcription_factor_path, sep="\t"
        ).squeeze()
        assert isinstance(series, pd.Series)
        series.name = "transcription_factor"
        return series

    def _load_reference_network_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.reference_network_path, sep="\t")
        assert list(df.columns) == [
            "transcription_factor",
            "target_gene",
            "label",
        ]
        return df


if __name__ == "__main__":
    net_id_to_net_name = {
        1: "in-silico",
        3: "e-coli",
        4: "s-cerevisiae",
    }
    NETWORK_ID = 4

    net_id = NETWORK_ID
    net_name = net_id_to_net_name[NETWORK_ID]
    PROCESSED_DATA_ROOT = Path("local_data/processed/dream_five")
    GENE_EXPRESSION_PATH = (
        PROCESSED_DATA_ROOT
        / f"net{net_id}_{net_name}"
        / f"net{net_id}_{net_name}_expression_data.tsv"
    )
    REFERENCE_NETWORK_PATH = (
        PROCESSED_DATA_ROOT
        / f"net{net_id}_{net_name}"
        / f"net{net_id}_{net_name}_reference_network_data.tsv"
    )
    TRANSCRIPTION_FACTOR_PATH = (
        PROCESSED_DATA_ROOT
        / f"net{net_id}_{net_name}"
        / f"net{net_id}_{net_name}_transcription_factors.tsv"
    )

    grn_dataset = GRNDataset(
        gene_expression_path=GENE_EXPRESSION_PATH,
        reference_network_path=REFERENCE_NETWORK_PATH,
        transcription_factor_path=TRANSCRIPTION_FACTOR_PATH,
    )
