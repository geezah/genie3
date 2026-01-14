from pathlib import Path
from typing import Any, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .regressor import CUDA_AVAILABLE

if CUDA_AVAILABLE:
    try: 
        from cudf.pandas import install
        install()
    except ImportError:
        pass
import pandas as pd  # noqa : F401


class GRNDataset(BaseModel):
    """
    A class representing a Gene Regulatory Network (GRN) dataset.

    Attributes:
    ----------
    gene_expressions : pd.DataFrame
        A DataFrame where rows represent samples and columns represent genes.
        Entries in the DataFrame are the gene expression values.

    transcription_factor_names : Optional[pd.Series]
        An optional Series where each entry represents the name of a transcription factor (TF).
        If provided, it will be checked against the gene_expressions columns.

    reference_network : Optional[pd.DataFrame]
        An optional DataFrame with columns:
        - `transcription_factor` (str): Name of the transcription factor.
        - `target_gene` (str): Name of the target gene.
        - `label` ({0, 1}): Indicates whether there is a regulatory interaction (1) or not (0).
        If provided, it will be checked to ensure the `transcription_factor` and `target_gene`
        columns are present in the gene_expressions DataFrame.

    _gene_names : List[str]
        A dynamically created list of gene names derived from the columns of the gene_expressions DataFrame.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gene_expressions: pd.DataFrame = Field(
        ...,
        description="A DataFrame of gene expression values with samples as rows and genes as columns.",
    )
    transcription_factor_names: Optional[pd.Series] = Field(
        None, description="A Series containing transcription factor names."
    )
    reference_network: Optional[pd.DataFrame] = Field(
        None,
        description="A DataFrame representing the reference network with columns: transcription_factor, target_gene, and label.",
    )
    _gene_names: List[str] = PrivateAttr()  # Set dynamically
    _transcription_factor_indices: List[int] = PrivateAttr()  # Set dynamically

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._gene_names = list(self.gene_expressions.columns)
        names_to_indices = {name: index for index, name in enumerate(self._gene_names)}
        self._transcription_factor_indices = self.transcription_factor_names.map(
            names_to_indices
        )

    @field_validator("reference_network", mode="after")
    @classmethod
    def check_label_values(cls, value: pd.DataFrame) -> pd.DataFrame:
        # Verify that the label column contains only 0s and 1s
        if value is not None:
            if not set(value["label"].unique()) == {0, 1}:
                raise ValueError(
                    "The label column in the reference_network DataFrame must contain only 0s and 1s."
                )
        return value

    @model_validator(mode="after")
    def tfs_subset_gene_expression_columns(self) -> Self:
        # If transcription_factor_names is provided, check if it is a subset of the gene_expressions columns

        if self.transcription_factor_names is not None:
            invalid_tfs = set(self.transcription_factor_names) - set(
                self.gene_expressions.columns
            )
            if invalid_tfs:
                raise ValueError(
                    f"The following transcription factors are not present in the gene_expressions columns: {invalid_tfs}"
                )
        else:
            # If transcription_factor_names is not provided, create a Series from the gene_expressions columns
            self.transcription_factor_names = pd.Series(self.gene_expressions.columns)
        return self

    @model_validator(mode="after")
    def validate_unique_gene_names(self) -> Self:
        """Validate that gene names in gene_expressions are unique."""
        if len(self.gene_expressions.columns) != len(
            set(self.gene_expressions.columns)
        ):
            duplicates = [
                col
                for col in self.gene_expressions.columns
                if list(self.gene_expressions.columns).count(col) > 1
            ]
            raise ValueError(
                f"Gene names must be unique. Found duplicate gene names: {set(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def validate_reference_network(self) -> Self:
        if self.reference_network is not None:
            required_columns = {
                "transcription_factor",
                "target_gene",
                "label",
            }
            missing_columns = required_columns - set(self.reference_network.columns)
            if missing_columns:
                raise ValueError(
                    f"The reference_network DataFrame is missing the following required columns: {missing_columns}"
                )

            # Check for duplicate entries in the reference network
            is_duplicate = self.reference_network.duplicated(
                subset=["transcription_factor", "target_gene"]
            )
            if is_duplicate.any():
                duplicates = self.reference_network[is_duplicate]
                raise ValueError(
                    f"Found duplicate entries in the reference network: \n{duplicates}"
                )

            tfs_not_in_columns = set(
                self.reference_network["transcription_factor"]
            ) - set(self.gene_expressions.columns)
            targets_not_in_columns = set(self.reference_network["target_gene"]) - set(
                self.gene_expressions.columns
            )
            non_tfs_in_tfs = set(self.reference_network["transcription_factor"]) - set(
                self.transcription_factor_names
            )
            if tfs_not_in_columns or targets_not_in_columns or non_tfs_in_tfs:
                errors = []
                if tfs_not_in_columns:
                    errors.append(
                        f"Transcription factors not found in gene expressions columns: {tfs_not_in_columns}"
                    )
                if targets_not_in_columns:
                    errors.append(
                        f"Target genes not found in gene expressions columns: {targets_not_in_columns}"
                    )
                if non_tfs_in_tfs:
                    errors.append(
                        f"Transcription factors in reference_network but not in transcription_factor_names: {non_tfs_in_tfs}"
                    )

                raise ValueError(
                    "\n".join(
                        [
                            "The reference_network DataFrame is invalid due to the following errors:",
                            *errors,
                        ]
                    )
                )

        return self


def load_gene_expression_data(
    gene_expression_path: Path,
) -> pd.DataFrame:
    """
    Load gene expression data from a file.

    Args:
        gene_expression_path (Path): Path to the gene expression data file.

    Returns:
        pd.DataFrame: Gene expression data.
    """
    df = pd.read_csv(gene_expression_path, sep="\t", header=0)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pd.Series:
    """
    Load transcription factor data from a file.

    Args:
        transcription_factor_path (Path): Path to the transcription factor data file.

    Returns:
        pd.Series: Transcription factor data.
    """
    return pd.read_csv(
        transcription_factor_path, sep="\t", header=0
    ).squeeze()


def load_reference_network_data(reference_network_path: Path) -> pd.DataFrame:
    """
    Load reference network data from a given file path.
    This function reads a tab-separated values (TSV) file containing reference network data
    and returns it as a pandas DataFrame. The file is expected to have the following columns:
    "transcription_factor", "target_gene", and "label".
    Args:
        reference_network_path (Path): The file path to the reference network data.
    Returns:
        pd.DataFrame: A DataFrame containing the reference network data.
    """
    df: pd.DataFrame = pd.read_csv(reference_network_path, sep="\t", header=0)
    return df


def init_grn_dataset(
    gene_expressions_path: Path,
    transcription_factor_path: Optional[Path],
    reference_network_path: Optional[Path],
) -> GRNDataset:
    gene_expressions: pd.DataFrame = load_gene_expression_data(gene_expressions_path)
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
