from pathlib import Path
from typing import List, Optional

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class GRNDataset(BaseModel):
    """
    A class representing a Gene Regulatory Network (GRN) dataset.

    Attributes:
    ----------
    gene_expressions : pl.LazyFrame
        A LazyFrame where rows represent samples and columns represent genes.
        Entries are the gene expression values.

    transcription_factor_names : Optional[pl.Series]
        An optional Series where each entry represents the name of a transcription factor (TF).
        If provided, it will be checked against the gene_expressions schema.

    reference_network : Optional[pl.LazyFrame]
        An optional LazyFrame with columns:
        - `transcription_factor` (str): Name of the transcription factor.
        - `target_gene` (str): Name of the target gene.
        - `label` ({0, 1}): Indicates whether there is a regulatory interaction (1) or not (0).
        If provided, it will be checked to ensure the `transcription_factor` and `target_gene`
        columns are present in the gene_expressions schema.

    _gene_names : List[str]
        A dynamically created list of gene names derived from the gene_expressions schema,
        with sorted TF columns first followed by sorted non-TF columns.

    _transcription_factor_indices : List[int]
        A dynamically created list of transcription factor column indices.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gene_expressions: pl.LazyFrame = Field(
        ...,
        description="A LazyFrame of gene expression values with samples as rows and genes as columns.",
    )
    transcription_factor_names: Optional[pl.Series] = Field(
        None, description="A Series containing transcription factor names."
    )
    reference_network: Optional[pl.LazyFrame] = Field(
        None,
        description="A LazyFrame representing the reference network with columns: "
        "transcription_factor, target_gene, and label.",
    )

    _gene_names: List[str] = PrivateAttr()
    _transcription_factor_indices: List[int] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        tf_columns = sorted(self.transcription_factor_names.to_list())
        non_tf_columns = sorted(
            col
            for col in self.gene_expressions.collect_schema().names()
            if col not in tf_columns
        )
        ordered_columns = tf_columns + non_tf_columns
        self.gene_expressions = self.gene_expressions.select(ordered_columns)
        self._gene_names = ordered_columns
        self._transcription_factor_indices = list(range(len(tf_columns)))

    @field_validator("reference_network", mode="after")
    @classmethod
    def check_label_values(
        cls, value: Optional[pl.LazyFrame]
    ) -> Optional[pl.LazyFrame]:
        """Verify that the label column contains only 0s and 1s."""
        if value is not None:
            invalid_labels = value.filter(
                pl.col("label").lt(0), pl.col("label").gt(1)
            ).collect()
            if invalid_labels.height > 0:
                raise ValueError(
                    "The label column in the reference_network must contain only 0s and 1s."
                )
        return value

    @model_validator(mode="after")
    def tfs_subset_gene_expression_columns(self) -> Self:
        """
        If transcription_factor_names is provided, verify every TF name appears
        in the gene_expressions schema.  Otherwise, default to all column names.
        """
        if self.transcription_factor_names is not None:
            gene_columns = self.gene_expressions.collect_schema().names()
            invalid_tfs = self.transcription_factor_names.filter(
                self.transcription_factor_names.is_in(gene_columns).not_()
            )
            if invalid_tfs.len() > 0:
                raise ValueError(
                    "The following transcription factors are not present in the "
                    f"gene_expressions columns: {set(invalid_tfs.to_list())}"
                )
        else:
            self.transcription_factor_names = pl.Series(
                "transcription_factor_names",
                self.gene_expressions.collect_schema().names(),
            )
        return self

    @model_validator(mode="after")
    def validate_unique_gene_names(self) -> Self:
        """Validate that gene names in gene_expressions are unique."""
        columns = self.gene_expressions.collect_schema().names()
        if len(columns) != len(set(columns)):
            duplicates = {col for col in columns if columns.count(col) > 1}
            raise ValueError(
                f"Gene names must be unique. Found duplicate gene names: {duplicates}"
            )
        return self

    @model_validator(mode="after")
    def validate_reference_network(self) -> Self:
        if self.reference_network is not None:
            # --- Schema-level: required columns present ------------------
            required_columns = {"transcription_factor", "target_gene", "label"}
            missing_columns = required_columns - set(
                self.reference_network.collect_schema().names()
            )
            if missing_columns:
                raise ValueError(
                    "The reference_network LazyFrame is missing the following required "
                    f"columns: {missing_columns}"
                )

            # --- Data-level: no duplicate (tf, target_gene) pairs -------
            duplicates = (
                self.reference_network.group_by(["transcription_factor", "target_gene"])
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
                .collect()
            )
            if duplicates.height > 0:
                raise ValueError(
                    f"Found duplicate entries in the reference network:\n{duplicates}"
                )

            # --- Data-level: TFs and targets must exist in gene_expressions
            gene_columns = self.gene_expressions.collect_schema().names()
            tf_names = self.transcription_factor_names.to_list()

            tfs_not_in_columns = (
                self.reference_network.filter(
                    pl.col("transcription_factor").is_in(gene_columns).not_()
                )
                .select(pl.col("transcription_factor").unique())
                .collect()
            )
            targets_not_in_columns = (
                self.reference_network.filter(
                    pl.col("target_gene").is_in(gene_columns).not_()
                )
                .select(pl.col("target_gene").unique())
                .collect()
            )
            non_tfs_in_network = (
                self.reference_network.filter(
                    pl.col("transcription_factor").is_in(tf_names).not_()
                )
                .select(pl.col("transcription_factor").unique())
                .collect()
            )

            errors = []
            if tfs_not_in_columns.height > 0:
                errors.append(
                    "Transcription factors not found in gene expressions columns: "
                    f"{set(tfs_not_in_columns['transcription_factor'].to_list())}"
                )
            if targets_not_in_columns.height > 0:
                errors.append(
                    "Target genes not found in gene expressions columns: "
                    f"{set(targets_not_in_columns['target_gene'].to_list())}"
                )
            if non_tfs_in_network.height > 0:
                errors.append(
                    "Transcription factors in reference_network but not in "
                    f"transcription_factor_names: "
                    f"{set(non_tfs_in_network['transcription_factor'].to_list())}"
                )
            if errors:
                raise ValueError(
                    "\n".join(
                        [
                            "The reference_network LazyFrame is invalid due to the following errors:",
                            *errors,
                        ]
                    )
                )
        return self


def load_gene_expression_data(gene_expression_path: Path) -> pl.LazyFrame:
    """
    Lazily scan gene expression data from a tab-separated file.

    Args:
        gene_expression_path (Path): Path to the gene expression TSV file.

    Returns:
        pl.LazyFrame: Lazily scanned gene expression data.
    """
    return pl.scan_csv(gene_expression_path, separator="\t", has_header=True)


def load_transcription_factor_data(
    transcription_factor_path: Path,
) -> pl.Series:
    """
    Load transcription factor names from a tab-separated file.

    Collected eagerly since TF names must be in memory for schema-level
    validation and column reordering during GRNDataset initialisation.

    Args:
        transcription_factor_path (Path): Path to the transcription factor TSV file.

    Returns:
        pl.Series: Transcription factor names.
    """
    return (
        pl.scan_csv(transcription_factor_path, separator="\t", has_header=True)
        .collect()
        .to_series(0)
    )


def load_reference_network_data(reference_network_path: Path) -> pl.LazyFrame:
    """
    Lazily scan reference network data from a tab-separated file.

    The file is expected to contain the following columns:
    ``transcription_factor``, ``target_gene``, and ``label``.

    Args:
        reference_network_path (Path): Path to the reference network TSV file.

    Returns:
        pl.LazyFrame: Lazily scanned reference network data.
    """
    return pl.scan_csv(reference_network_path, separator="\t", has_header=True)


def init_grn_dataset(
    gene_expressions_path: Path,
    transcription_factor_path: Optional[Path] = None,
    reference_network_path: Optional[Path] = None,
) -> GRNDataset:
    return GRNDataset(
        gene_expressions=load_gene_expression_data(gene_expressions_path),
        transcription_factor_names=(
            load_transcription_factor_data(transcription_factor_path)
            if transcription_factor_path is not None
            else None
        ),
        reference_network=(
            load_reference_network_data(reference_network_path)
            if reference_network_path is not None
            else None
        ),
    )


__all__ = [
    "GRNDataset",
    "init_grn_dataset",
    "load_gene_expression_data",
    "load_transcription_factor_data",
    "load_reference_network_data",
]
