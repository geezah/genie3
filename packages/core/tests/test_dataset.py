import pytest
import pandas as pd
from core.data import GRNDataset


@pytest.fixture
def valid_gene_expressions():
    return pd.DataFrame(
        {
            "gene1": [0.1, 0.2, 0.3],
            "gene2": [0.4, 0.5, 0.6],
            "gene3": [0.7, 0.8, 0.9],
        }
    )


@pytest.fixture
def valid_tf_names():
    return pd.Series(["gene1", "gene2"])


@pytest.fixture
def valid_reference_network():
    return pd.DataFrame(
        {
            "transcription_factor": ["gene1", "gene2"],
            "target_gene": ["gene2", "gene3"],
            "label": [1, 0],
        }
    )


class TestGRNDataset:
    def test_valid_dataset_creation(
        self, valid_gene_expressions, valid_tf_names, valid_reference_network
    ):
        # Test that a valid dataset can be created without errors
        dataset = GRNDataset(
            gene_expressions=valid_gene_expressions,
            transcription_factor_names=valid_tf_names,
            reference_network=valid_reference_network,
        )

        assert dataset._gene_names == ["gene1", "gene2", "gene3"]
        assert list(dataset._transcription_factor_indices) == [0, 1]

    def test_dataset_without_tf_names(
        self, valid_gene_expressions, valid_reference_network
    ):
        # Test that dataset can be created without providing TF names
        dataset = GRNDataset(
            gene_expressions=valid_gene_expressions,
            reference_network=valid_reference_network,
        )

        # Should use all genes as TFs
        assert list(dataset.transcription_factor_names) == [
            "gene1",
            "gene2",
            "gene3",
        ]
        assert list(dataset._transcription_factor_indices) == [0, 1, 2]

    def test_dataset_without_reference_network(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that dataset can be created without providing a reference network
        dataset = GRNDataset(
            gene_expressions=valid_gene_expressions,
            transcription_factor_names=valid_tf_names,
        )

        assert dataset.reference_network is None
        assert dataset._gene_names == ["gene1", "gene2", "gene3"]

    def test_invalid_tf_names(self, valid_gene_expressions):
        # Test that an error is raised when TF names are not in gene expressions
        invalid_tf_names = pd.Series(["gene1", "gene4"])

        with pytest.raises(
            ValueError, match="not present in the gene_expressions columns"
        ):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=invalid_tf_names,
            )

    def test_duplicate_gene_names(self):
        # Test that an error is raised when gene names are duplicated
        # Create DataFrame with duplicate columns using a different approach
        gene_expressions = pd.DataFrame(
            [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
            columns=["gene1", "gene2", "gene1"],  # Duplicate column name
        )

        with pytest.raises(ValueError, match="Gene names must be unique"):
            GRNDataset(gene_expressions=gene_expressions)

    def test_invalid_reference_network_missing_columns(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that an error is raised when reference network is missing required columns
        invalid_reference_network = pd.DataFrame(
            {
                "transcription_factor": ["gene1", "gene2"],
                "target": [
                    "gene2",
                    "gene3",
                ],  # Here, "target" should be named "target_gene"
                "label": [1, 0],
            }
        )

        with pytest.raises(ValueError, match="missing the following required columns"):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=valid_tf_names,
                reference_network=invalid_reference_network,
            )

    def test_invalid_reference_network_labels(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that an error is raised when reference network has invalid labels
        invalid_reference_network = pd.DataFrame(
            {
                "transcription_factor": ["gene1", "gene2"],
                "target_gene": ["gene2", "gene3"],
                "label": [
                    1,
                    2,
                ],  # "2" is an invalid label as we only accept labels in {0,1}
            }
        )

        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=valid_tf_names,
                reference_network=invalid_reference_network,
            )

    def test_invalid_reference_network_genes(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that an error is raised when reference network contains genes not in expressions
        invalid_reference_network = pd.DataFrame(
            {
                "transcription_factor": [
                    "gene1",
                    "gene4",
                ],  # gene4 not in expressions
                "target_gene": ["gene2", "gene3"],
                "label": [1, 0],
            }
        )

        with pytest.raises(ValueError, match="not found in gene expressions columns"):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=valid_tf_names,
                reference_network=invalid_reference_network,
            )

    def test_invalid_reference_network_non_tf(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that an error is raised when reference network contains TFs not in TF names
        invalid_reference_network = pd.DataFrame(
            {
                "transcription_factor": [
                    "gene1",
                    "gene3",
                ],  # gene3 not in TF names
                "target_gene": ["gene2", "gene1"],
                "label": [1, 0],
            }
        )

        with pytest.raises(ValueError, match="not in transcription_factor_names"):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=valid_tf_names,
                reference_network=invalid_reference_network,
            )

    def test_duplicate_reference_network_entries(
        self, valid_gene_expressions, valid_tf_names
    ):
        # Test that an error is raised when reference network contains duplicate entries
        duplicate_reference_network = pd.DataFrame(
            {
                "transcription_factor": ["gene1", "gene1"],  # Duplicate entry
                "target_gene": ["gene2", "gene2"],  # Duplicate entry
                "label": [1, 0],
            }
        )

        with pytest.raises(
            ValueError,
            match="Found duplicate entries in the reference network",
        ):
            GRNDataset(
                gene_expressions=valid_gene_expressions,
                transcription_factor_names=valid_tf_names,
                reference_network=duplicate_reference_network,
            )
