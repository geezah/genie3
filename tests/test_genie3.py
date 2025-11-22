import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from genie3.genie3 import (
    run,
    partition_data,
    calculate_importances,
    rank_genes_by_importance,
)
from genie3.data import GRNDataset
from genie3.config import RegressorConfig


@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    gene_expressions = pd.DataFrame(
        {
            "gene1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "gene2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "gene3": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    tf_names = pd.Series(["gene1", "gene2"])

    return GRNDataset(
        gene_expressions=gene_expressions,
        transcription_factor_names=tf_names,
    )


@pytest.fixture
def regressor_config():
    """Create a sample regressor configuration."""
    return RegressorConfig(
        name="RandomForestRegressor",
        init_params={"n_estimators": 10, "random_state": 42},
        fit_params={},
    )


class TestGenie3:
    def test_partition_data(self):
        """Test the partition_data function."""
        gene_expressions = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        tf_indices = np.array([0, 1, 2])
        target_gene = 1

        X, y, input_genes = partition_data(
            gene_expressions, tf_indices, target_gene
        )

        # Check that target gene is removed from input genes
        assert np.array_equal(input_genes, np.array([0, 2]))
        # Check that X contains the correct columns
        assert X.shape == (3, 2)
        assert np.array_equal(X[:, 0], gene_expressions[:, 0])
        assert np.array_equal(X[:, 1], gene_expressions[:, 2])
        # Check that y contains the target gene expression
        assert np.array_equal(y, gene_expressions[:, target_gene])

    def test_rank_genes_by_importance(self, sample_dataset):
        """Test the rank_genes_by_importance function."""
        importance_matrix = np.array(
            [
                [0.0, 0.7],  # gene1's importance for each TF
                [0.5, 0.0],  # gene2's importance for each TF
                [0.3, 0.2],  # gene3's importance for each TF
            ],
            dtype=np.float32,
        )

        predicted_network = rank_genes_by_importance(
            sample_dataset, importance_matrix
        )

        # Check that the result is a DataFrame with the expected columns
        assert isinstance(predicted_network, pd.DataFrame)
        assert set(predicted_network.columns) == {
            "transcription_factor",
            "target_gene",
            "importance",
        }

        # Check that the network is sorted by importance (descending)
        assert (
            predicted_network["importance"].iloc[0]
            >= predicted_network["importance"].iloc[1]
        )
        assert (
            predicted_network["importance"].iloc[1]
            >= predicted_network["importance"].iloc[2]
        )

        # Check that the highest importance is gene1 regulated by gene2
        assert predicted_network["transcription_factor"].iloc[0] == "gene2"
        assert predicted_network["target_gene"].iloc[0] == "gene1"
        assert np.isclose(predicted_network["importance"].iloc[0], 0.7)

        # Check that the second highest importance is gene2 regulated by gene1
        assert predicted_network["transcription_factor"].iloc[1] == "gene1"
        assert predicted_network["target_gene"].iloc[1] == "gene2"
        assert np.isclose(predicted_network["importance"].iloc[1], 0.5)

    @patch("genie3.genie3.calculate_importances")
    @patch("genie3.genie3.rank_genes_by_importance")
    def test_run(
        self, mock_rank, mock_calculate, sample_dataset, regressor_config
    ):
        """Test the run function with mocked dependencies."""
        # Use the fixtures properly instead of calling them directly

        # Mock the importance calculation
        mock_importance_matrix = np.array(
            [
                [0.0, 0.7],
                [0.5, 0.0],
                [0.3, 0.2],
            ],
            dtype=np.float32,
        )
        mock_calculate.return_value = mock_importance_matrix

        # Mock the ranking function
        mock_predicted_network = pd.DataFrame(
            {
                "transcription_factor": ["gene2", "gene1", "gene1"],
                "target_gene": ["gene1", "gene2", "gene3"],
                "importance": [0.7, 0.5, 0.3],
            }
        )
        mock_rank.return_value = mock_predicted_network

        # Run the function
        result = run(sample_dataset, regressor_config)

        # Check that the functions were called with correct arguments
        mock_calculate.assert_called_once()
        mock_rank.assert_called_once_with(
            sample_dataset,
            mock_importance_matrix,
        )

        # Check that the result is the predicted network
        assert result is mock_predicted_network

    def test_calculate_importances_integration(
        self, sample_dataset, regressor_config
    ):
        """Integration test for calculate_importances with a real regressor."""
        # Use a small number of estimators for faster testing
        regressor_config.init_params["n_estimators"] = 5

        importance_matrix = calculate_importances(
            sample_dataset,
            regressor_config,
        )

        # Check the shape of the importance matrix
        assert importance_matrix.shape == (3, 2)  # 3 genes, 2 TFs

        # Check that the diagonal elements are zero (a gene can't regulate itself)
        assert importance_matrix[0, 0] == 0.0  # gene1 can't regulate itself
        assert importance_matrix[1, 1] == 0.0  # gene2 can't regulate itself

        # Check that all values are between 0 and 1
        assert np.all(importance_matrix >= 0.0)
        assert np.all(importance_matrix <= 1.0)

    def test_run_integration(self, sample_dataset, regressor_config):
        """Integration test for the full GENIE3 pipeline."""
        # Use a small number of estimators for faster testing
        regressor_config.init_params["n_estimators"] = 5

        result = run(sample_dataset, regressor_config)

        # Check that the result has the expected structure
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "transcription_factor",
            "target_gene",
            "importance",
        }

        # Check that we have the expected number of rows
        assert len(result) == 6  # 2 TFs x 3 genes = 6 potential regulations

        # Check that the network is sorted by importance (descending)
        assert result["importance"].iloc[0] >= result["importance"].iloc[1]
