"""FeatureCloud app states for federated GRN inference using GENIE3.

This module implements the FeatureCloud app states for federated Gene Regulatory Network (GRN)
inference using GENIE3. It includes states for initialization, local computation, and aggregation
of results across multiple sites.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from core.data import GRNDataset, init_grn_dataset
from core.genie3 import calculate_importances, rank_genes_by_importance
from FeatureCloud.app.engine.app import AppState, LogLevel, Role, app_state
from federated.config import ParticipantConfig
from numpy.typing import ArrayLike, NDArray

# Path to input and output directories in the FeatureCloud docker container
INPUT_DIR_PATH = Path("/mnt/input")
OUTPUT_DIR_PATH = Path("/mnt/output")


@app_state("initial", role=Role.BOTH)
class InitialState(AppState):
    """Initial state for loading configurations and data."""

    def register(self):
        self.register_transition("compute_local_importance_matrix", role=Role.BOTH)

    def run(self) -> str:
        """Initialize configurations and load data.

        Returns:
            str: Name of the next state
        """
        self.log("Initializing the application")

        self.log(f"Input directory path: {INPUT_DIR_PATH}")
        self.log(f"Output directory path: {OUTPUT_DIR_PATH}")

        # Get the config path
        client_config_path = INPUT_DIR_PATH / "client.yaml"

        self.log(f"Client Config path: {client_config_path}")

        with open(client_config_path, "r") as f:
            participant_config_dict: Dict[str, Any] = yaml.safe_load(f)

        participant_config: ParticipantConfig = ParticipantConfig(
            **participant_config_dict
        )
        self.store("participant_config", participant_config)
        self.log(f"Participant Config: {participant_config}")

        gene_expressions_path = (
            INPUT_DIR_PATH / participant_config.data.gene_expressions_path
        )
        if participant_config.data.transcription_factors_path:
            transcription_factors_path = (
                INPUT_DIR_PATH / participant_config.data.transcription_factors_path
            )
        else:
            transcription_factors_path = None

        self.log(f"Gene expressions path: {gene_expressions_path}")
        self.log(f"Transcription factors path: {transcription_factors_path}")

        # Load and store the GRN dataset without reference network
        dataset = init_grn_dataset(
            gene_expressions_path=gene_expressions_path,
            transcription_factor_path=transcription_factors_path,
            reference_network_path=None,  # No reference network in production
        )
        self.store("dataset", dataset)

        # Validate data consistency across clients
        self.validate_data_consistency(dataset)

        return "compute_local_importance_matrix"

    def validate_data_consistency(self, dataset: GRNDataset) -> None:
        """Validate that all clients have the same genes and transcription factors.

        Args:
            dataset: The GRN dataset loaded by the client

        Raises:
            ValueError: If validation fails
        """
        # Extract gene names and transcription factor names for validation
        gene_names = list(dataset.gene_expressions.columns)
        tf_names = list(dataset.transcription_factor_names)
        num_samples = len(dataset.gene_expressions)

        # Send gene names and transcription factor names to coordinator for validation
        self.send_data_to_coordinator(
            (gene_names, tf_names, num_samples),
            send_to_self=True,
            memo="data_validation",
        )

        # If coordinator, validate that all clients have the same genes and transcription factors
        if self.is_coordinator:
            # Wait for data from all clients
            validation_data = self.gather_data(memo="data_validation")

            # Extract gene names and transcription factor names from all clients
            all_gene_names = [data[0] for data in validation_data]
            all_tf_names = [data[1] for data in validation_data]
            all_num_samples = [data[2] for data in validation_data]

            # Check if all clients have the same gene names
            reference_genes = set(all_gene_names[0])
            for i, client_genes in enumerate(all_gene_names[1:], 1):
                client_genes = set(client_genes)
                if reference_genes != client_genes:
                    missing_genes = reference_genes - client_genes
                    extra_genes = client_genes - reference_genes
                    error_msg = (
                        f"Client {i} has different genes than the reference client."
                    )
                    if missing_genes:
                        error_msg += f" Missing genes: {missing_genes}."
                    if extra_genes:
                        error_msg += f" Extra genes: {extra_genes}."
                    self.log(error_msg, level=LogLevel.ERROR)
                    raise ValueError(error_msg)

            # Check if all clients have the same transcription factor names
            reference_tfs = set(all_tf_names[0])
            for i, client_tfs in enumerate(all_tf_names[1:], 1):
                client_tfs = set(client_tfs)
                if reference_tfs != client_tfs:
                    missing_tfs = reference_tfs - client_tfs
                    extra_tfs = client_tfs - reference_tfs
                    error_msg = f"Client {i} has different transcription factors than the reference client."
                    if missing_tfs:
                        error_msg += f" Missing TFs: {missing_tfs}."
                    if extra_tfs:
                        error_msg += f" Extra TFs: {extra_tfs}."
                    self.log(error_msg, level=LogLevel.ERROR)
                    raise ValueError(error_msg)

            self.log(
                "Data validation successful: All clients have the same genes and transcription factors."
            )

            # Broadcast validation result to all clients
            total_num_samples = sum(all_num_samples)
            response: tuple[bool, int] = (True, total_num_samples)
            self.broadcast_data(response, memo="validation_result")
        else:
            # Participants wait for validation result from coordinator
            response_code, total_num_samples = self.await_data(
                n=1, memo="validation_result"
            )
            if not response_code:
                # TODO: Improve error messaging from coordinator
                self.log("Data validation failed. Exiting.", level=LogLevel.ERROR)
                raise ValueError(
                    "Data validation failed. Check coordinator logs for details."
                )
            self.log("Data validation successful.")
        self.store("total_num_samples", total_num_samples)


@app_state("compute_local_importance_matrix", role=Role.BOTH)
class ComputeLocalImportanceMatrix(AppState):
    """State for computing local importance matrices at each site."""

    def register(self):
        self.register_transition("aggregate", role=Role.COORDINATOR)
        self.register_transition("terminal", role=Role.PARTICIPANT)

    def run(self) -> str:
        """Compute local importance matrix and predicted network.

        Returns:
            str: Name of the next state
        """
        dataset: GRNDataset = self.load("dataset")
        participant_config: ParticipantConfig = self.load("participant_config")

        # Calculate local importance scores
        importance_matrix: ArrayLike = calculate_importances(
            dataset, participant_config.regressor
        )
        importance_matrix: NDArray = np.asarray(importance_matrix)
        predicted_network = rank_genes_by_importance(
            dataset,
            importance_matrix,
        )
        self.log(f"Local predicted network preview:\n{predicted_network.head(2)}")

        # Save local predicted network
        output_network_path = OUTPUT_DIR_PATH / "local_predicted_network.csv"
        predicted_network.to_csv(output_network_path, index=False)
        self.log(f"Local predicted network saved to {output_network_path}")

        num_samples: int = len(dataset.gene_expressions)
        total_num_samples: int = self.load("total_num_samples")
        sample_size_weight = num_samples / total_num_samples
        weighted_importance_matrix = np.multiply(sample_size_weight, importance_matrix)
        # Send data to coordinator
        self.send_data_to_coordinator(
            weighted_importance_matrix, send_to_self=True, memo="importance_matrix"
        )
        return "aggregate" if self.is_coordinator else "terminal"


@app_state("aggregate", role=Role.COORDINATOR)
class AggregationState(AppState):
    """State for aggregating results from all sites."""

    def register(self):
        self.register_transition("terminal", role=Role.COORDINATOR)

    def run(self) -> str:
        """Aggregate local results and save global predictions.

        Returns:
            str: Name of the next state
        """
        dataset: GRNDataset = self.load("dataset")

        # Gather data from all sites
        global_importance_matrix: NDArray = self.aggregate_data(
            memo="importance_matrix"
        )
        # Generate global predictions
        global_predicted_network = rank_genes_by_importance(
            dataset,
            global_importance_matrix,
        )
        # Log results
        self.log(
            f"Global predicted network preview:\n{global_predicted_network.head(2)}"
        )
        # Save outputs
        output_network_path = OUTPUT_DIR_PATH / "global_predicted_network.csv"
        global_predicted_network.to_csv(output_network_path, index=False)

        self.log(f"Global predicted network saved to {output_network_path}")

        return "terminal"
