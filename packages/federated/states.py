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
from federated.config import CoordinatorConfig, ParticipantConfig
from numpy.typing import ArrayLike, NDArray

# Path to input and output directories in the FeatureCloud docker container
INPUT_DIR_PATH = Path("/mnt/input")
OUTPUT_DIR_PATH = Path("/mnt/output")


def init_config(
    config_path: Path, config_class: ParticipantConfig | CoordinatorConfig
) -> ParticipantConfig | CoordinatorConfig:
    with open(config_path, "r") as f:
        config_dict: Dict[str, Any] = yaml.safe_load(f)
    config = config_class(**config_dict)
    return config


@app_state("initial", role=Role.BOTH)
class InitialState(AppState):
    """Initial state for loading configurations and data."""

    def register(self):
        self.register_transition("validate_metadata", role=Role.BOTH)

    def run(self) -> str:
        """Initialize configurations and load data.

        Returns:
            str: Name of the next state
        """
        self.log("Initializing the application")
        self.log(f"Number of participants: {len(self.clients)}")
        self.log(f"List of Participant IDs: {self.clients}")
        self.log(f"Participant ID: {self.id}")
        self.log(f"Coordinator ID: {self.coordintor_id}")
        self.log(f"Participant is coordinator: {self.is_coordinator}")
        self.log(f"Input directory path: {INPUT_DIR_PATH}")
        self.log(f"Output directory path: {OUTPUT_DIR_PATH}")

        # Load the participant configuration
        participant_config_path = INPUT_DIR_PATH / "participant.yaml"
        self.log(f"Participant config path: {participant_config_path}")
        participant_config = init_config(participant_config_path, ParticipantConfig)
        self.store("participant_config", participant_config)
        self.log(f"Participant Config: {participant_config}")

        coordinator_config_path = INPUT_DIR_PATH / "coordinator.yaml"

        self.log(f"Coordinator config path: {coordinator_config_path}")
        coordinator_config = init_config(coordinator_config_path, CoordinatorConfig)
        self.store("coordinator_config", coordinator_config)
        self.log(f"Coordinator Config: {coordinator_config}")

        gene_expressions_path = (
            INPUT_DIR_PATH / participant_config.gene_expressions_path
        )

        transcription_factors_path = None
        if coordinator_config.transcription_factors_path is not None:
            transcription_factors_path = (
                INPUT_DIR_PATH / coordinator_config.transcription_factors_path
            )

        self.log(f"Gene expressions path: {gene_expressions_path}")
        self.log(f"Transcription factors path: {transcription_factors_path}")

        # Load and store the GRN dataset without reference network
        dataset = init_grn_dataset(
            gene_expressions_path=gene_expressions_path,
            transcription_factor_path=transcription_factors_path,
            reference_network_path=None,  # No reference network in production
        )
        self.store("dataset", dataset)
        return "validate_metadata"


@app_state("validate_metadata", role=Role.BOTH)
class ValidateMetadata(AppState):
    def register(self):
        self.register_transition("compute_local_importance_matrix", role=Role.BOTH)
        self.register_transition("terminal", role=Role.BOTH)

    def run(self) -> str:
        """Validate that all participants have the same genes and transcription factors.

        Args:
            dataset: The GRN dataset loaded by the participants

        Raises:
            ValueError: If validation fails
        """
        dataset: GRNDataset = self.load("dataset")
        # Extract gene names and transcription factor names for validation
        gene_names = set(dataset.gene_expressions.columns)
        num_samples = len(dataset.gene_expressions)

        # Send gene names and transcription factor names to coordinator for validation
        self.send_data_to_coordinator(
            (self.id, gene_names, num_samples),
            send_to_self=True,
            memo="send_metadata",
        )

        # If coordinator, validate that all participants have the same genes and transcription factors
        if self.is_coordinator:
            # Collect metadata from all participants
            metadata: list[tuple[int, set[str], set[str], int]] = self.gather_data(
                memo="send_metadata"
            )
            metadata: dict[int, tuple[set[str], set[str], int]] = {
                participant[0]: (
                    participant[1],
                    participant[2],
                )
                for participant in metadata
            }

            # The data from the coordinator serves as the reference the participant data is validated against
            reference_id = self.coordintor_id
            reference_genes = metadata[reference_id][1]

            for participant_id in metadata.keys():
                if participant_id == reference_id:
                    continue

                participant_genes = metadata[participant_id][1]

                if participant_genes != reference_genes:
                    missing_genes = reference_genes - participant_genes
                    extra_genes = participant_genes - reference_genes
                    error_msg = f"The participant's set of gene names {participant_id} does not equal the coordinator's set of gene names."
                    if missing_genes:
                        error_msg += f"Missing genes: {missing_genes}."
                    if extra_genes:
                        error_msg += f"Extra genes: {extra_genes}."

                    response: tuple[bool, int] = (
                        False,
                        error_msg,
                        -1,
                    )
                    self.broadcast_data(response, memo="validation_response")
                    self.log(error_msg, level=LogLevel.ERROR)
                    return "terminal"

            # Broadcast validation result to all clients
            total_num_samples: int = sum(
                [participant_metadata[1] for participant_metadata in metadata.values()]
            )
            self.store("total_num_samples", total_num_samples)
            response: tuple[bool, int] = (
                True,
                "",
                total_num_samples,
            )
            self.broadcast_data(response, memo="validation_response")
            return "compute_local_importance_matrix"
        else:
            # Participants wait for validation result from coordinator
            response: tuple[bool, int] = self.await_data(
                n=1, memo="validation_response"
            )
            response_code, error_msg, total_num_samples = response
            if response_code:
                self.log(
                    f"Data validation successful. Total number of samples across participants: {total_num_samples}"
                )
                self.store("total_num_samples", total_num_samples)
                return "compute_local_importance_matrix"
            else:
                self.log(f"Data validation failed: {error_msg}")
                return "terminal"


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
        coordinator_config: CoordinatorConfig = self.load("coordinator_config")

        # Calculate local importance scores
        regressor_config = (
            participant_config.regressor
            if participant_config.regressor is not None
            else coordinator_config.regressor
        )
        importance_matrix: ArrayLike = calculate_importances(dataset, regressor_config)
        importance_matrix: NDArray = np.asarray(importance_matrix)
        predicted_network = rank_genes_by_importance(
            dataset,
            importance_matrix,
        )
        self.log(f"Local predicted network preview:\n{predicted_network.head(2)}")

        # Save local predicted network
        output_network_path = OUTPUT_DIR_PATH / "local_predicted_network.csv"
        predicted_network.to_csv(output_network_path, index=False, sep="\t")
        self.log(f"Local predicted network saved to {output_network_path}")

        num_samples: int = len(dataset.gene_expressions)
        total_num_samples: int = self.load("total_num_samples")
        sample_size_weight = num_samples / total_num_samples
        weighted_importance_matrix = np.multiply(sample_size_weight, importance_matrix)
        # Send data to coordinator
        self.send_data_to_coordinator(
            weighted_importance_matrix,
            send_to_self=True,
            memo="importance_matrix",
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
        global_predicted_network.to_csv(output_network_path, index=False, sep="\t")

        self.log(f"Global predicted network saved to {output_network_path}")

        return "terminal"
