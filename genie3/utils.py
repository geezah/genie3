from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from numpy.typing import NDArray

from genie3.config import GENIE3Config


def write_config(config: GENIE3Config, output_dir: Path) -> None:
    """
    Write the configuration to a YAML file.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        output_dir (Path): The directory where the configuration will be saved.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        # Convert the GENIE3Config object to a dictionary with paths casted to strings and dump it as JSON
        config.data.gene_expressions_path = str(
            config.data.gene_expressions_path.resolve()
        )
        if config.data.transcription_factors_path is not None:
            config.data.transcription_factors_path = str(
                config.data.transcription_factors_path.resolve()
            )
        if config.data.reference_network_path is not None:
            config.data.reference_network_path = str(
                config.data.reference_network_path.resolve()
            )

        config = config.model_dump()
        yaml.safe_dump(config, f)


def write_ndarray(array: NDArray, output_path: Path) -> None:
    """
    Write a numpy array to a file.

    Args:
        array (NDArray): The array to save.
        output_path (Path): The path to save the array to.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def write_network(network: pd.DataFrame, output_path: Path) -> None:
    """
    Write a network to a file.

    Args:
        network (pd.DataFrame): The network to save.
        output_path (Path): The path to save the network to.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.to_csv(output_path, index=False)


def write_metrics(
    auroc: float, auprc: float, pos_frac: float, output_dir: Path
) -> None:
    """
    Write the AUROC and AUPRC metrics to a CSV file.

    Args:
        auroc (float): The Area Under the Receiver Operating Characteristic curve score.
        auprc (float): The Area Under the Precision-Recall curve score.
        pos_frac (float): The fraction of positive examples in the dataset.
        output_dir (Path): The directory where the metrics will be saved.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": ["auroc", "auprc", "pos_frac"],
            "score": [auroc, auprc, pos_frac],
        },
    ).to_csv(output_dir / "metrics.csv", index=False)


def write_plot(plot: Figure, output_path: Path) -> None:
    """
    Write a plot to a file.

    Args:
        plot (Figure): The plot to save.
        output_path (Path): The path to save the plot to.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(output_path)


def write_results_inference_only(
    config: GENIE3Config,
    predicted_network: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> None:
    """
    Save the results of the inference phase only.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        predicted_network (pd.DataFrame): The predicted network as a pandas DataFrame.
        output_dir (Path, optional): The directory where the results will be saved. Defaults to "results".

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dump the model configuration
    write_config(config, output_dir)
    # Dump the predicted network
    write_network(predicted_network, output_dir / "predicted_network.csv")


def write_results_full_pipeline(
    config: GENIE3Config,
    auroc: float,
    auprc: float,
    pos_frac: float,
    fpr: NDArray,
    tpr: NDArray,
    recall: NDArray,
    precision: NDArray,
    predicted_network: pd.DataFrame,
    reference_network: pd.DataFrame,
    roc_curve_plot: Figure,
    precision_recall_curve_plot: Figure,
    output_dir: Path = Path("results"),
) -> None:
    """
    Save all results including metrics, predicted and reference networks, and plots.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        auroc (float): The Area Under the Receiver Operating Characteristic curve score.
        auprc (float): The Area Under the Precision-Recall curve score.
        pos_frac (float): The fraction of positive examples in the dataset.
        fpr (NDArray): The false positive rates.
        tpr (NDArray): The true positive rates.
        recall (NDArray): The recall scores.
        precision (NDArray): The precision scores.
        predicted_network (pd.DataFrame): The predicted network as a pandas DataFrame.
        reference_network (pd.DataFrame): The reference network as a pandas DataFrame.
        roc_curve_plot (Figure): The ROC curve plot as a matplotlib Figure.
        precision_recall_curve_plot (Figure): The precision-recall curve plot as a matplotlib Figure.
        output_dir (Path): The directory where the results will be saved.

    Returns:
        None
    """
    write_config(config, output_dir)
    # Dump the metrics
    write_metrics(auroc, auprc, pos_frac, output_dir)
    # Dump the arrays
    write_ndarray(fpr, output_dir / "fpr.npy")
    write_ndarray(tpr, output_dir / "tpr.npy")
    write_ndarray(recall, output_dir / "recall.npy")
    write_ndarray(precision, output_dir / "precision.npy")
    # Dump the predicted and reference networks
    write_network(predicted_network, output_dir / "predicted_network.csv")
    write_network(reference_network, output_dir / "reference_network.csv")
    # Dump the plots
    write_plot(roc_curve_plot, output_dir / "roc_curve.png")
    write_plot(precision_recall_curve_plot, output_dir / "precision_recall_curve.png")
