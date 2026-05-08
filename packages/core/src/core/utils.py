from pathlib import Path

import numpy as np
import polars as pl
import yaml
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .config import GENIE3Config


def write_config(config: GENIE3Config, output_dir: Path) -> None:
    """
    Write the configuration to a YAML file.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        output_dir (Path): The directory where the configuration will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        config_dict = config.model_dump()
        config_dict["data"]["gene_expressions_path"] = str(
            config.data.gene_expressions_path.resolve()
        )
        if config.data.transcription_factors_path is not None:
            config_dict["data"]["transcription_factors_path"] = str(
                config.data.transcription_factors_path.resolve()
            )
        if config.data.reference_network_path is not None:
            config_dict["data"]["reference_network_path"] = str(
                config.data.reference_network_path.resolve()
            )
        yaml.safe_dump(config_dict, f)


def write_ndarray(array: NDArray, output_path: Path) -> None:
    """
    Write a numpy array to a file.

    Args:
        array (NDArray): The array to save.
        output_path (Path): The path to save the array to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)


def write_network(network: pl.LazyFrame, output_path: Path) -> None:
    """
    Write a network to a tab-separated file.

    Args:
        network (pl.LazyFrame): The network to save.
        output_path (Path): The path to save the network to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.collect().write_csv(output_path, separator="\t")


def write_metrics(
    auroc: float, auprc: float, pos_frac: float, output_dir: Path
) -> None:
    """
    Write the AUROC, AUPRC, and pos_frac metrics to a tab-separated CSV file.

    Args:
        auroc (float): The Area Under the Receiver Operating Characteristic curve score.
        auprc (float): The Area Under the Precision-Recall curve score.
        pos_frac (float): The fraction of positive examples in the dataset.
        output_dir (Path): The directory where the metrics will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "metric": ["auroc", "auprc", "pos_frac"],
            "score": [auroc, auprc, pos_frac],
        }
    ).write_csv(output_dir / "metrics.csv", separator="\t")


def write_plot(plot: Figure, output_path: Path) -> None:
    """
    Write a plot to a file.

    Args:
        plot (Figure): The plot to save.
        output_path (Path): The path to save the plot to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(output_path)


def write_results_inference_only(
    config: GENIE3Config,
    predicted_network: pl.LazyFrame,
    output_dir: Path = Path("results"),
) -> None:
    """
    Save the results of the inference phase only.

    Args:
        config (GENIE3Config): The configuration object for the GENIE3 model.
        predicted_network (pl.LazyFrame): The predicted network.
        output_dir (Path, optional): The directory where the results will be saved. Defaults to "results".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, output_dir)
    write_network(predicted_network, output_dir / "predicted_network.tsv")


def write_results_full_pipeline(
    config: GENIE3Config,
    auroc: float,
    auprc: float,
    pos_frac: float,
    fpr: NDArray,
    tpr: NDArray,
    recall: NDArray,
    precision: NDArray,
    predicted_network: pl.LazyFrame,
    reference_network: pl.LazyFrame,
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
        predicted_network (pl.LazyFrame): The predicted network.
        reference_network (pl.LazyFrame): The reference network.
        roc_curve_plot (Figure): The ROC curve plot as a matplotlib Figure.
        precision_recall_curve_plot (Figure): The precision-recall curve plot as a matplotlib Figure.
        output_dir (Path): The directory where the results will be saved.
    """
    write_config(config, output_dir)
    write_metrics(auroc, auprc, pos_frac, output_dir)
    write_ndarray(fpr, output_dir / "fpr.npy")
    write_ndarray(tpr, output_dir / "tpr.npy")
    write_ndarray(recall, output_dir / "recall.npy")
    write_ndarray(precision, output_dir / "precision.npy")
    write_network(predicted_network, output_dir / "predicted_network.tsv")
    write_network(reference_network, output_dir / "reference_network.tsv")
    write_plot(roc_curve_plot, output_dir / "roc_curve.png")
    write_plot(precision_recall_curve_plot, output_dir / "precision_recall_curve.png")
