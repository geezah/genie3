import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray


def plot_roc_curve(
    fpr: NDArray, tpr: NDArray, roc_auc: float, regressor_name: str = ""
) -> plt.Figure:
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (NDArray): Array of false positive rates.
        tpr (NDArray): Array of true positive rates.
        roc_auc (float): Area Under the ROC Curve (AUC) score.
        regressor_name (str, optional): Name of the regressor to be displayed in the plot label. Defaults to "".

    Returns:
        plt.Figure: The matplotlib figure object containing the ROC curve plot.
    """
    regressor_name = regressor_name + " " if regressor_name else ""
    fig, ax = plt.subplots()
    sns.lineplot(
        x=fpr,
        y=tpr,
        label=f"{regressor_name}AUC = {roc_auc:.4f}",
        linewidth=2,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="lower right")
    ax.grid(True)
    return fig


def plot_precision_recall_curve(
    recall: NDArray,
    precision: NDArray,
    pos_frac: float,
    auprc: float,
    regressor_name: str = "",
) -> plt.Figure:
    """
    Plots a precision-recall curve using the provided recall and precision values.

    Args:
        recall (NDArray): Array of recall values.
        precision (NDArray): Array of precision values.
        pos_frac (float): Fraction of positive samples.
        auprc (float): Area under the precision-recall curve.
        regressor_name (str, optional): Name of the regressor to be displayed in the plot label. Defaults to "".

    Returns:
        plt.Figure: The matplotlib figure object containing the precision-recall curve.
    """
    regressor_name = regressor_name + " " if regressor_name else ""
    fig, ax = plt.subplots()
    sns.lineplot(
        x=recall,
        y=precision,
        label=f"{regressor_name}AUC = {auprc:.4f}, %P: {pos_frac:.4f}",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall Gain")
    ax.set_ylabel("Precision Gain")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig
