from typing import Tuple

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import auc, precision_recall_curve, roc_curve


class Results(BaseModel):
    """
    Container model to store and verify evaluation results of GENIE3.

    Attributes:
        auroc (float): Area under the ROC curve. Must be between 0 and 1.
        auprc (float): Area under the precision-recall curve. Must be between 0 and 1.
        fpr (ArrayLike): False positive rates.
        tpr (ArrayLike): True positive rates.
        recall (ArrayLike): Recall scores.
        precision (ArrayLike): Precision scores.
        pos_frac (float): Fraction of positive examples in the dataset. Must be between 0 and 1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    auroc: float = Field(..., description="Area under the ROC curve", ge=0, le=1)
    auprc: float = Field(
        ..., description="Area under the precision-recall curve", ge=0, le=1
    )
    pos_frac: float = Field(
        ...,
        description="Fraction of positive examples in the dataset",
        ge=0,
        le=1,
    )
    fpr: ArrayLike = Field(..., description="False positive rates")
    tpr: ArrayLike = Field(..., description="True positive rates")
    recall: ArrayLike = Field(..., description="Recall scores")
    precision: ArrayLike = Field(..., description="Precision scores")


def prepare_evaluation(
    predicted_network: pl.LazyFrame, true_network: pl.LazyFrame
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Prepare the predicted and ground truth network for evaluation.

    Args:
        predicted_network (pl.LazyFrame): Predicted network with columns
            ``transcription_factor``, ``target_gene``, and ``importance``.
        true_network (pl.LazyFrame): Ground truth network with columns
            ``transcription_factor``, ``target_gene``, and ``label``.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Importance scores and ground-truth labels
            as NumPy arrays.
    """
    merged = (
        predicted_network.join(
            true_network,
            on=["transcription_factor", "target_gene"],
            how="right",
            coalesce=True,
        )
        .with_columns(
            pl.col("importance").fill_null(0.0),
        )
        .collect()
    )
    y_preds = merged["importance"].to_numpy()
    y_true = merged["label"].to_numpy()
    return y_preds, y_true


def run_evaluation(y_preds: ArrayLike, y_true: ArrayLike) -> Results:
    """
    Evaluate the predictions against the ground truth data.

    Args:
        y_preds (ArrayLike): Predicted importance scores
        y_true (ArrayLike): Ground truth labels
    Returns:
        Tuple[float, float]: AUROC and AUPRC scores
    """
    pos_frac: float = y_true.sum() / len(y_true)
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_preds))
    precision, recall, _ = precision_recall_curve(
        np.asarray(y_true), np.asarray(y_preds)
    )
    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)
    return Results(
        auroc=auroc,
        auprc=auprc,
        pos_frac=pos_frac,
        fpr=fpr,
        tpr=tpr,
        recall=recall,
        precision=precision,
    )
