from datetime import datetime
from pathlib import Path

from typer import Typer
from yaml import safe_load

from genie3.config import GENIE3Config
from genie3.data import init_grn_dataset
from genie3.eval import prepare_evaluation, run_evaluation
from genie3.genie3 import run
from genie3.plot import plot_precision_recall_curve, plot_roc_curve
from genie3.utils import (
    write_results_full_pipeline,
    write_results_inference_only,
)

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    cfg_path: Path,
):
    with open(cfg_path, "r") as f:
        cfg = safe_load(f)
    cfg = GENIE3Config.model_validate(cfg)
    grn_dataset = init_grn_dataset(
        cfg.data.gene_expressions_path,
        cfg.data.transcription_factors_path,
        cfg.data.reference_network_path,
    )
    predicted_network = run(grn_dataset, cfg.regressor)

    output_dir = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.data.reference_network_path is None:
        write_results_inference_only(cfg, predicted_network, output_dir)
        return
    y_preds, y_true = prepare_evaluation(
        predicted_network, grn_dataset.reference_network
    )
    results = run_evaluation(y_preds, y_true)
    roc_curve_plot = plot_roc_curve(
        results.fpr,
        results.tpr,
        results.auroc,
        regressor_name=cfg.regressor.name,
    )
    precision_recall_curve_plot = plot_precision_recall_curve(
        results.recall,
        results.precision,
        results.pos_frac,
        results.auprc,
        regressor_name=cfg.regressor.name,
    )
    write_results_full_pipeline(
        cfg,
        results.auroc,
        results.auprc,
        results.pos_frac,
        results.fpr,
        results.tpr,
        results.precision,
        results.recall,
        predicted_network,
        grn_dataset.reference_network,
        roc_curve_plot,
        precision_recall_curve_plot,
        output_dir,
    )


if __name__ == "__main__":
    app()
