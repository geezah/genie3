from pathlib import Path

from typer import Typer
from yaml import safe_load

import wandb
from genie3.config import GENIE3Config
from genie3.data import init_grn_dataset
from genie3.eval import prepare_evaluation, run_evaluation
from genie3.genie3 import run as run_genie3

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
    with wandb.init(
        project="genie3",
        config={
            "data.gene_expressions_path": cfg.data.gene_expressions_path,
            "data.transcription_factors_path": cfg.data.transcription_factors_path,
            "data.reference_network_path": cfg.data.reference_network_path,
            "regressor.name": cfg.regressor.name,
            "regressor.init_params": cfg.regressor.init_params,
            "regressor.fit_params": cfg.regressor.fit_params,
        },
    ) as run:
        predicted_network = run_genie3(grn_dataset, cfg.regressor)

        y_preds, y_true = prepare_evaluation(
            predicted_network, grn_dataset.reference_network
        )
        results = run_evaluation(y_preds, y_true)
        run.log({"auroc": results.auroc})


if __name__ == "__main__":
    app()
