from yaml import safe_load

import gradio as gr
from genie3.config import GENIE3Config
from genie3.data import init_grn_dataset
from genie3.genie3 import run


def main(
    gene_expressions,
    transcription_factors,
    reference_network,
    cfg
):
    cfg_path = cfg.name
    gene_expressions_path = gene_expressions.name
    transcription_factors_path = transcription_factors.name
    reference_network_path = reference_network.name

    with open(cfg_path, "r") as f:
        cfg = safe_load(f)

    cfg = GENIE3Config.model_validate(cfg)
    grn_dataset = init_grn_dataset(
        gene_expressions_path,
        transcription_factors_path,
        reference_network_path,
    )
    predicted_network = run(grn_dataset, cfg.regressor)
    return predicted_network


gr.Interface(
    fn=main,
    inputs=[
        gr.File(label="Gene Expression Data"),
        gr.File(label="Transcription Factors"),
        gr.File(label="Reference Network"),
        gr.File(label="Regressor Configuration (YAML)"),
    ],
    outputs=[
        gr.DataFrame(label="Predicted Network", headers=["Transcription Factor", "Target Gene", "Score"]),
    ],
    title="GENIE3",
    description="Generate a gene regulatory network from gene expression data and transcription factors.",
    examples=[
        ["local_data/processed/dream_five/net1_in-silico/gene_expression_data.tsv",
         "local_data/processed/dream_five/net1_in-silico/transcription_factors.tsv",
         "local_data/processed/dream_five/net1_in-silico/reference_network_data.tsv",
         "configs/extratrees.yaml",
        ],
    ],
).launch()

