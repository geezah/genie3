from pathlib import Path
from typing import Optional

from core.config import RegressorConfig
from pydantic import BaseModel, Field


class ParticipantConfig(BaseModel):
    gene_expressions_path: Path = Field(
        ..., description="Path to the gene expression data"
    )
    regressor: Optional[RegressorConfig] = Field(None)


class CoordinatorConfig(BaseModel):
    regressor: RegressorConfig
    transcription_factors_path: Optional[Path] = Field(
        None, description="Path to the transcription factor data"
    )


if __name__ == "__main__":
    from pprint import pprint

    import yaml

    CONFIG_PATH = Path("controller_data/clients/client_1/participant.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ParticipantConfig(**cfg_dict)
    pprint(cfg.model_dump())

    CONFIG_PATH = Path("controller_data/generic/coordinator.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CoordinatorConfig(**cfg_dict)
    pprint(cfg.model_dump())
