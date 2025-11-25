from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import DataConfig, RegressorConfig
from pydantic import BaseModel, Field

from federated.simulation import SimulationStrategyType


class ParticipantConfig(BaseModel):
    regressor: RegressorConfig
    data: DataConfig


class SimulationConfig(BaseModel):
    strategy: SimulationStrategyType = Field(..., description="Simulation strategy")


class SimCoordinatorConfig(BaseModel):
    regressor: RegressorConfig
    simulation: Optional[SimulationConfig] = Field(
        None, description="Simulation configuration"
    )

    def model_post_init(self, *args, **kwargs):
        # Convert single aggregation config to list if needed
        if not isinstance(self.aggregation, list):
            self.aggregation = [self.aggregation]


if __name__ == "__main__":
    from pprint import pprint

    import yaml

    CONFIG_PATH = Path("controller_data/generic/server.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimCoordinatorConfig.model_validate(cfg)
    pprint(cfg.model_dump())

    CONFIG_PATH = Path("controller_data/generic/client.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = ParticipantConfig.model_validate(cfg)
    pprint(cfg.model_dump())
