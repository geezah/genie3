from typing import Type, Dict
from .protocol import RegressorProtocol
from .extratrees import DefaultExtraTreesConfiguration, ExtraTreesRegressor
from .gradientboosting import (
    DefaultGradientBoostingConfiguration,
    GradientBoostingRegressor,
)
from .lightgbm import DefaultLightGBMConfiguration, LGBMRegressor
from .randomforest import (
    DefaultRandomForestConfiguration,
    RandomForestRegressor,
)


class RegressorRegistry:
    _regressors: Dict[str, Type[RegressorProtocol]] = {
        "RandomForestRegressor": RandomForestRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "LGBMRegressor": LGBMRegressor,
    }

    @classmethod
    def register(
        cls, name: str, regressor_class: Type[RegressorProtocol]
    ) -> None:
        """Register a new regressor class."""
        cls._regressors[name] = regressor_class

    @classmethod
    def get(cls, name: str) -> Type[RegressorProtocol]:
        """Get a regressor class by name."""
        if name not in cls._regressors:
            raise ValueError(f"Unknown regressor type: {name}")
        return cls._regressors[name]


# Replace the dictionary with the registry
RegressorFactory = RegressorRegistry

ConfigurationFactory = {
    "RandomForestRegressor": DefaultRandomForestConfiguration,
    "ExtraTreesRegressor": DefaultExtraTreesConfiguration,
    "GradientBoostingRegressor": DefaultGradientBoostingConfiguration,
    "LGBMRegressor": DefaultLightGBMConfiguration,
}

__all__ = ["RegressorFactory", "ConfigurationFactory"]
