from typing import Dict, Type

from .protocol import RegressorProtocol

from .sklearn import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from .xgboost import XGBGradientBoostingRegressor, XGBRandomForestRegressor


class RegressorRegistry:
    _regressors: Dict[str, Type[RegressorProtocol]] = {
        "RandomForestRegressor": RandomForestRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRandomForestRegressor": XGBRandomForestRegressor,
        "XGBGradientBoostingRegressor": XGBGradientBoostingRegressor,
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

__all__ = ["RegressorFactory"]
