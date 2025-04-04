from typing import Callable, Dict, Optional, Type

from .forest import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from .protocol import RegressorProtocol


class RegressorRegistry:
    _regressors: Dict[str, Type[RegressorProtocol]] = {
        "RandomForestRegressor": RandomForestRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
    }

    @classmethod
    def register(
        cls, name: Optional[str] = None
    ) -> Callable[[Type[RegressorProtocol]], Type[RegressorProtocol]]:
        """Decorator to register a new regressor class."""
        def decorator(
            regressor_class: Type[RegressorProtocol],
        ) -> Type[RegressorProtocol]:
            key = name or regressor_class.__name__
            cls._regressors[key] = regressor_class
            return regressor_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[RegressorProtocol]:
        """Retrieve a registered regressor class by name."""
        if name not in cls._regressors:
            raise ValueError(f"Unknown regressor type: {name}")
        return cls._regressors[name]

    @classmethod
    def list(cls) -> Dict[str, Type[RegressorProtocol]]:
        return dict(cls._regressors)


# Replace the dictionary with the registry
RegressorFactory = RegressorRegistry

__all__ = ["RegressorFactory"]
