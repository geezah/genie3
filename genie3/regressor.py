from typing import Any, Callable, Dict, Optional, Protocol, Type

from cupy.typing import ArrayLike

try:
    from cuml.accel import install  # type: ignore

    install()
except ImportError:
    pass

from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor as _ExtraTreesRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor as _RandomForestRegressor,
)


class RegressorProtocol(Protocol):
    DefaultConfiguration: Dict[str, Dict[str, Any]]

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def fit(
        self, X: ArrayLike, y: ArrayLike, fit_params: Dict[str, Any]
    ) -> None: ...

    @property
    def feature_importances_(self) -> ArrayLike:
        if not hasattr(self, "_feature_importances_"):
            raise ValueError(
                "Model has not been fitted yet. Therefore, no feature importances available."
            )
        return self._feature_importances_

    @feature_importances_.setter
    def feature_importances_(self, value: ArrayLike) -> None:
        self._feature_importances_ = value


DefaultExtraTreesConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "max_features": 0.1,
        "n_jobs": 8,
    },
    "fit_params": {},
}

DefaultRandomForestConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "max_features": 0.1,
    },
    "fit_params": {},
}


class ExtraTreesRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultExtraTreesConfiguration

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.regressor: _ExtraTreesRegressor = _ExtraTreesRegressor(
            **init_params
            if init_params
            else DefaultExtraTreesConfiguration["init_params"]
        )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fit_params = (
            fit_params
            if fit_params
            else DefaultRandomForestConfiguration["fit_params"]
        )
        self.regressor.fit(X, y, **fit_params)
        self.feature_importances_ = self.regressor.feature_importances_


class RandomForestRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultRandomForestConfiguration

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.regressor: _RandomForestRegressor = _RandomForestRegressor(
            **init_params
            if init_params
            else DefaultRandomForestConfiguration["init_params"]
        )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fit_params = (
            fit_params
            if fit_params
            else DefaultRandomForestConfiguration["fit_params"]
        )
        self.regressor.fit(X, y, **fit_params)
        self.feature_importances_ = self.regressor.feature_importances_


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
