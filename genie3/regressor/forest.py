try: 
    from cuml.accel import install # type:ignore
    install()
except ImportError:
    pass

from typing import Any, Dict, Optional

from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor as _ExtraTreesRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor as _RandomForestRegressor,
)

from .protocol import RegressorProtocol

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
        use_gpu: bool = False,
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
        use_gpu: bool = False,
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
