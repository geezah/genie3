from typing import Any, Dict

from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor as _ExtraTreesRegressor,
)

from .protocol import RegressorProtocol

DefaultExtraTreesConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "random_state": 42,
        "max_features": 0.1,
        "n_jobs": 8,
    },
    "fit_params": {},
}


class ExtraTreesRegressor(RegressorProtocol):
    
    DefaultConfiguration = DefaultExtraTreesConfiguration
    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _ExtraTreesRegressor = _ExtraTreesRegressor(
            **init_params
            if init_params
            else DefaultExtraTreesConfiguration["init_params"]
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        self.regressor.fit(
            X,
            y,
            **fit_params
            if fit_params
            else DefaultExtraTreesConfiguration["fit_params"],
        )
        self.feature_importances = self.regressor.feature_importances_
