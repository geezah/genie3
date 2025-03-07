from typing import Any, Dict

from numpy.typing import NDArray
from sklearn.ensemble import (
    RandomForestRegressor as _RandomForestRegressor,
)

from .protocol import RegressorProtocol

DefaultRandomForestConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "random_state": 42,
        "max_features": 0.1,
    },
    "fit_params": {},
}


class RandomForestRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultRandomForestConfiguration

    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _RandomForestRegressor = _RandomForestRegressor(
            **init_params
            if init_params
            else DefaultRandomForestConfiguration["init_params"]
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        self.regressor.fit(
            X,
            y,
            **fit_params
            if fit_params
            else DefaultRandomForestConfiguration["fit_params"],
        )
        self.feature_importances = self.regressor.feature_importances_
