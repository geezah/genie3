from typing import Any, Dict

from numpy.typing import NDArray
from sklearn.ensemble import (
    GradientBoostingRegressor as _GradientBoostingRegressor,
)

from .protocol import RegressorProtocol

DefaultGradientBoostingConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_features": 0.1,
        "n_iter_no_change": 25,
        "random_state": 42,
    },
    "fit_params": {},
}


class GradientBoostingRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultGradientBoostingConfiguration

    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _GradientBoostingRegressor = (
            _GradientBoostingRegressor(
                **init_params
                if init_params
                else DefaultGradientBoostingConfiguration["init_params"]
            )
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        self.regressor.fit(
            X,
            y,
            **fit_params
            if fit_params
            else DefaultGradientBoostingConfiguration["fit_params"],
        )
        self.feature_importances = self.regressor.feature_importances_
