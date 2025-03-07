from typing import Any, Dict

from lightgbm import LGBMRegressor as _LGBMRegressor
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .protocol import RegressorProtocol

DefaultLightGBMConfiguration = {
    "init_params": {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 3,
        "n_iter_no_change": 25,
        "early_stopping_min_delta": 0.0001,
        "random_state": 42,
        "importance_type": "gain",
        "subsample": 0.9,
        "colsample_bytree": 0.1,
        "verbose": -1,
    },
    "fit_params": {},
}


class LGBMRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultLightGBMConfiguration

    def __init__(self, **init_params: Dict[str, Any]):
        self.regressor: _LGBMRegressor = _LGBMRegressor(
            **init_params
            if init_params
            else DefaultLightGBMConfiguration["init_params"]
        )

    def fit(self, X: NDArray, y: NDArray, **fit_params: Dict[str, Any]) -> Any:
        # In the case that early_stopping_rounds is provided, we need to split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        regressor = self.regressor.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **fit_params
            if fit_params
            else DefaultLightGBMConfiguration["fit_params"],
        )
        importances = regressor.feature_importances_
        normalized_importances = importances / importances.sum()
        self.feature_importances = normalized_importances
