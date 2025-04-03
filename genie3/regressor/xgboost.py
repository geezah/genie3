from typing import Any, Dict, Optional

from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from xgboost import (
    XGBRegressor as _XGBGradientBoostingRegressor,
)
from xgboost import (
    XGBRFRegressor as _XGBRandomForestRegressor,
)

from .protocol import RegressorProtocol

DefaultXGBRandomForestConfiguration = {
    "init_params": {
        "n_estimators": 100,
        "random_state": 42,
        "colsample_bytree": 0.8,
        "colsample_bynode": 1.0,
        "subsample": 0.8,
    },
    "fit_params": {},
}

DefaultXGBGradientBoostingConfiguration = {
    "init_params": {
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.1,
        "grow_policy": "lossguide",  # mimics lightgbm's growth policy
        "tree_method": "hist",
        "subsample": 1.0,
        "colsample_bytree": 0.1,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "random_state": 42,
        "importance_type": "gain",
        "early_stopping_rounds": 30,
    },
    "fit_params": {},
}


class XGBRandomForestRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultXGBRandomForestConfiguration

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
    ):
        self.init_params = (
            init_params
            if init_params
            else DefaultXGBRandomForestConfiguration["init_params"]
        )
        self.regressor: _XGBRandomForestRegressor = _XGBRandomForestRegressor(
            **self.init_params
        )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> NDArray:
        self.fit_params = (
            fit_params
            if fit_params
            else DefaultXGBRandomForestConfiguration["fit_params"]
        )
        self.regressor.fit(X, y, **self.fit_params)
        self.feature_importances = self.regressor.feature_importances_
        self.feature_importances = (
            self.feature_importances / self.feature_importances.sum()
        )


class XGBGradientBoostingRegressor(RegressorProtocol):
    DefaultConfiguration = DefaultXGBGradientBoostingConfiguration

    def __init__(self, init_params: Optional[Dict[str, Any]] = None) -> None:
        self.init_params = (
            init_params
            if init_params
            else DefaultXGBRandomForestConfiguration["init_params"]
        )
        self.regressor: _XGBGradientBoostingRegressor = (
            _XGBGradientBoostingRegressor(**self.init_params)
        )

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fit_params = (
            fit_params
            if fit_params
            else DefaultXGBRandomForestConfiguration["fit_params"]
        )
        if self.init_params["early_stopping_rounds"] is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.1,  # TODO: hardcoded for now
            )
            eval_set = [(X_val, y_val)]
            self.regressor.fit(
                X_train, y_train, eval_set=eval_set, **self.fit_params
            )
        else:
            self.regressor.fit(X_train, y_train, **self.fit_params)

        self.feature_importances = self.regressor.feature_importances_
        self.feature_importances = (
            self.feature_importances / self.feature_importances.sum()
        )
