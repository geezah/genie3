from typing import Any, Dict, Optional, Protocol

from cupy.typing import ArrayLike


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
