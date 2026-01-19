from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

import cupy as cp
from cupy.typing import ArrayLike
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor

# Workaround for bug: https://github.com/cupy/cupy/issues/9091
try:
    cp.cuda.is_available()
except CUDARuntimeError:
    pass
CUDA_AVAILABLE: bool = cp.cuda.is_available()

CUML_AVAILABLE: bool = False
if CUDA_AVAILABLE:
    try:
        from cuml.ensemble import (
            RandomForestRegressor as _CuRandomForestRegressor,
        )
        from cuml.explainer import TreeExplainer
    except ImportError:
        pass


class BaseRegressor(ABC):
    DefaultConfiguration: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._feature_importances_ = None

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit the regressor model.

        Must be implemented by all subclasses.
        """

    @staticmethod
    @abstractmethod
    def convert_inputs(
        gene_expressions: ArrayLike,
        transcription_factor_indices: ArrayLike,
        importance_matrix: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert input types to the appropriate format for the regressor."""
        pass

    @property
    def feature_importances_(self) -> ArrayLike:
        if (
            not hasattr(self, "_feature_importances_")
            or self._feature_importances_ is None
        ):
            raise ValueError(
                "Model has not been fitted yet. Therefore, no feature importances available."
            )
        return self._feature_importances_

    @feature_importances_.setter
    def feature_importances_(self, value: ArrayLike) -> None:
        self._feature_importances_ = value


class RegressorRegistry:
    _regressors: Dict[str, Type[BaseRegressor]] = {}

    # Create separate registries for CPU and GPU
    class CPU:
        _regressors: Dict[str, Type[BaseRegressor]] = {}

        @classmethod
        def register(
            cls, name: Optional[str] = None
        ) -> Callable[[Type[BaseRegressor]], Type[BaseRegressor]]:
            """Decorator to register a new CPU-based regressor class."""

            def decorator(
                regressor_class: Type[BaseRegressor],
            ) -> Type[BaseRegressor]:
                key = name if name is not None else regressor_class.__name__
                cls._regressors[key] = regressor_class
                # Also register in the main registry
                RegressorRegistry._regressors[key] = regressor_class
                return regressor_class

            return decorator

        @classmethod
        def get(cls, name: str) -> Type[BaseRegressor]:
            """Retrieve a registered CPU regressor class by name."""
            if name not in cls._regressors:
                raise ValueError(f"Unknown CPU regressor type: {name}")
            return cls._regressors[name]

        @classmethod
        def list(cls) -> Dict[str, Type[BaseRegressor]]:
            return dict(cls._regressors)

    class GPU:
        _regressors: Dict[str, Type[BaseRegressor]] = {}

        @classmethod
        def register(
            cls, name: Optional[str] = None
        ) -> Callable[[Type[BaseRegressor]], Type[BaseRegressor]]:
            """Decorator to register a new GPU-based regressor class."""

            def decorator(
                regressor_class: Type[BaseRegressor],
            ) -> Type[BaseRegressor]:
                key = name if name is not None else regressor_class.__name__
                cls._regressors[key] = regressor_class
                # Also register in the main registry
                RegressorRegistry._regressors[key] = regressor_class
                return regressor_class

            return decorator

        @classmethod
        def get(cls, name: str) -> Type[BaseRegressor]:
            """Retrieve a registered GPU regressor class by name."""
            if name not in cls._regressors:
                raise ValueError(f"Unknown GPU regressor type: {name}")
            return cls._regressors[name]

        @classmethod
        def list(cls) -> Dict[str, Type[BaseRegressor]]:
            return dict(cls._regressors)

        @classmethod
        def available(cls) -> bool:
            """Check if GPU regressors are available."""
            try:
                import cupy.cuda

                return cupy.cuda.runtime.getDeviceCount() > 0
            except Exception:
                return False

    @classmethod
    def register(
        cls, name: Optional[str] = None
    ) -> Callable[[Type[BaseRegressor]], Type[BaseRegressor]]:
        """Decorator to register a new regressor class in the main registry only."""

        def decorator(
            regressor_class: Type[BaseRegressor],
        ) -> Type[BaseRegressor]:
            key = name if name is not None else regressor_class.__name__
            cls._regressors[key] = regressor_class
            return regressor_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRegressor]:
        """Retrieve a registered regressor class by name."""
        if name not in cls._regressors:
            raise ValueError(f"Unknown regressor type: {name}")
        return cls._regressors[name]

    @classmethod
    def list(cls) -> Dict[str, Type[BaseRegressor]]:
        return dict(cls._regressors)


DefaultExtraTreesConfiguration = {
    "init_params": {
        "random_state": 42,
        "n_jobs": -1,
    },
    "fit_params": {},
}

DefaultRandomForestConfiguration = {
    "init_params": {
        "random_state": 42,
        "n_jobs": -1,
    },
    "fit_params": {},
}

# Define specific configuration for CUDA-based random forest
DefaultCuRandomForestConfiguration = {
    "init_params": {
        "random_state": 42,
    },  # No n_jobs for cuML
    "fit_params": {},
}


@RegressorRegistry.CPU.register("ExtraTreesRegressor")
class ExtraTreesRegressor(BaseRegressor):
    DefaultConfiguration = DefaultExtraTreesConfiguration

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        params = (
            init_params
            if init_params
            else DefaultExtraTreesConfiguration["init_params"]
        )
        self.regressor: _ExtraTreesRegressor = _ExtraTreesRegressor(**params)

    @staticmethod
    def convert_inputs(
        gene_expressions: ArrayLike,
        transcription_factor_indices: ArrayLike,
        importance_matrix: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert input types to the appropriate format for the regressor."""
        gene_expressions = cp.asnumpy(gene_expressions)
        transcription_factor_indices = cp.asnumpy(transcription_factor_indices)
        importance_matrix = cp.asnumpy(importance_matrix)
        return (
            gene_expressions,
            transcription_factor_indices,
            importance_matrix,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fit_params = (
            fit_params if fit_params else DefaultExtraTreesConfiguration["fit_params"]
        )
        self.regressor.fit(X, y, **fit_params)
        self._feature_importances_ = self.regressor.feature_importances_


@RegressorRegistry.CPU.register("RandomForestRegressor")
class RandomForestRegressor(BaseRegressor):
    DefaultConfiguration = DefaultRandomForestConfiguration

    def __init__(
        self,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        params = (
            init_params
            if init_params
            else DefaultRandomForestConfiguration["init_params"]
        )
        self.regressor: _RandomForestRegressor = _RandomForestRegressor(**params)

    @staticmethod
    def convert_inputs(
        gene_expressions: ArrayLike,
        transcription_factor_indices: ArrayLike,
        importance_matrix: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert input types to the appropriate format for the regressor."""
        gene_expressions = cp.asnumpy(gene_expressions)
        transcription_factor_indices = cp.asnumpy(transcription_factor_indices)
        importance_matrix = cp.asnumpy(importance_matrix)
        return (
            gene_expressions,
            transcription_factor_indices,
            importance_matrix,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        fit_params = (
            fit_params if fit_params else DefaultRandomForestConfiguration["fit_params"]
        )
        self.regressor.fit(X, y, **fit_params)
        self._feature_importances_ = self.regressor.feature_importances_


if CUDA_AVAILABLE:

    @RegressorRegistry.GPU.register("CuRandomForestRegressor")
    class CuRandomForestRegressor(BaseRegressor):
        DefaultConfiguration = DefaultCuRandomForestConfiguration

        def __init__(
            self,
            init_params: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__()
            params = (
                init_params
                if init_params
                else DefaultCuRandomForestConfiguration["init_params"]
            )
            self.regressor = _CuRandomForestRegressor(**params)

        @staticmethod
        def convert_inputs(
            gene_expressions: ArrayLike,
            transcription_factor_indices: ArrayLike,
            importance_matrix: ArrayLike,
        ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
            """Convert input types to the appropriate format for the regressor."""
            gene_expressions = cp.asarray(gene_expressions)
            transcription_factor_indices = cp.asarray(transcription_factor_indices)
            importance_matrix = cp.asarray(importance_matrix)
            return (
                gene_expressions,
                transcription_factor_indices,
                importance_matrix,
            )

        def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            fit_params: Optional[Dict[str, Any]] = None,
        ) -> None:
            fit_params = (
                fit_params
                if fit_params
                else DefaultCuRandomForestConfiguration["fit_params"]
            )
            X, y = cp.asarray(X), cp.asarray(y)
            self.regressor.fit(X, y, **fit_params)
            # Note: cuML does not provide feature importances directly, so we use SHAP values instead.
            explainer = TreeExplainer(model=self.regressor)
            shap_values: ArrayLike = explainer.shap_values(X)
            abs_shap_values = cp.abs(shap_values)
            mean_abs_shap_values = cp.mean(abs_shap_values, axis=0)
            normalized_shap_values = mean_abs_shap_values / cp.sum(mean_abs_shap_values)
            self._feature_importances_ = normalized_shap_values


__all__ = ["RegressorRegistry"]
