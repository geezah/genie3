from unittest.mock import MagicMock

import numpy as np
import pytest

from genie3.regressor import (
    RegressorRegistry,
)


@pytest.fixture
def sample_data():
    """Create sample data for regressor testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.normal(0, 0.1, 100)
    return X, y


@pytest.fixture(params=["CPU", "GPU"])
def regressor_registry(request):
    """Parameterized fixture to get either CPU or GPU registry."""
    registry_name = request.param
    registry = getattr(RegressorRegistry, registry_name)

    # Skip GPU registry tests when not available
    if registry_name == "GPU" and not registry.available():
        pytest.skip(f"{registry_name} regressors not available")

    return registry


@pytest.fixture
def all_regressor_classes():
    """Get all available regressor classes from all registries."""
    regressors = {}

    # Add CPU regressors
    regressors.update(RegressorRegistry.CPU.list())

    # Add GPU regressors if available
    if RegressorRegistry.GPU.available():
        regressors.update(RegressorRegistry.GPU.list())

    return regressors


class TestRegressorRegistry:
    """Tests related to the regressor registry functionality."""

    def test_get_unknown_regressor(
        self, regressor_registry: RegressorRegistry
    ):
        """Test that the registry raises an error for unknown regressor types."""
        with pytest.raises(
            ValueError,
            match=f"Unknown {regressor_registry.__name__} regressor type: UnknownRegressor",
        ):
            regressor_registry.get("UnknownRegressor")

    def test_register_new_regressor(self):
        """Test registering a new regressor class."""
        mock_regressor = MagicMock()

        # Register and verify
        RegressorRegistry.register("MockRegressor")(mock_regressor)
        assert RegressorRegistry.get("MockRegressor") == mock_regressor

        # Clean up
        RegressorRegistry._regressors.pop("MockRegressor")

    @pytest.mark.parametrize(
        "registry_method",
        [RegressorRegistry.CPU.register, RegressorRegistry.GPU.register],
    )
    def test_register_in_specific_registry(self, registry_method):
        """Test registering regressors in specific registries."""
        registry_name = registry_method.__self__.__name__
        mock_regressor = MagicMock()
        test_name = f"Mock{registry_name}Regressor"

        # Register in specific registry
        registry_method(test_name)(mock_regressor)

        # Verify registration
        assert (
            getattr(RegressorRegistry, registry_name).get(test_name)
            == mock_regressor
        )
        assert RegressorRegistry.get(test_name) == mock_regressor

        # Clean up
        getattr(RegressorRegistry, registry_name)._regressors.pop(test_name)
        RegressorRegistry._regressors.pop(test_name)

    def test_gpu_availability(self):
        """Test that the GPU availability check works correctly."""
        # This test always passes - it just verifies the method runs without error
        is_available = RegressorRegistry.GPU.available()
        assert isinstance(
            is_available, bool
        ), "available() should return a boolean value"

        # If GPU is available, ensure there are GPU regressors registered
        if is_available:
            assert (
                len(RegressorRegistry.GPU.list()) > 0
            ), "GPU is available but no GPU regressors are registered"


class TestRegressorConfigurations:
    """Tests related to regressor configurations."""

    def test_configuration_structure(self, all_regressor_classes):
        """Test that all regressors have properly structured configurations."""
        for name, regressor_class in all_regressor_classes.items():
            assert hasattr(
                regressor_class, "DefaultConfiguration"
            ), f"{name} has no DefaultConfiguration"

            config = regressor_class.DefaultConfiguration
            assert (
                "init_params" in config
            ), f"{name} configuration missing init_params"
            assert (
                "fit_params" in config
            ), f"{name} configuration missing fit_params"

    def test_configuration_validity(self, regressor_registry, sample_data):
        """Test that configurations can create working regressors."""
        X, y = sample_data

        for name, regressor_class in regressor_registry.list().items():
            # Create and fit with default configuration
            regressor = regressor_class(
                regressor_class.DefaultConfiguration["init_params"]
            )
            regressor.fit(
                X, y, regressor_class.DefaultConfiguration["fit_params"]
            )

            # Verify feature importances
            importances = regressor.feature_importances_
            assert importances.shape == (
                X.shape[1],
            ), f"{name} has wrong feature importances shape"
            assert np.isclose(
                np.sum(importances), 1.0
            ), f"{name} feature importances don't sum to 1"


class TestRegressorImplementations:
    """Tests for concrete regressor implementations."""

    def test_regressor_protocol_compliance(self, regressor_registry):
        """Test that all regressors comply with the required interface."""
        for name, regressor_class in regressor_registry.list().items():
            # Check required class attributes
            assert hasattr(
                regressor_class, "fit"
            ), f"{name} missing fit method"
            assert hasattr(
                regressor_class, "feature_importances_"
            ), f"{name} missing feature_importances_"
            assert hasattr(
                regressor_class, "DefaultConfiguration"
            ), f"{name} missing DefaultConfiguration"

            # Create instance and check instance attributes
            regressor = regressor_class()
            assert hasattr(
                regressor, "fit"
            ), f"{name} instance missing fit method"

            # Check that feature_importances_ raises error when not fitted
            with pytest.raises(
                ValueError, match="Model has not been fitted yet"
            ):
                _ = regressor.feature_importances_
