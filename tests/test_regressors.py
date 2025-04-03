import pytest
import numpy as np
from unittest.mock import MagicMock

from genie3.regressor import (
    RegressorFactory,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    XGBGradientBoostingRegressor, 
    XGBRandomForestRegressor
)

@pytest.fixture
def sample_data():
    """Create sample data for regressor testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.normal(0, 0.1, 100)
    return X, y


class TestRegressorFactory:
    def test_get_regressor(self):
        """Test that the factory returns the correct regressor class."""
        assert RegressorFactory.get("RandomForestRegressor") == RandomForestRegressor
        assert RegressorFactory.get("ExtraTreesRegressor") == ExtraTreesRegressor
        assert RegressorFactory.get("GradientBoostingRegressor") == GradientBoostingRegressor
        assert RegressorFactory.get("XGBRandomForestRegressor") == XGBRandomForestRegressor
        assert RegressorFactory.get("XGBGradientBoostingRegressor") == XGBGradientBoostingRegressor

    def test_get_unknown_regressor(self):
        """Test that the factory raises an error for unknown regressor types."""
        with pytest.raises(ValueError, match="Unknown regressor type"):
            RegressorFactory.get("UnknownRegressor")

    def test_register_regressor(self):
        """Test registering a new regressor class."""
        # Create a mock regressor class
        mock_regressor = MagicMock()
        
        # Register the mock regressor
        RegressorFactory.register("MockRegressor", mock_regressor)
        
        # Check that the mock regressor can be retrieved
        assert RegressorFactory.get("MockRegressor") == mock_regressor
        
        # Clean up by removing the mock regressor
        RegressorFactory._regressors.pop("MockRegressor")


class TestConfigurationFactory:
    def test_configuration_exists(self):
        """Test that configurations exist for all regressor types."""
        for regressor_class in RegressorFactory._regressors.values():
            assert regressor_class.DefaultConfiguration is not None
            
            # Check that the configuration has the expected structure
            config = regressor_class.DefaultConfiguration
            assert "init_params" in config
            assert "fit_params" in config

    def test_configuration_validity(self, sample_data):
        """Test that all configurations are valid and can be used to create and fit regressors."""
        X, y = sample_data
        
        for regressor_class in RegressorFactory._regressors.values():
            
            # Create a regressor with the configuration
            regressor = regressor_class(regressor_class.DefaultConfiguration["init_params"])
            
            # Fit the regressor with the configuration
            regressor.fit(X, y, regressor_class.DefaultConfiguration["fit_params"])
            
            # Check that feature importances are available
            assert hasattr(regressor, "feature_importances")
            assert regressor.feature_importances.shape == (X.shape[1],)
            assert np.isclose(np.sum(regressor.feature_importances), 1.0)


class TestRegressors:
    def test_regressor_protocol_compliance(self):
        """Test that all regressors comply with the RegressorProtocol interface."""
        for regressor_class in RegressorFactory._regressors.values():
            # Check that the regressor class has the required methods
            assert hasattr(regressor_class, 'fit')
            assert hasattr(regressor_class, 'feature_importances')
            
            # Create an instance and check that it has the required attributes
            regressor = regressor_class()
            assert hasattr(regressor, 'fit')
            
            # Check that the feature_importances property raises an error when not fitted
            with pytest.raises(ValueError, match="Model has not been fitted yet"):
                _ = regressor.feature_importances 
    
    def test_custom_parameters(self, sample_data):
        """Test that custom parameters are passed to the regressor."""
        X, y = sample_data
        
        # Create regressor with custom parameters
        custom_params = {
            "n_estimators": 15,
            "max_depth": 5,
            "random_state": 123,
        }
        regressor = RandomForestRegressor(custom_params)
        
        # Check that the parameters were passed to the underlying regressor
        for param, value in custom_params.items():
            assert getattr(regressor.regressor, param) == value
        
        # Fit with custom fit parameters
        custom_fit_params = {"sample_weight": np.ones(len(y))}
        regressor.fit(X, y, custom_fit_params)
        
        # Feature importances should be available
        assert hasattr(regressor, "feature_importances")
    
    @pytest.mark.parametrize("regressor_class", [
        RandomForestRegressor,
    ])
    def test_regressors(self, regressor_class, sample_data):
        """Test implemented regressors."""
        X, y = sample_data
        
        # Create regressor with default parameters
        regressor = regressor_class()
        
        # Fit the regressor
        regressor.fit(X, y)
        
        # Check that feature importances are available and normalized
        importances = regressor.feature_importances
        assert importances.shape == (X.shape[1],)
        assert np.isclose(np.sum(importances), 1.0)
        
        # Check that the most important features are the ones we used to generate the data
        assert importances[0] > 0.2  # Feature 0 should be important
        assert importances[2] > 0.2  # Feature 2 should be important