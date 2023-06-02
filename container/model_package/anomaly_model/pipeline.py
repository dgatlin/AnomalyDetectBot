# todo - add file description and docstring
"""
Machine learning module for Python
==================================

FastAPI application for serving machine learning models.

** Explain the purpose of the module in the ML
   deployment Pipeline **

** Explain how this module fits in the system architecture **

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

from container.model_package.anomaly_model.config.core import config

# todo - add a test for this function
anomaly_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute variables with string missing
        # (
        # todo add the function
        # "missing_imputation",
        # CategoricalImputer(
        #     imputation_method="missing",
        #     variables=config.model_config.categorical_vars_with_na_missing,
        # ),
        # ),
        # (
        # todo add the function
        #  "mean_imputation",
        #  MeanMedianImputer(
        #      imputation_method="mean",
        #      variables=config.model_config.numerical_vars_with_na,
        #  ),
        # ),
        ("scaler", MinMaxScaler()),
        (
            "model",
            RandomForestClassifier(),
        ),
    ]
)
