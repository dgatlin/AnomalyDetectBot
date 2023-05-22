from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from container.model_package.anomaly_model.config.core import config

anomaly_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute variables with string missing
        (
            # todo add the function
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
            ),
        ),
        (
            # todo add the function
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        ("scaler", MinMaxScaler()),
        (
            "model",
            RandomForestClassifier(),
        ),
    ]
)
