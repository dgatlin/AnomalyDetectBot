from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from container.anomaly_model.anomaly_model.config.core import config

anomaly_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
            ),
        ),
        (
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
