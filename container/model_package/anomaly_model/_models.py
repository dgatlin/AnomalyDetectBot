# todo - add file description and docstring
"""
Machine learning module for Python
==================================

 The actual implementation of the machine learning algorithm within its own
 Python package. This allows me to compartmentalize the logic of the model
 with the logic needed to run it in SageMaker, and to modify and test each
 part independently. Then, the model can be reused in other environments as well.


"""


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import typing as t


# from container.model_package.anomaly_model import __version__ as _version
from container.model_package.anomaly_model.config.core import config
from container.model_package.anomaly_model.processing.data_manager import load_pipeline
from container.model_package.anomaly_model.processing.validation import validate_inputs
from container.model_package.anomaly_model.processing.data_manager import (
    load_dataset,
    save_pipeline,
)

# pipeline_file_name = f"{config.app_config.pipeline_save_file}{'_version'}.pkl"
# _anomaly_pipe = load_pipeline(file_name=pipeline_file_name)


class AnomalyModel:
    def __init__(self):
        model_name = "test"

    # todo - add a test for this function
    def run_training(self) -> None:
        """Train the model."""

        # read training data
        data = load_dataset(file_name=config.app_config.training_data_file)

        # divide train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data[config.model_config.features],  # predictors
            data[config.model_config.target],
            test_size=config.model_config.test_size,
            # we are setting the random seed here
            # for reproducibility
            random_state=config.model_config.random_state,
        )
        y_train = np.log(y_train)

        # fit model
        # _anomaly_pipe.fit(X_train, y_train)

        # persist trained model
        # save_pipeline(pipeline_to_persist=_anomaly_pipe)

        return self

    # todo - add a test for this function
    def make_prediction(
        self,
        *,
        input_data: t.Union[pd.DataFrame, dict],
    ) -> dict:
        """Make a prediction using a saved model pipeline."""

        data = pd.DataFrame(input_data)
        validated_data, errors = validate_inputs(input_data=data)
        results = {"predictions": None, "version": "_version", "errors": errors}

        if not errors:
            predictions = _anomaly_pipe.predict(
                X=validated_data[config.model_config.features]
            )
            results = {
                "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
                "version": "_version",
                "errors": errors,
            }

        return results

    def adb_evaluate_model(self):
        """
        Returns scores from cross validation evaluation on the malicious / benign classifier
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.clf_X, self.clf_y, test_size=0.2, random_state=0
        )
        y_train = np.array(y_train)
        eval_cls = RandomForestClassifier(n_estimators=100, max_features=0.2)
        eval_cls.fit(X_train, y_train)

        recall = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="recall")
        precision = cross_val_score(
            eval_cls, X_train, y_train, cv=5, scoring="precision"
        )
        accuracy = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="accuracy")
        f1_score = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring="f1_macro")

        return {
            "accuracy": accuracy,
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
        }
