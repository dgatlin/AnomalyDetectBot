# todo - add a test for this function
import numpy as np
from sklearn.model_selection import train_test_split

from container.model_package import config
from container.model_package.anomaly_model.pipeline import anomaly_pipe
from container.model_package.anomaly_model.processing.data_manager import (
    load_dataset,
    save_pipeline,
)


def run_training() -> None:
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
        random_state=config.model_config.random_seed,
    )
    y_train = y_train

    # fit model
    anomaly_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=anomaly_pipe)


run_training()
