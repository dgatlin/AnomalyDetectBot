import pandas as pd

from container.model_package.anomaly_model.config.core import config
from container.model_package.anomaly_model.processing.data_manager import load_pipeline
from container.model_package.anomaly_model.processing.validation import validate_inputs

pipeline_file_name = "anomaly_model_output_v_version.pkl"
_anom_pipe = load_pipeline(file_name=pipeline_file_name)


# todo - add a test for this function
def make_prediction(*, input_data) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": "_version", "errors": errors}

    if not errors:
        predictions = _anom_pipe.predict(X=validated_data[config.model_config.features])
        results = {
            "predictions": predictions.tolist(),
            "version": "_version",
            "errors": errors,
        }

    return results
