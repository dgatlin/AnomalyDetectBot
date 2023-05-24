# todo - add file description and docstring
"""
Machine learning module for Python
==================================

FastAPI application for serving machine learning models.

** Explain the purpose of the module in the ML
   deployment Pipeline **

** Explain how this module fits in the system architecture **

"""

# todo - update this to use the config file

from typing import List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, ValidationError

from container.model_package.anomaly_model import config


# todo - add a test for this function
def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


# todo - add a test for this function
def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        pass

    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class AnomalyDataInputSchema(BaseModel):
    None
