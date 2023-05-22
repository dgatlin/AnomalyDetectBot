import pytest

from anomaly_model.config.core import config
from anomaly_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)
