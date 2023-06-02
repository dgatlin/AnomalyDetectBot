import unittest
from container.model_package.anomaly_model.processing.data_manager import load_pipeline
from container.model_package.anomaly_model.predict_pipe import make_prediction


class TestYourModule(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.input_data = {"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]}

    def test_make_prediction(self):
        # Load the pipeline
        pipeline_file_name = "anomaly_model_output_v_version.pkl"
        _anom_pipe = load_pipeline(file_name=pipeline_file_name)

        # Test make_prediction function
        expected_predictions = [0, 0, 0]
        expected_version = "_version"
        expected_errors = None

        output = make_prediction(input_data=self.input_data)

        self.assertEqual(output["predictions"], expected_predictions)
        self.assertEqual(output["version"], expected_version)
        self.assertEqual(output["errors"], expected_errors)


if __name__ == "__main__":
    unittest.main()
