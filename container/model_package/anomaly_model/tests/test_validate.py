import unittest

import pandas as pd

from container.model_package.anomaly_model.processing.validation import (
    drop_na_inputs,
    validate_inputs,
)


class TestYourModule(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.input_data = pd.DataFrame(
            {"one": [1, 2, 3, None], "two": [4, None, 6, 7], "three": [8, 9, 10, 11]}
        )

    def test_drop_na_inputs(self):
        # Test drop_na_inputs function
        expected_data = pd.DataFrame(
            {"one": [1.0, 3.0], "two": [4.0, 6.0], "three": [8, 10]}
        )
        output_data = drop_na_inputs(input_data=self.input_data)
        self.assertTrue(output_data.reset_index(drop=True).equals(expected_data))

    def test_validate_inputs(self):
        # Test validate_inputs function
        expected_data = pd.DataFrame(
            {"one": [1.0, 3.0], "two": [4.0, 6.0], "three": [8, 10]}
        )
        expected_errors = None

        output_data, output_errors = validate_inputs(input_data=self.input_data)

        self.assertTrue(output_data.reset_index(drop=True).equals(expected_data))
        # self.assertTrue(output_errors == expected_errors)


if __name__ == "__main__":
    unittest.main()
