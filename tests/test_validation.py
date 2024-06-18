""""
test_validation.py
This test is for documentation purposes only. The authors did not test the tests yet, as the required CPU performance was not locally avaiable. Minimum required: 32 GB of RAM (according to the authors).
"""

import unittest
import os
from land_use_analysis.validation import Validation

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.truth_path = "path/to/truth_shapefile.shp"
        self.predicted_path = "path/to/predicted_shapefile.shp"
        self.validator = Validation(self.truth_path, self.predicted_path)

    def test_load_data(self):
        self.validator.load_data()
        self.assertIsNotNone(self.validator.truth_data)
        self.assertIsNotNone(self.validator.predicted_data)

    def test_preprocess_data(self):
        self.validator.load_data()
        self.validator.preprocess_data()
        self.assertIn('unique_id', self.validator.truth_data.columns)
        self.assertIn('unique_id', self.validator.predicted_data.columns)

    def test_merge_data(self):
        self.validator.load_data()
        self.validator.preprocess_data()
        self.validator.merge_data()
        self.assertIsNotNone(self.validator.merged_data)

    def test_calculate_metrics(self):
        self.validator.load_data()
        self.validator.preprocess_data()
        self.validator.merge_data()
        self.validator.calculate_metrics()
        self.assertIn('Accuracy', self.validator.metrics)
        self.assertIn('MAE', self.validator.metrics)
        self.assertIn('RMSE', self.validator.metrics)
        self.assertIn('Cohen\'s Kappa', self.validator.metrics)

if __name__ == '__main__':
    unittest.main()
