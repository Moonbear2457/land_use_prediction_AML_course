""""
test_analysis.py
This test is for documentation purposes only. The authors did not test the tests yet, as the required CPU performance was not locally avaiable.
"""

import unittest
import numpy as np
import pandas as pd
from land_use_analysis.analysis import read_raster, calculate_change_matrix, calculate_transition_probabilities, apply_transition

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.before = np.array([[1, 2], [3, 4]]) # Dummy array
        self.after = np.array([[1, 2], [3, 4]]) # Dummy array

    def test_read_raster(self):
        data, transform = read_raster('path/to/raster.tif') # Raster has to be small size
        self.assertIsInstance(data, np.ndarray)

    def test_calculate_change_matrix(self):
        change_matrix = calculate_change_matrix(self.before, self.after)
        self.assertIsInstance(change_matrix, pd.DataFrame)

    def test_calculate_transition_probabilities(self):
        change_matrix = calculate_change_matrix(self.before, self.after)
        transition_matrix = calculate_transition_probabilities(change_matrix)
        self.assertIsInstance(transition_matrix, pd.DataFrame)

    def test_apply_transition(self):
        change_matrix = calculate_change_matrix(self.before, self.after)
        transition_matrix = calculate_transition_probabilities(change_matrix)
        new_land_use = apply_transition(self.before, transition_matrix)
        self.assertEqual(new_land_use.shape, self.before.shape)

if __name__ == '__main__':
    unittest.main()
