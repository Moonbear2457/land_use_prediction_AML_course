""""
test_plotter.py
This test is for documentation purposes only. The authors did not test the tests yet, as the required CPU performance was not locally avaiable. It is likely that the 'save_path' argument will not work, as it did not work in the notebook.
"""

import unittest
import os
from land_use_analysis.plotter import LandUseMapPlotter
from land_use_analysis.config import LAND_USE_COLORS, LABELS

class TestLandUseMapPlotter(unittest.TestCase):
    def setUp(self):
        self.raw_directory = "path/to/raw_directory"
        self.file_paths = [os.path.join(self.raw_directory, f) for f in os.listdir(self.raw_directory) if f.endswith('.shp')]
        self.years = [1990, 2000]  # Example years
        self.plotter = LandUseMapPlotter(self.file_paths, self.years, LAND_USE_COLORS, LABELS)

    def test_load_files(self):
        self.plotter.load_files()
        self.assertEqual(len(self.plotter.geodataframes), len(self.file_paths))
        self.assertTrue(all(isinstance(gdf, type(None)) or gdf.empty == False for gdf in self.plotter.geodataframes))

    def test_plot_maps(self):
        self.plotter.load_files()
        self.plotter.plot_maps(save_path="test_plot_maps.png", show_plot=False)
        self.assertTrue(os.path.exists("test_plot_maps.png"))
        os.remove("test_plot_maps.png")

    def test_plot_land_use_evolution(self):
        self.plotter.load_files()
        self.plotter.plot_land_use_evolution(save_path="test_land_use_evolution.png", show_plot=False)
        self.assertTrue(os.path.exists("test_land_use_evolution.png"))
        os.remove("test_land_use_evolution.png")

if __name__ == '__main__':
    unittest.main()
