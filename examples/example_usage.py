# example_usage.py

from land_use_analysis.plotter import LandUseMapPlotter
from land_use_analysis.analysis import main
from land_use_analysis.utils import extract_years
from land_use_analysis.validation import Validation
import os

# Define file paths and years
before_file = 'path/to/before_raster.tif'
after_file = 'path/to/after_raster.tif'
current_land_use_map = 'path/to/current_land_use_map.tif'
output_raster_path = 'path/to/save/output_raster.tif'

# Running the main analysis
change_matrix, transition_matrix = main(before_file, after_file, current_land_use_map, output_raster_path)

# Validation
truth_path = "path/to/truth_shapefile.shp"
predicted_path = "path/to/predicted_shapefile.shp"
validator = Validation(truth_path, predicted_path)
validator.validate()

# Visualization
RAW_DIRECTORY = "path/to/raw_directory"
file_paths_original = [os.path.join(RAW_DIRECTORY, f) for f in os.listdir(RAW_DIRECTORY) if f.endswith('.shp') and f.startswith('original')]
file_paths_predicted = [os.path.join(RAW_DIRECTORY, f) for f in os.listdir(RAW_DIRECTORY) if f.endswith('.shp') and f.startswith('predicted')]

# Extract years from the file names
years_original = extract_years(file_paths_original)
years_predicted = extract_years(file_paths_predicted)

# Create plotter instances and plot
plotter_original = LandUseMapPlotter(file_paths_original, years_original, LAND_USE_COLORS, LABELS)
plotter_original.load_files()
plotter_original.plot_maps()

# Plot land use evolution over time
plotter_original.plot_land_use_evolution(save_path='path/to/save/land_use_evolution.png', save_format='png', show_plot=True)

# Visualizing Validation
from land_use_analysis.visualization import TextProcessor

folder_path = 'path/to/validation/output/'
save_folder = 'path/to/save/validation/figures/'

processor = TextProcessor(folder_path, save_folder)
processor.process_files(as_percentage=True)
classification_report = processor.process_classification_reports()
processor.generate_yearly_report()

print(classification_report)
