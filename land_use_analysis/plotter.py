# plotter.py

"""
This module contains the LandUseMapPlotter class, which has been adapted from Alamanos (2023).

The LandUseMapPlotter class provides methods to visualize geospatial data using polygon shapefiles.
It includes the following functionalities:

1. plot_maps: Plots land use maps for each year with titles indicating the year and whether the map is "original" or "predicted".
2. plot_land_use_evolution: Plots the evolution of land use over time in a bar chart.

Reference:
Alamanos, A. (2023). A Cellular Automata Markov (CAM) model for future land use change prediction using GIS and Python. DOI: 10.13140/RG.2.2.20309.19688. Available at: https://github.com/Alamanos11/Land_uses_prediction
"""

import os
import time
import math
import logging
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from rasterio.plot import show
from .config import LAND_USE_COLORS, LAND_USE_LABELS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LandUseMapPlotter:
    def __init__(self, file_paths: List[str], years: List[int], land_use_colors: Dict[int, str] = LAND_USE_COLORS, land_use_labels: Dict[int, str] = LAND_USE_LABELS, gridcode_column: str = 'gridcode'):
        """
        Initialize the LandUseMapPlotter.

        Parameters:
        - file_paths (List[str]): List of file paths for shapefiles and raster files.
        - years (List[int]): List of years corresponding to the file paths.
        - land_use_colors (Dict[int, str]): Dictionary mapping land use codes to colors.
        - land_use_labels (Dict[int, str]): Dictionary mapping land use codes to labels.
        - gridcode_column (str): Column name for land use codes. Default is 'gridcode'.
        """
        if len(file_paths) != len(years):
            raise ValueError(f"Lengths of file_paths (length: {len(file_paths)}) and years (length: {len(years)}) do not match")
        if len(land_use_colors) != len(land_use_labels):
            raise ValueError("Lengths of land_use_colors and land_use_labels do not match")

        self.file_paths = file_paths
        self.years = years
        self.land_use_colors = land_use_colors
        self.land_use_labels = land_use_labels
        self.gridcode_column = gridcode_column
        self.geodataframes = []
        self.raster_datasets = []
        self.land_use_counts = pd.DataFrame(columns=[land_use_labels[i] for i in land_use_labels.keys()])

        # Sort the years and corresponding file paths
        sorted_data = sorted(zip(self.years, self.file_paths))
        self.years, self.file_paths = zip(*sorted_data)
        self.years = list(self.years)
        self.file_paths = list(self.file_paths)

    def load_files(self) -> None:
        """
        Load shapefiles and raster files based on the provided paths.

        This method iterates through the list of file paths and loads the corresponding 
        shapefiles or raster files. It handles errors and unsupported file formats 
        by logging appropriate messages.

        Example:
            plotter.load_files()
        """
        start_time = time.time()
        for path in self.file_paths:
            try:
                if path.endswith('.tif'):
                    self._load_raster_file(path)
                elif path.endswith('.shp'):
                    self._load_shapefile(path, start_time)
                else:
                    logger.warning(f"Unsupported file format for '{path}'")
                    self._append_none()
            except Exception as e:
                logger.error(f"Error loading file '{path}': {e}")
                self._append_none()

        logger.info(f"Total loaded geodataframes: {len([gdf for gdf in self.geodataframes if gdf is not None])}")

    def _load_raster_file(self, path: str) -> None:
        """
        Load a raster file and append it to the raster_datasets list.

        This method attempts to open a raster file and adds it to the list of raster datasets.
        If loading fails, it logs an error and appends None to maintain the list structure.

        Parameters:
        - path (str): The file path of the raster file.

        Example:
            self._load_raster_file('data/land_use_1990.tif')
        """
        try:
            self.raster_datasets.append(rasterio.open(path))
            self.geodataframes.append(None)
        except Exception as e:
            logger.error(f"Failed to load raster file '{path}': {e}")
            self._append_none()

    def _load_shapefile(self, path: str, start_time: float) -> None:
        """
        Load a shapefile and append it to the geodataframes list.

        This method attempts to read a shapefile and adds the GeoDataFrame to the list.
        If loading fails or if the specified gridcode column is not found, it logs an error
        and appends None to maintain the list structure.

        Parameters:
        - path (str): The file path of the shapefile.
        - start_time (float): The time when the file loading started.

        Example:
            self._load_shapefile('data/land_use_1990.shp', time.time())
        """
        try:
            load_time = time.time() - start_time
            gdf = gpd.read_file(path)
            if self.gridcode_column not in gdf.columns:
                raise ValueError(f"Column '{self.gridcode_column}' not found in shapefile '{path}'. Please rename the column to '{self.gridcode_column}' in your GIS system.")
            self.geodataframes.append(gdf)
            self.raster_datasets.append(None)
            logger.info(f"Loaded shapefile '{path}' with {len(gdf)} records in {load_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load shapefile '{path}': {e}")
            self._append_none()

    def _append_none(self) -> None:
        """
        Append None to both geodataframes and raster datasets lists.

        This method is used to maintain the structure of the lists when an error occurs
        or an unsupported file format is encountered.
        """
        self.raster_datasets.append(None)
        self.geodataframes.append(None)

    def plot_maps(self, figsize: tuple = (15, 10), num_cols: int = 3, save_path: Optional[str] = None, save_format: str = 'png', show_plot: bool = True) -> None:
        """
        Plot land use maps for each year and add a legend to the last plot.

        This method creates subplots for each shapefile or raster file and plots the 
        corresponding land use map. It also adds a legend to the plot and saves it 
        if a save_path is provided.

        Parameters:
        - figsize (tuple): Figure size for the plot.
        - num_cols (int): Number of columns in the subplot grid.
        - save_path (Optional[str]): Path to save the plot.
        - save_format (str): Format to save the plot ('png', 'jpg', 'pdf', etc.).
        - show_plot (bool): Whether to display the plot.

        Example:
            plotter.plot_maps(save_path='plots/land_use_maps.png', save_format='png', show_plot=False)
        """
        num_files = len(self.file_paths)
        num_rows = math.ceil(num_files / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()

        for i, (gdf, dataset, year, path) in enumerate(zip(self.geodataframes, self.raster_datasets, self.years, self.file_paths)):
            ax = axes[i]
            title_suffix = "Original" if "original" in path else "Predicted"
            if gdf is not None:
                self.plot_single_map(ax, gdf, year, title_suffix)
            elif dataset is not None:
                self.plot_single_raster(ax, dataset, year, title_suffix)

        if num_files < len(axes):
            self.add_legend(axes[num_files])
            for j in range(num_files + 1, len(axes)):
                fig.delaxes(axes[j])
        else:
            self.add_legend(axes[-1])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if save_path:
            try:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                    logger.info(f"Created directory: {os.path.dirname(save_path)}")

                plt.savefig(save_path, format=save_format)
                logger.info(f"Plot saved successfully to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save plot to {save_path}: {e}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_single_map(self, ax: plt.Axes, gdf: gpd.GeoDataFrame, year: int, title_suffix: str) -> None:
        """
        Plot a single shapefile map.

        This method plots a land use map for a given year from a GeoDataFrame. The map is colored
        according to the specified land use colors.

        Parameters:
        - ax (plt.Axes): Matplotlib Axes object.
        - gdf (gpd.GeoDataFrame): GeoDataFrame containing the shapefile data.
        - year (int): Year corresponding to the map.
        - title_suffix (str): Suffix to add to the title to distinguish original and predicted maps.

        Example:
            plotter.plot_single_map(ax, gdf, 1990, "Original")
        """
        for land_use_code, color in self.land_use_colors.items():
            gdf[gdf[self.gridcode_column] == land_use_code].plot(ax=ax, color=color)
        ax.set_title(f'Year {year} ({title_suffix})', fontsize=16, fontweight='bold')
        ax.axis('off')

    def plot_single_raster(self, ax: plt.Axes, dataset: rasterio.io.DatasetReader, year: int, title_suffix: str) -> None:
        """
        Plot a single raster map.

        This method plots a land use map for a given year from a raster dataset. The map is displayed
        using a colormap.

        Parameters:
        - ax (plt.Axes): Matplotlib Axes object.
        - dataset (rasterio.io.DatasetReader): Raster dataset.
        - year (int): Year corresponding to the map.
        - title_suffix (str): Suffix to add to the title to distinguish original and predicted maps.

        Example:
            plotter.plot_single_raster(ax, dataset, 2000, "Predicted")
        """
        show((dataset, 1), ax=ax, cmap='viridis', title=f'Year {year} ({title_suffix})')
        ax.axis('off')

    def add_legend(self, ax: plt.Axes) -> None:
        """
        Add a legend to the specified plot.

        This method adds a legend to the given Matplotlib Axes object, with labels and colors
        corresponding to the land use categories.

        Parameters:
        - ax (plt.Axes): Matplotlib Axes object.

        Example:
            plotter.add_legend(ax)
        """
        legend_patches = []
        for land_use_code, color in self.land_use_colors.items():
            legend_label = self.land_use_labels.get(land_use_code, f'Land Use {land_use_code}')
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
            legend_patches.append((patch, legend_label))

        ax.legend(*zip(*legend_patches), loc='center', fontsize=12)
        ax.axis('off')

    def count_land_use(self) -> None:
        """
        Count the number of occurrences for each land use category in shapefiles.

        This method counts the occurrences of each land use category in the loaded shapefiles
        and updates the land_use_counts DataFrame.

        Example:
            plotter.count_land_use()
        """
        counts_list = []
        for gdf, year in zip(self.geodataframes, self.years):
            if gdf is not None:
                if self.gridcode_column not in gdf.columns:
                    logger.warning(f"Column '{self.gridcode_column}' not found in data for year {year}. Skipping...")
                    continue
                category_counts = {self.land_use_labels[i]: (gdf[self.gridcode_column] == i).sum() for i in self.land_use_labels.keys()}
                counts_list.append(category_counts)
                logger.info(f"Year {year} counts: {category_counts}")

        if counts_list:
            counts_df = pd.DataFrame(counts_list)
            counts_df.index = self.years[:len(counts_list)]
            self.land_use_counts = pd.concat([self.land_use_counts, counts_df], axis=0).drop_duplicates()
            logger.info("Updated land use counts DataFrame:\n%s", self.land_use_counts)
        else:
            logger.info("No data to append to land_use_counts DataFrame.")

    def plot_land_use_evolution(self, save_path: Optional[str] = None, save_format: str = 'png', show_plot: bool = True) -> None:
        """
        Plot the evolution of land use categories over time.

        This method plots a bar chart showing the evolution of land use categories over the specified years.
        It saves the plot to a file if a save_path is provided.

        Parameters:
        - save_path (Optional[str]): Path to save the plot.
        - save_format (str): Format to save the plot ('png', 'jpg', 'pdf', etc.).
        - show_plot (bool): Whether to display the plot.

        Example:
            plotter.plot_land_use_evolution(save_path='plots/land_use_evolution.png', save_format='png', show_plot=False)
        """
        self.count_land_use()  # Call count_land_use to ensure land_use_counts is updated

        if self.land_use_counts.empty:
            logger.warning("The land use counts DataFrame is empty. No data to plot.")
            print("The land use counts DataFrame is empty. No data to plot.")  # Print a warning message
            return

        label_to_code = {label: code for code, label in self.land_use_labels.items()}

        print("land_use_counts DataFrame:\n", self.land_use_counts)  # Print the DataFrame for verification

        plt.figure(figsize=(10, 6))
        try:
            colors = [self.land_use_colors[label_to_code[col]] for col in self.land_use_counts.columns]
        except KeyError as e:
            logger.error(f"KeyError: {e}. Check the keys in land_use_colors and columns in land_use_counts DataFrame for mismatches.")
            print(f"KeyError: {e}. Check the keys in land_use_colors and columns in land_use_counts DataFrame for mismatches.")  # Print error message
            return

        self.land_use_counts.plot(kind='bar', stacked=True, color=colors)
        plt.title("Land Use Evolution Over Time")
        plt.xlabel("Year")
        plt.ylabel("Land Use Area (Number of Pixels)")
        plt.xticks(range(len(self.years)), self.years, rotation=45)
        plt.tight_layout()

        if save_path:
            try:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                    logger.info(f"Created directory: {os.path.dirname(save_path)}")

                plt.savefig(save_path, format=save_format)
                logger.info(f"Plot saved successfully to {save_path}")
                print(f"Plot saved successfully to {save_path}")  # Print success message
            except Exception as e:
                logger.error(f"Failed to save plot to {save_path}: {e}")
                print(f"Failed to save plot to {save_path}: {e}")  # Print error message

        if show_plot:
            plt.show()
        else:
            plt.close()


