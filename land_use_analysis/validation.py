# validation.py
"""
Adapted and object oriented form of a script written by Alamanos (2003).

Reference:
Alamanos, A. (2023). A Cellular Automata Markov (CAM) model for future land use change prediction using GIS and Python. DOI: 10.13140/RG.2.2.20309.19688. Available at: https://github.com/Alamanos11/Land_uses_prediction
"""
import geopandas as gpd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Validation:
    """
    A class to validate spatial prediction data against ground truth data.
    """

    def __init__(self, truth_path: str, predicted_path: str, truth_label_col: str = 'grid_code', predicted_label_col: str = 'grid_code'):
        self.truth_path = truth_path
        self.predicted_path = predicted_path
        self.truth_label_col = truth_label_col
        self.predicted_label_col = predicted_label_col
        self.truth_data = None
        self.predicted_data = None
        self.merged_data = None
        self.metrics = {}

    def load_data(self) -> None:
        """Loads the ground truth and predicted data from the given file paths."""
        try:
            logger.info("Loading ground truth data from %s", self.truth_path)
            self.truth_data = gpd.read_file(self.truth_path)
            logger.info("Loading predicted data from %s", self.predicted_path)
            self.predicted_data = gpd.read_file(self.predicted_path)
        except Exception as e:
            logger.error("Error loading data: %s", e)
            raise ValueError(f"Error loading data: {e}")

    def preprocess_data(self) -> None:
        """Preprocesses the data by rounding coordinates and creating unique identifiers."""
        logger.info("Preprocessing data")
        self.truth_data['rounded_x'] = self.truth_data.geometry.x.round(6)
        self.truth_data['rounded_y'] = self.truth_data.geometry.y.round(6)
        self.predicted_data['rounded_x'] = self.predicted_data.geometry.x.round(6)
        self.predicted_data['rounded_y'] = self.predicted_data.geometry.y.round(6)

        self.truth_data['unique_id'] = self.truth_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')
        self.predicted_data['unique_id'] = self.predicted_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')

    def merge_data(self) -> None:
        """Merges the ground truth and predicted data on unique identifiers."""
        logger.info("Merging ground truth and predicted data")
        self.merged_data = self.truth_data.merge(self.predicted_data, on='unique_id', how='inner')

        if len(self.truth_data) != len(self.predicted_data):
            logger.warning("The number of points in truth and predicted datasets is not the same.")

        if len(self.merged_data) == 0:
            logger.error("There are no common points between the truth and predicted datasets.")
            raise ValueError("Error: There are no common points between the truth and predicted datasets.")

    def plot_data(self) -> None:
        """Plots the ground truth and predicted data on a map."""
        logger.info("Plotting data")
        ax = self.truth_data.plot(color='blue', label='Truth Data')
        self.predicted_data.plot(ax=ax, color='red', label='Predicted Data')
        plt.legend()
        plt.title('Truth vs Predicted Data')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def calculate_metrics(self) -> None:
        """Calculates various validation metrics comparing ground truth and predicted labels."""
        logger.info("Calculating metrics")
        truth_labels = self.merged_data[f'{self.truth_label_col}_x'].astype(int)
        predicted_labels = self.merged_data[f'{self.predicted_label_col}_y'].astype(int)

        self.metrics['Accuracy'] = accuracy_score(truth_labels, predicted_labels)
        self.metrics['MAE'] = mean_absolute_error(truth_labels, predicted_labels)
        self.metrics['RMSE'] = mean_squared_error(truth_labels, predicted_labels, squared=False)
        self.metrics['Cohen\'s Kappa'] = cohen_kappa_score(truth_labels, predicted_labels)
        self.metrics['Confusion Matrix'] = confusion_matrix(truth_labels, predicted_labels)
        self.metrics['Classification Report'] = classification_report(truth_labels, predicted_labels, output_dict=True)

    def display_metrics(self) -> None:
        """Displays the calculated validation metrics."""
        logger.info("Displaying metrics")
        df_metrics = pd.DataFrame(self.metrics['Classification Report']).transpose()
        df_metrics['Accuracy'] = self.metrics['Accuracy']
        df_metrics['MAE'] = self.metrics['MAE']
        df_metrics['RMSE'] = self.metrics['RMSE']
        df_metrics['Cohen\'s Kappa'] = self.metrics['Cohen\'s Kappa']
        df_confusion = pd.DataFrame(self.metrics['Confusion Matrix'])

        print("\nAccuracy, MAE, RMSE, Cohen's Kappa:\n", df_metrics)
        print("\nConfusion Matrix:\n", df_confusion)

    def validate(self) -> None:
        """Runs the complete validation process: loading data, preprocessing, merging, plotting, and calculating metrics."""
        logger.info("Starting validation process")
        self.load_data()
        self.preprocess_data()
        self.merge_data()
        self.plot_data()
        self.calculate_metrics()
        self.display_metrics()
        logger.info("Validation process completed")
