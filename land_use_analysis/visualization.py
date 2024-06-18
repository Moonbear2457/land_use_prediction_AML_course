# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd
from .config import LABELS, LAND_USE_COLORS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextProcessor:
    """
    A class to process and visualize text files containing confusion matrices and classification reports.
    """

    def __init__(self, folder_path: str, save_folder: str):
        """
        Initialize the TextProcessor with the folder paths for reading and saving files.

        Parameters:
        - folder_path (str): Path to the folder containing text files.
        - save_folder (str): Path to the folder where processed plots and reports will be saved.
        """
        self.folder_path = folder_path
        self.save_folder = save_folder
        self.labels = LABELS
        self.land_use_colors = LAND_USE_COLORS
        os.makedirs(save_folder, exist_ok=True)

        logger.info(f"TextProcessor initialized with folder_path: {folder_path} and save_folder: {save_folder}")

    def read_confusion_matrix(self, file_path: str) -> np.ndarray:
        """
        Read and parse a confusion matrix from a text file.

        Parameters:
        - file_path (str): Path to the text file containing the confusion matrix.

        Returns:
        - np.ndarray: Parsed confusion matrix as a NumPy array.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        matrix_start_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("Confusion Matrix:"):
                matrix_start_idx = idx + 1
                break

        matrix = []
        for line in lines[matrix_start_idx + 1:]:
            parts = line.strip().split()[1:]  # Skip the row label
            if all(part.isdigit() for part in parts):  # Ensure all parts are digits
                matrix.append(list(map(int, parts)))

        return np.array(matrix)

    def plot_confusion_matrix(self, matrix: np.ndarray, title: str, save_path: str) -> None:
        """
        Plot and save a confusion matrix.

        Parameters:
        - matrix (np.ndarray): Confusion matrix.
        - title (str): Title for the plot.
        - save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def calculate_false_negatives(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate the false negatives from a confusion matrix.

        Parameters:
        - matrix (np.ndarray): Confusion matrix.

        Returns:
        - np.ndarray: Array of false negatives for each class.
        """
        return matrix.sum(axis=1) - np.diag(matrix)

    def process_files(self, as_percentage: bool = False) -> None:
        """
        Process and plot confusion matrices from all text files in the specified folder.

        Parameters:
        - as_percentage (bool): Whether to display the false negatives as a percentage in a stacked plot.
        """
        txt_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))

        false_negatives_dict = {label: [] for label in self.labels}
        years = []

        for txt_file in txt_files:
            file_path = os.path.join(self.folder_path, txt_file)
            confusion_matrix = self.read_confusion_matrix(file_path)
            if confusion_matrix.size > 0:
                save_path = os.path.join(self.save_folder, f'{txt_file[:-4]}_confusion_matrix.png')
                self.plot_confusion_matrix(confusion_matrix, title=f'Confusion Matrix: {txt_file[:-4]}', save_path=save_path)

                false_negatives = self.calculate_false_negatives(confusion_matrix)
                for i, label in enumerate(self.labels):
                    false_negatives_dict[label].append(false_negatives[i])
                years.append(txt_file[:-4])

        self.plot_false_negatives(false_negatives_dict, years, as_percentage)

    def plot_false_negatives(self, false_negatives_dict: dict, years: list, as_percentage: bool) -> None:
        """
        Plot false negatives for each class across all files.

        Parameters:
        - false_negatives_dict (dict): Dictionary of false negatives for each class.
        - years (list): List of years corresponding to the files.
        - as_percentage (bool): Whether to display the false negatives as a percentage in a stacked plot.
        """
        if as_percentage:
            total_false_negatives_per_year = np.sum(list(false_negatives_dict.values()), axis=0)
            for label in self.labels:
                false_negatives_dict[label] = (np.array(false_negatives_dict[label]) / total_false_negatives_per_year) * 100

            bottom = np.zeros(len(years))
            plt.figure(figsize=(12, 8))
            for label in self.labels:
                plt.bar(years, false_negatives_dict[label], bottom=bottom, label=f'Class {label}', color=self.land_use_colors[label])
                bottom += np.array(false_negatives_dict[label])

            plt.xlabel('Years')
            plt.ylabel('Percentage of False Negatives')
            plt.title('Percentage of False Negatives for Each Land-Use Class Across Years')
        else:
            plt.figure(figsize=(12, 8))
            width = 0.1
            x = np.arange(len(years))
            for i, label in enumerate(self.labels):
                plt.bar(x + i * width, false_negatives_dict[label], width, label=f'Class {label}', color=self.land_use_colors[label])

            plt.xlabel('Years')
            plt.ylabel('Total False Negatives')
            plt.title('Total False Negatives for Each Land-Use Class Across Years')
            plt.xticks(x + width * (len(self.labels) - 1) / 2, years, rotation=45)

        plt.legend()
        plt.tight_layout()
        false_neg_plot_path = os.path.join(self.save_folder, 'total_false_negatives_clustered.png' if not as_percentage else 'total_false_negatives_stacked.png')
        plt.savefig(false_neg_plot_path)
        plt.close()
        logger.info(f"False negatives plot saved to {false_neg_plot_path}")

    def read_classification_report(self, file_path: str) -> pd.DataFrame:
        """
        Read and parse a classification report from a text file.

        Parameters:
        - file_path (str): Path to the text file containing the classification report.

        Returns:
        - pd.DataFrame: Parsed classification report as a pandas DataFrame.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        report_start_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("precision"):
                report_start_idx = idx
                break

        data = []
        for line in lines[report_start_idx + 1:]:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] in self.labels:  # Ensure there are enough parts and the first part is a label
                data.append(parts[:5])  # Extract precision, recall, f1-score, support, and the label

        return pd.DataFrame(data, columns=["class", "precision", "recall", "f1-score", "support"]).astype({"precision": float, "recall": float, "f1-score": float, "support": float})

    def process_classification_reports(self) -> pd.DataFrame:
        """
        Process and combine classification reports from all text files in the specified folder.

        Returns:
        - pd.DataFrame: Combined classification report as a pandas DataFrame.
        """
        txt_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))
        reports = []

        for txt_file in txt_files:
            file_path = os.path.join(self.folder_path, txt_file)
            report = self.read_classification_report(file_path)
            report["year"] = txt_file[:-4]
            reports.append(report)

        combined_report = pd.concat(reports, ignore_index=True).drop_duplicates(subset=["class", "year"])
        combined_report_path = os.path.join(self.save_folder, 'combined_classification_report.csv')
        combined_report.to_csv(combined_report_path, index=False)

        combined_report_html_path = os.path.join(self.save_folder, 'combined_classification_report.html')
        combined_report.to_html(combined_report_html_path, index=False)

        logger.info(f"Combined classification report saved to {combined_report_path} and {combined_report_html_path}")
        return combined_report

    def generate_yearly_report(self) -> None:
        """
        Generate and save yearly classification reports from the combined classification report.
        """
        combined_report = pd.read_csv(os.path.join(self.save_folder, 'combined_classification_report.csv'))

        years = combined_report['year'].unique()
        for year in years:
            yearly_report = combined_report[combined_report['year'] == year]
            yearly_report_html_path = os.path.join(self.save_folder, f'classification_report_{year}.html')
            yearly_report.to_html(yearly_report_html_path, index=False)
            logger.info(f"Yearly classification report for {year} saved to {yearly_report_html_path}")
