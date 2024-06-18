# land_use_analysis/analysis.py

"""
Adapted and enhanced funcionality from a script written by Alamanos (2003).

Reference:
Alamanos, A. (2023). A Cellular Automata Markov (CAM) model for future land use change prediction using GIS and Python. DOI: 10.13140/RG.2.2.20309.19688. Available at: https://github.com/Alamanos11/Land_uses_prediction
"""

import os
import numpy as np
import pandas as pd
import rasterio
import logging

logger = logging.getLogger(__name__)

def read_raster(file_path):
    """
    Read a raster file and return its data and transform.
    
    Parameters:
    - file_path (str): Path to the raster file.
    
    Returns:
    - tuple: Numpy array of raster data and raster transform.
    """
    with rasterio.open(file_path) as src:
        return src.read(1), src.transform

def calculate_change_matrix(before, after):
    """
    Calculate the change matrix between two rasters.
    
    Parameters:
    - before (numpy array): Raster data before.
    - after (numpy array): Raster data after.
    
    Returns:
    - pandas DataFrame: Change matrix.
    """
    unique_before = np.unique(before)
    unique_after = np.unique(after)
    all_classes = np.unique(np.concatenate((unique_before, unique_after)))

    change_matrix = pd.DataFrame(0, index=all_classes, columns=all_classes)

    for i in range(before.shape[0]):
        for j in range(before.shape[1]):
            from_class = before[i, j]
            to_class = after[i, j]
            change_matrix.loc[from_class, to_class] += 1

    return change_matrix

def calculate_transition_probabilities(change_matrix):
    """
    Calculate transition probabilities from the change matrix.
    
    Parameters:
    - change_matrix (pandas DataFrame): Change matrix.
    
    Returns:
    - pandas DataFrame: Transition probability matrix.
    """
    transition_matrix = change_matrix.div(change_matrix.sum(axis=1), axis=0)
    return transition_matrix

def apply_transition(land_use, transition_probs):
    """
    Apply transition probabilities to a land use map.
    
    Parameters:
    - land_use (numpy array): Current land use map.
    - transition_probs (pandas DataFrame): Transition probability matrix.
    
    Returns:
    - numpy array: New land use map.
    """
    new_land_use = np.copy(land_use)
    rows, cols = land_use.shape
    for row in range(rows):
        for col in range(cols):
            current_category = int(land_use[row, col])
            if 1 <= current_category <= 5:
                transition_probs_normalized = transition_probs.loc[current_category].values
                new_category = np.argmax(np.random.multinomial(1, transition_probs_normalized))
                new_land_use[row, col] = new_category
    return new_land_use

def main(before_file, after_file, current_land_use_map, output_raster_path):
    """
    Main function to calculate change and transition matrices and apply transitions.
    
    Parameters:
    - before_file (str): Path to the before raster file.
    - after_file (str): Path to the after raster file.
    - current_land_use_map (str): Path to the current land use map raster file.
    - output_raster_path (str): Path to save the output raster file.
    
    Returns:
    - tuple: Change matrix and transition matrix.
    """
    # read input rasters
    before, _ = read_raster(before_file)
    after, _ = read_raster(after_file)

    # calculate change matrix
    change_matrix = calculate_change_matrix(before, after)
    logger.info("Change matrix:")
    logger.info(change_matrix)

    # calculate transition matrix
    transition_matrix = calculate_transition_probabilities(change_matrix)
    logger.info("Transition matrix:")
    logger.info(transition_matrix)

    # current land use map
    with rasterio.open(current_land_use_map) as src:
        current_land_use_data = src.read(1)
        transform = src.transform
        crs = src.crs

    # apply transition to current land use map
    predicted_land_use = apply_transition(current_land_use_data, transition_matrix)

    # save predicted land use map as .tif
    rows, cols = predicted_land_use.shape
    with rasterio.open(output_raster_path, 'w', driver='GTiff', width=cols, height=rows, count=1, dtype=rasterio.int32, crs=crs, transform=transform) as dst:
        dst.write(predicted_land_use, 1)

    logger.info(f"Prediction completed. The result is saved to: {output_raster_path}")

    return change_matrix, transition_matrix

