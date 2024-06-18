# MCCA Land Use Prediction Model for the Peruvian Amazon

## Introduction

This project is part of the CAS in Advanced Machine Learning. Our objective is to simulate and predict land use changes in the Peruvian Amazon using a combination of Markov Chain and Cellular Automata models. Understanding and predicting land use changes is critical for formulating effective conservation strategies, sustainable development policies, and climate mitigation efforts.

## Project Overview

The Peruvian Amazon, a vital component of the world's largest tropical rainforest, faces significant land use and land cover changes driven by deforestation, agricultural expansion, and infrastructure development. This project aims to create a robust predictive model integrating Markov Chain and Cellular Automata methods to simulate land use dynamics accurately.

## Methodology

The project employs a Markov Chain model to simulate land use change. This approach leverages the probabilistic transition capabilities of Markov Chains.

1. **Markov Chain Transition Probability Prediction**: This step involves calculating change and transition matrices based on historical land use data.
2. **Transformation/Prediction**: The transition matrices are used to predict future land use distributions.
3. **Validation**: The predicted maps are validated using metrics such as Accuracy, Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Kappa coefficient (κ), and Confusion Matrix statistics.

## Data

### Land Use Data

Land cover data was obtained from the ESA Climate Change Initiative (CCI), providing a consistent global annual land cover dataset at 300m spatial resolution from 1992 to 2015. Each pixel value corresponds to the label of a land cover class defined based on the UN Land Cover Classification System (LCCS).

### Study Area

The study area chosen for this project is the Peruvian Amazon, one of the most biodiverse regions on Earth. It acts as a significant carbon sink, absorbing large amounts of carbon dioxide from the atmosphere, and is vital for maintaining ecological balance and resilience.

## Data Preprocessing

To reduce computational workload and increase the quantity of class-specific land use changes, the land use classes were aggregated. The analysis focused on 5-year time slices (1992 - 1996, 1997 - 2001, 2002 - 2006, 2007 - 2011) using GIS software (ArcGIS Pro 3.2.0).

## Results

The results section includes validation statistics, accuracy metrics, and the confusion matrix for different years (1997, 2002, 2007, 2012). The accuracy remained high, and the predictive performance improved over time.

## References

A list of references and further reading is provided in the project documentation.

## Authors

- Tobias Liechti (tobiliechti@gmail.com)
- Jan Göpel (Jan.Goepel@wyssacademy.org)

## Usage

Due to the large size of the data files (tifs and shape files), they are not included in this repository. On request, access can be provided via Jan Göpel's Google Drive. Please contact Jan Göpel (Jan.Goepel@wyssacademy.org) to request access to the data files.
