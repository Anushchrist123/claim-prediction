# Customer Risk Profiling & Claim Prediction Analysis

## Overview
This Streamlit application provides a comprehensive analysis of customer risk profiling and claim prediction. It includes data exploration, model training, and a user interface to predict insurance claims based on various factors.

## Features
- Dataset preview and summary statistics
- Missing value handling and correlation heatmap
- Outlier detection using box plots
- Claim distribution analysis
- Machine learning models (Random Forest, Logistic Regression, XGBoost)
- Model evaluation metrics (Precision, Recall, F1-Score, Accuracy)
- Feature importance visualization
- Prediction interface for new customer claim probability
- Claim probability visualization for different models

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Install the required packages using the following command:


pip install -r requirements.txt



## Usage
1. Run the Streamlit app using the command:

streamlit run app.py

2. The application will launch in your web browser.
3. Upload the `dataset_balanced.csv` file in the same directory.
4. Explore the dataset, train models, and make predictions.

## File Structure