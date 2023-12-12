# Gaussian Bayes Classifier for Diabetes Prediction

## Overview

This repository contains a Python script that implements a Gaussian Bayes classifier for predicting diabetes based on a dataset. The script uses the scikit-learn library for data preprocessing and model evaluation and relies on the multivariate normal distribution to model the likelihood of each class.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd diabetes-prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:

   ```bash
   python diabetes_prediction.py
   ```

   The script reads the "diabetes.xlsx" dataset, splits it into training and testing sets, and then trains a Gaussian Naive Bayes classifier. The accuracy of the model on the test set is printed.

## Dataset

The dataset used for training and testing is "diabetes.xlsx". It includes the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is "Outcome," indicating whether an individual has diabetes (1) or not (0).

## Model Details

The script calculates the prior probabilities, means, covariances, and determinants for each class (Outcome 0 and Outcome 1). It then applies the Gaussian Naive Bayes formula to make predictions on the test set and evaluates the accuracy of the model.

Feel free to use and modify the code as needed. If you find it useful, consider giving it a star!
