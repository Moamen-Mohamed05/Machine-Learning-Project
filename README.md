🤖 Machine Learning Desktop App

A Python-based desktop application for data preprocessing, visualization, outlier detection, and Machine Learning classification using an interactive PyQt5 graphical interface.

📌 Project Overview

This project provides a complete workflow for working with structured datasets.

Users can upload CSV or Excel files, preview the data, perform preprocessing operations, visualize numerical features, detect outliers, train Machine Learning models, evaluate their performance, and save the processed dataset.

The application is designed to combine the main steps of a Machine Learning workflow into one simple desktop application.

✨ Features
📂 Data Loading

Load CSV files.

Load Excel files (.xls, .xlsx).

Display dataset information.

Preview the uploaded data.

📊 Data Visualization

The application supports:

Line Plot

Scatter Plot

Box Plot

🔍 Outlier Detection

Outliers are detected using the IQR (Interquartile Range) method.

IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR


Detected rows are displayed in a separate Outliers tab.

🧹 Data Preprocessing

The application supports:

Missing value handling.

KNN Imputation.

Categorical encoding.

Position encoding.

Standard feature scaling.

Min-Max scaling.

Polynomial feature generation.

Normalization.

🤖 Machine Learning Models

Three classification algorithms are included:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

📈 Model Evaluation

After training a model, the application displays:

Accuracy

Confusion Matrix

💾 Data Export

Processed datasets can be exported as:

CSV

Excel

The original dataset is preserved and is not modified.

🛠️ Technologies Used

Python

PyQt5

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

📁 Project Structure
Machine-Learning-Desktop-App/
│
├── main.py
├── preprocessing.py
├── models.py
├── visualization.py
├── Scale.py
├── file_handler.py
└── README.md

📄 Main Files
main.py

The main application file responsible for:

Building the PyQt5 interface.

Loading datasets.

Displaying data.

Running preprocessing operations.

Running visualizations.

Running Machine Learning models.

Displaying results.

Saving processed data.

preprocessing.py

Handles:

Missing values.

KNN imputation.

Seasonal zero handling.

Feature encoding.

models.py

Contains:

Logistic Regression.

Decision Tree.

KNN.

Train/test splitting.

Prediction.

Accuracy calculation.

Confusion matrix generation.

visualization.py

Contains:

Line Plot.

Scatter Plot.

Box Plot.

IQR-based outlier detection.

Scale.py

Provides feature scaling functionality using:

StandardScaler.

MinMaxScaler.

PolynomialFeatures.

Normalizer.

file_handler.py

Provides helper functions for:

Loading CSV and Excel files.

Displaying data previews.

🔄 Application Workflow
        Upload Dataset
              ↓
        Preview Data
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Visualization Preprocessing ML Models
    ↓         ↓         ↓
  Plots    Missing Values  Logistic Regression
 Outliers    Encoding      Decision Tree
             Scaling       KNN
                ↓             ↓
          Processed Data   Evaluation
                ↓             ↓
             Save Data   Accuracy + Confusion Matrix

📦 Installation

Install the required dependencies:

pip install pandas numpy PyQt5 scikit-learn matplotlib seaborn openpyxl

▶️ Running the Project

Run the main application:

python main.py


After launching the application:

Select a CSV or Excel dataset.

Preview the uploaded data.

Choose a visualization or preprocessing operation.

Select a target column when training a Machine Learning model.

View the model results.

Save the processed dataset if needed.

🎯 Project Objectives

The main objectives of the project are to demonstrate:

Data loading and exploration.

Missing value handling.

Feature encoding.

Feature scaling.

Data visualization.

Outlier detection.

Machine Learning classification.

Model evaluation.

Processed data export.

📊 Machine Learning Workflow
Raw Dataset
    ↓
Data Cleaning
    ↓
Missing Value Handling
    ↓
Encoding
    ↓
Feature Scaling
    ↓
Train / Test Split
    ↓
Machine Learning Model
    ↓
Prediction
    ↓
Accuracy + Confusion Matrix

🎓 Purpose

This project is intended as an educational Machine Learning application that demonstrates how different data analysis, preprocessing, visualization, and classification techniques can be integrated into a single desktop application.

📜 License

This project is intended for educational and academic purposes.
