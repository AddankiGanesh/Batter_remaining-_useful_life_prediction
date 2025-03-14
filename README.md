# Batter_remaining-_useful_life_prediction
Overview
This project focuses on predicting the Remaining Useful Life (RUL) of batteries using Machine Learning (ML) techniques. The goal is to estimate how many charge-discharge cycles a battery can undergo before it reaches its end of life. This is crucial for optimizing battery usage, maintenance, and replacement strategies in various applications such as electric vehicles, portable electronics, and renewable energy storage systems.

Dataset
The dataset used in this project is Battery_RUL.csv, which contains various features related to battery cycles, such as:

Cycle_Index
Discharge Time (s)
Decrement 3.6-3.4V (s)
Max. Voltage Dischar. (V)
Min. Voltage Charg. (V)
Time at 4.15V (s)
Time constant current (s)
Charging time (s)
RUL (Remaining Useful Life)
The dataset consists of 15,064 entries with 9 features.

Models
Six different machine learning models were trained and evaluated for this project:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)
K-Nearest Neighbors (KNN) Regressor
XGBoost Regressor

All models achieved an accuracy of around 95% in predicting the RUL of the batteries.

Methodology
Data Preprocessing:
Loaded and inspected the dataset.
Handled missing values and duplicates.
Split the data into features (X) and target (y).
Normalized/Standardized the data where necessary.

Exploratory Data Analysis (EDA):

Visualized the distribution of features using histograms and box plots.
Identified and handled outliers.
Analyzed the correlation between features and the target variable.
Model Training:
Split the data into training and testing sets.
Trained each model on the training set.
Tuned hyperparameters using cross-validation.

Model Evaluation:

Evaluated the models on the test set using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
Compared the performance of all models.
Results:
All models achieved an accuracy of around 95%.
The best-performing model was selected based on the evaluation metrics.

Requirements
To run the code in this project, you need the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

You can install these libraries using pip:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Run the Jupyter Notebook:
jupyter notebook Battery_RUL_Prediction.ipynb
