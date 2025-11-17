# Machine-Learning-Predictive-Analytics-System

House Price Prediction
This notebook demonstrates a complete machine learning workflow for predicting house prices using various regression models. The process covers data loading, preprocessing, model training, evaluation, and saving the best-performing model for future use.

Table of Contents
Project Overview
Data
Preprocessing
Model Training and Evaluation
Model Persistence
Making Predictions
Project Overview
This project aims to predict house sale prices based on a dataset of various house features. Several regression algorithms are implemented and compared to identify the most accurate model for the task.

Data
The dataset used is train.csv, which contains detailed information about residential homes. Key steps performed on the data:

Loading: The data is loaded into a pandas DataFrame.
Initial Inspection: df.head() and df.info() are used to inspect the data's structure and identify missing values.
Column Dropping: Columns with a high number of missing values or those deemed irrelevant (Alley, PoolQC, Fence, MiscFeature) are dropped.
Preprocessing
Before training the models, the data undergoes several preprocessing steps:

Target and Feature Separation: SalePrice is separated as the target variable (y), and the remaining columns form the feature set (X).
One-Hot Encoding: Categorical features in X are converted into numerical format using one-hot encoding (pd.get_dummies).
Missing Value Imputation: Missing numerical values in X are imputed with the mean of their respective columns.
Feature Scaling: Numerical features in X are scaled using StandardScaler (X_scaler).
Target Scaling: The target variable y (SalePrice) is also scaled using a separate StandardScaler (y_scaler). This is crucial for models sensitive to scale and for inverse transforming predictions later.
Train-Test Split: The processed data is split into training and testing sets (X_train, X_test, y_train, y_test).
Model Training and Evaluation
Four different regression models are trained and evaluated:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Each model is trained on X_train and y_train, and predictions are made on X_test. The models are evaluated using:

Root Mean Squared Error (RMSE): Measures the average magnitude of the errors.
R-squared (R2) Score: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
The model with the highest R2 score is identified as the best_model.

Model Persistence
The best_model, X_scaler, and y_scaler are saved to disk using joblib:

house_price_model.pkl: The trained best model.
X_scaler.pkl: The scaler fitted on the feature set.
y_scaler.pkl: The scaler fitted on the target variable.
These files allow for the model and scalers to be loaded and used later without retraining.

Making Predictions
To make a prediction for a new house:

Select a sample house from X_test.
Use the best_model to predict its SalePrice. The prediction will be in a scaled format.
Apply the y_scaler.inverse_transform() method to convert the scaled prediction back to the original SalePrice scale.
This workflow ensures robust house price prediction and provides all necessary artifacts for deployment.
