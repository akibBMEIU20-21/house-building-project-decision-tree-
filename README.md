# Housing Price Prediction

This project explores the prediction of house prices using machine learning algorithms. It leverages Python libraries like `pandas`, `numpy`, `seaborn`, `matplotlib`, and various machine learning models to preprocess, analyze, and predict housing prices based on features in a dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [Future Work](#future-work)

## Project Overview
The primary goal of this project is to predict house prices based on various features, such as area, bedrooms, bathrooms, furnishing status, and more. The project involves data preprocessing, feature engineering, exploratory data analysis, and model training using different regression algorithms, including Linear Regression, Gradient Boosting, Random Forest, and Voting Regressor.

## Dataset
The dataset used in this project is stored as a CSV file, `Housing.csv`. Ensure this file is in the correct path (`/content/Housing.csv` for Colab or adjust accordingly) before running the notebook.

## Dependencies
The following Python packages are required to run the code:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Data Preprocessing
The data preprocessing pipeline includes:
1. **Label Encoding** for categorical features.
2. **Ordinal Encoding** for features with a clear order (e.g., `furnishingstatus`).
3. **Scaling** using `StandardScaler` to normalize numerical data.

## Exploratory Data Analysis
Exploratory analysis is performed to:
- Identify missing values and duplicates.
- Visualize distributions of `price` and other numerical features.
- Check correlations between features using heatmaps.

## Model Training
Multiple models are trained to predict house prices:
- **Linear Regression**
- **Decision Tree Regressor**
- **Support Vector Regressor (SVR)**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Voting Regressor** (an ensemble model combining above models)

## Model Evaluation
The models are evaluated using **R-squared** scores from cross-validation. The Voting Regressor is fine-tuned by adjusting weights, providing an optimal ensemble model performance.

## How to Run
1. Clone the repository and navigate to the project directory.
2. Ensure the `Housing.csv` dataset is located at `/content/Housing.csv`.
3. Install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook in Jupyter or Colab.

## Results
Each model's R-squared score is displayed, with the Voting Regressor generally providing the best results after fine-tuning weights.

## Future Work
Future improvements include:
- Exploring more features.
- Hyperparameter tuning for models.
- Trying additional ensemble methods.
