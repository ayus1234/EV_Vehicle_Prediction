# EV_Vehicle_Prediction
<!-- # 🚗 Electric Vehicle Population Forecasting by County -->

This project provides a comprehensive pipeline to forecast the total number of Electric Vehicles (EVs) in each county over time using historical data. The workflow includes data preprocessing, feature engineering, model training with hyperparameter tuning, and model evaluation.

## 📊 Project Overview

Using the **Electric_Vehicle_Population_By_County.csv** dataset, this project:

- Cleans and processes time-series EV data at the county level.
- Engineers features to capture temporal trends and growth patterns.
- Trains a `RandomForestRegressor` to predict EV totals.
- Evaluates model performance and visualizes feature importance.
- Saves and tests the trained model with `joblib`.

## 📁 Dataset

The dataset should include columns such as:

- `Date`
- `County`, `State`
- `Electric Vehicle (EV) Total`
- `Battery Electric Vehicles (BEVs)`, `Plug-In Hybrid Electric Vehicles (PHEVs)`
- `Non-Electric Vehicle Total`, `Total Vehicles`
- `Percent Electric Vehicles`

## ⚙️ Features Engineered

- Lag features (1 to 3 months)
- Rolling means
- Percent changes
- Cumulative totals
- Linear growth slope over rolling windows
- Encoded county identifiers
- Time since county EV data began

## 🛠 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/ev-forecasting.git
   cd ev-forecasting
