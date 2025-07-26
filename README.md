# EV_Vehicle_Prediction
<!-- # 🚗 Electric Vehicle Population Forecasting by County -->

This project provides a comprehensive pipeline to forecast the total number of Electric Vehicles (EVs) in each county over time using historical data. The workflow includes data preprocessing, feature engineering, model training with hyperparameter tuning, and model evaluation.

---

## 📊 Project Overview

Using the `Electric_Vehicle_Population_By_County.csv` dataset, this project:

- Cleans and processes time-series EV data at the county level.
- Engineers features to capture temporal trends and growth patterns.
- Trains a `RandomForestRegressor` to predict EV totals.
- Evaluates model performance and visualizes feature importance.
- Forecasts EV adoption for any given county over the next 3 years.
- Saves and tests the trained model with `joblib`.

---

## 📊 Project Highlights

### ✅ Data Preprocessing

- Handled missing values in `County` and `State`
- Converted vehicle count columns to numeric
- Capped outliers in `Percent Electric Vehicles`
- Converted `Date` column to datetime

### 🔍 Exploratory Data Analysis

- Identified top/bottom counties by EV adoption
- Visualized stacked vehicle distributions
- Calculated total counts for BEVs, PHEVs, EVs, and Non-EVs

### 🛠️ Feature Engineering

- Lag features (1 to 3 months)
- Rolling 3-month EV average
- Percent change over 1 and 3 months
- Cumulative EVs per county
- 6-month rolling slope for growth trend

### 🧠 Model Training

- **Model**: `RandomForestRegressor`
- **Hyperparameter Tuning**: `RandomizedSearchCV` (30 iterations, 3-fold CV)
- **Best Parameters**:
  
  ```python
  {
      'n_estimators': 200,
      'min_samples_split': 4,
      'min_samples_leaf': 1,
      'max_features': None,
      'max_depth': 15
  }

---

## 📁 Dataset

The dataset should include columns such as:

- `Date`
- `County`, `State`
- `Electric Vehicle (EV) Total`
- `Battery Electric Vehicles (BEVs)`, `Plug-In Hybrid Electric Vehicles (PHEVs)`
- `Non-Electric Vehicle Total`, `Total Vehicles`
- `Percent Electric Vehicles`

---

## ⚙️ Features Engineered

- Lag features (1 to 3 months)
- Rolling means
- Percent changes
- Cumulative totals
- Linear growth slope over rolling windows
- Encoded county identifiers
- Time since county EV data began

---

## 📈 Model and Evaluation

- **Model**: `RandomForestRegressor`
- **Tuning**: `RandomizedSearchCV` with cross-validation

### ✅ Evaluation Metrics

| Metric   | Value  |
|----------|--------|
| MAE      | 0.007  |
| RMSE     | 0.06   |
| R² Score | 1.00   |

- Extremely high R² score confirms the model performs very well on the test data.

---

## 🔮 Forecasting

### 📍 County-Level Forecasting

Forecasts next **36 months** of EV growth for a selected county (e.g., *Kings*).

Includes:
- Monthly predicted EV counts
- Cumulative EV count trendline
- Comparison between historical and forecasted values

### 🌍 Top-5 Counties Forecast

- Forecasted next 3 years for the top 5 counties (based on cumulative EV adoption)
- Combined historical and future trendlines
- Visual comparison of growth rates across counties

---

## 📌 Sample Output

---

## 📊 Feature Importance Plot
   
   The script will generate a bar chart showing which features were most impactful in the model's predictions.

## 💾 Model Persistence
   
   To avoid retraining:
   
   ```bash
   from joblib import load
   model = load('forecasting_ev_model.pkl')
   ```

---

## 📁 Files

| File                                | Description                               |
|-------------------------------------|-------------------------------------------|
| `Electric_Vehicle_Population_By_County.csv` | Raw EV dataset                    |
| `preprocessed_ev_data.csv`         | Cleaned and feature-engineered data       |
| `forecasting_ev_model.pkl`         | Trained RandomForest regression model     |
| `ev_forecasting.ipynb`             | Full pipeline notebook with forecasting   |
| `README.md`                        | Project overview and instructions         |

---

## 🛠 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/ev-forecasting.git
   cd ev-forecasting
   ```
   
2. **Install Dependencies**

   Make sure Python ≥ 3.7 is installed, then install required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

3. **Run the Script**

   Ensure the dataset Electric_Vehicle_Population_By_County.csv is in the working directory and run:

   ```bash
   jupyter notebook ev_forecasting.ipynb
   ```

---

## 📈 Model and Evaluation

   * Model: RandomForestRegressor

   * Tuning: RandomizedSearchCV with cross-validation

   * Metrics:

      * MAE (Mean Absolute Error)

      * RMSE (Root Mean Square Error)

      * R² Score

   The trained model is saved as `forecasting_ev_model.pkl`.

---

## 🧠 Future Improvements

   * Integrate demographic data like population, income, or GDP by county.
   * Try gradient boosting models like XGBoost or LightGBM.
   * Explore deep learning with LSTM for sequential forecasting.
   * Deploy the model with a Flask/Dash API or Streamlit dashboard.

---

## 📃 License
   This project is open-source and available under the MIT License.

---
