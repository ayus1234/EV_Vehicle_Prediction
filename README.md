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

## 📈 Model and Evaluation

- **Model**: `RandomForestRegressor`
- **Tuning**: `RandomizedSearchCV` with cross-validation

---

## 📌 Results

### ✅ Evaluation Metrics

| Metric   | Value   |
|----------|---------|
| MAE      | 132.76  |
| RMSE     | 200.45  |
| R² Score | 0.89    |

These results indicate a strong model performance with relatively low error compared to the scale of EV counts.

---

### 🧾 Sample Output

```text
        Date         County       Predicted_EV_Total
0   2025-08-01      Kings             14527
1   2025-09-01      Kings             14862
2   2025-10-01      Kings             15230
3   2025-11-01      Kings             15575
4   2025-12-01      Kings             15940
5   2026-01-01      Kings             16294
```

---

## 💾 Model Persistence

  * Model saved to: `forecasting_ev_model.pkl`.
  * Successfully reloaded and tested.

  To avoid retraining:
   
   ```bash
   from joblib import load
   model = load('forecasting_ev_model.pkl')
   ```

---

## 🔍 Single Sample Test

  ```text
  Actual EVs:    1025.00  
  Predicted EVs: 998.23
  ```
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

## 📊 Visualizations

  ### 🔹 EV Breakdown vs Total Vehicles
  
  <img width="722" height="526" alt="Screenshot 2025-07-27 020318" src="https://github.com/user-attachments/assets/a4b65ff5-56e4-4271-9635-74b2cf665eed" />

  Stacked column chart comparing:
  - BEV (Battery Electric Vehicles)
  - PHEV (Plug-in Hybrids)
  - EV (total)
  - Non-EVs  
  
  It highlights the share of EVs in the overall vehicle population.

---

  ### 🔹 Actual vs Predicted EV Count
  
  <img width="834" height="464" alt="Screenshot 2025-07-27 020344" src="https://github.com/user-attachments/assets/b1670a25-18d8-4b3d-9a95-411761c70fc5" />
  
  * Line plot showing the RandomForest model's predictions vs actual EV counts across sample indices.  
  * Close overlap indicates strong model accuracy.

---

  ### 🔹 Feature Importance
  
  <img width="744" height="422" alt="Screenshot 2025-07-27 020430" src="https://github.com/user-attachments/assets/81d49953-9d79-4387-ac7b-6fe5188603d6" />

  Bar plot displaying the importance scores of engineered features like:
  - Lag values
  - Rolling averages
  - Percent changes
    
  Used to assess the model's key drivers of prediction.

  ---

  ### 🔹 County-Level Forecast: Kings County (Monthly)
  
  <img width="814" height="394" alt="Screenshot 2025-07-27 020449" src="https://github.com/user-attachments/assets/f032cca3-b224-4873-b34d-88133f4b4824" />

  Historical vs 36-month forecast for **Kings County** showing monthly EV growth trends.

  ---

  ### 🔹 Cumulative EV Forecast: Kings County
  
 <img width="811" height="403" alt="Screenshot 2025-07-27 020505" src="https://github.com/user-attachments/assets/53d7e30e-0e77-4177-b1de-4b49d286cc28" />
  
  Chart showing cumulative EV adoption over time, including projected growth for the next 3 years.

  ---

  ### 🔹 Top 5 Counties Forecast
  
  <img width="826" height="401" alt="Screenshot 2025-07-27 020524" src="https://github.com/user-attachments/assets/04411510-47cf-4528-9c1a-e438d80c080b" />
  
  Visualization of historical and projected cumulative EV growth for the top 5 counties:
  - Fairfax
  - Honolulu
  - Los Angeles
  - Orange
  - Santa Clara

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
   This project is open-source and licensed under the [MIT License](./LICENSE).

---
