
# 🚖 Taxi Trip Distance Prediction

> A complete end-to-end machine learning system that predicts taxi trip distance (in miles) using pickup/dropoff coordinates and ride conditions — built for the **MachineHack Week 46 Hackathon**.

---

## 📌 Problem Statement

Predict the **trip distance in miles** for a taxi ride given pre-trip information such as pickup/dropoff coordinates, passenger count, and traffic conditions. This is a **regression problem** evaluated using **RMSE (Root Mean Square Error)**.

---

## 🏆 Results

| Model | RMSE |
|-------|------|
| CatBoost | **0.7768** Best individual model |
| XGBoost | **0.7817**|
| LightGBM | **0.7918** |
| **Ensemble (Weighted Avg)** | **0.0.7742** ✅ |

> Ensemble weights: **CatBoost (0.6) + XGBoost (0.4) + LightGBM (0.0)**

---

## 📁 Project Structure

```
taxi-trip-distance-prediction/
│
├──Py Code/
|    |-----notebook.ipynb              
|
├── app(steamlit code)/
|    |------ app.py                      
│
├── model files/
│   |--- catboost_model.pkl      
│   |--- xgb_model.pkl           
│   |--- lgb_model.pkl          
│
├── raw data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
│

```

---

## 🔧 Approach

### 1. Data Preprocessing
- Loaded and merged train/test datasets
- Handled missing values
- Parsed datetime columns → extracted `hour`, `day`, `month`, `is_weekend`
- Computed `trip_duration` and other derived features

### 2. Feature Engineering
- **Haversine Distance**: Calculated real-world distance between pickup and dropoff coordinates using the Haversine formula
- Final selected features (7 out of 29):
  - `pickup_latitude`, `pickup_longitude`
  - `dropoff_latitude`, `dropoff_longitude`
  - `passenger_count`
  - `traffic_level`
  - `haversine_distance`
  - `rush_hour`
  - `day_of_week`
  - `avg_speed`

### 3. Model Building
Trained three gradient boosting models:
- **CatBoost** — best individual RMSE
- **XGBoost**
- **LightGBM**

Split strategy: **Train / Validation / Test**

### 4. Ensemble Learning
Combined model predictions using **weighted averaging**:

```python
ensemble_pred = 0.6 * cat_pred + 0.4 * xgb_pred + 0.0 * lgb_pred
```

Ensemble RMSE: **0.7742**

### 5.Train and Validation Score

-Train RMSE: 0.6956514647529121
-Validation RMSE: 0.7768563387059666

### 6. Deployment
- Models saved with `joblib`
- Interactive **Streamlit** web app built for real-time predictions

---

## 🚀 Getting Started


### Run the Streamlit App

```bash
streamlit run app.py
```

The app accepts the following inputs and returns the predicted trip distance:
- Pickup latitude & longitude
- Dropoff latitude & longitude
- Passenger count
- Traffic level

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
catboost
xgboost
lightgbm
joblib
streamlit
```

---

## 📊 Haversine Distance Formula

To capture the real-world geographic distance between pickup and dropoff points:

```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
```

---

## 🎯 Key Takeaways

- Haversine distance was the **strongest predictor** of trip distance
- CatBoost outperformed XGBoost and LightGBM individually
- Ensemble learning provided a consistent improvement over any single model
- Limiting to 7 user-friendly features simplified deployment without sacrificing accuracy

---

## 🏅 Hackathon

This project was submitted to the **MachineHack Week 46 Hackathon — Taxi Trip Distance Prediction Challenge**.

- **Evaluation Metric**: RMSE
- **Level**: Intermediate
- **Domain**: Regression / Transportation
- **link:https://machinehack.com/hackathons/taxi_trip_distance_prediction_challenge

---

## 👤 Author

> Built with ❤️ for MachineHack Week 46


