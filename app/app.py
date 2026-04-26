import streamlit as st
import numpy as np
import joblib

# ---------------------------
# LOAD MODELS
# ---------------------------
cat = joblib.load(r"C:/Users/merug/Desktop/Taxi Trip distance prediction/model files/cat_model.pkl")
xgb = joblib.load(r"C:/Users/merug/Desktop/Taxi Trip distance prediction/model files/xgb_model.pkl")
lgb = joblib.load(r"C:/Users/merug/Desktop/Taxi Trip distance prediction/model files/lgb_model.pkl")

st.title("🚖 Taxi Trip Distance Prediction")

# ---------------------------
# INPUTS (7 FEATURES)
# ---------------------------
pickup_lat = st.number_input("Pickup Latitude", value=40.75)
pickup_lon = st.number_input("Pickup Longitude", value=-73.99)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.78)
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.95)

passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=2)

traffic = st.selectbox("Traffic Level", ["low", "medium", "high"])

# ---------------------------
# HAVERSINE FUNCTION
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Trip Distance"):

    # Map traffic
    traffic_map = {"low": 1, "medium": 2, "high": 3}
    traffic_val = traffic_map[traffic]

    # Calculate distance feature
    dist = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    # Create feature array (ORDER MUST MATCH TRAINING)
    features = np.array([[pickup_lat, pickup_lon,
                          dropoff_lat, dropoff_lon,
                          passenger_count,
                          traffic_val,
                          dist]])

    # Model predictions
    cat_pred = cat.predict(features)
    xgb_pred = xgb.predict(features)
    lgb_pred = lgb.predict(features)

    # Ensemble
    final_pred = (
        0.6 * cat_pred +
        0.2 * xgb_pred +
        0.2 * lgb_pred
    )

    st.success(f"🚀 Predicted Distance: {final_pred[0]:.2f} miles")