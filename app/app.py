import streamlit as st
import numpy as np
import joblib
from datetime import datetime


cat = joblib.load(r"C:\Users\merug\Desktop\Taxi Trip distance prediction\taxi trip distance prediction\model files\cat_model.pkl")
xgb = joblib.load(r"C:\Users\merug\Desktop\Taxi Trip distance prediction\taxi trip distance prediction\model files\xgb_model.pkl")
lgb = joblib.load(r"C:\Users\merug\Desktop\Taxi Trip distance prediction\taxi trip distance prediction\model files\lgb_model.pkl")

st.title("🚖 Taxi Trip Distance Prediction App")

pickup_lat = st.number_input("Pickup Latitude")
pickup_lon = st.number_input("Pickup Longitude")
dropoff_lat = st.number_input("Dropoff Latitude")
dropoff_lon = st.number_input("Dropoff Longitude")

passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=2)

pickup_hour = st.slider("Pickup Hour (0-23)", 0, 23, 12)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

traffic = st.selectbox("Traffic Level", ["low", "medium", "high"])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def is_rush_hour(hour):
    return 1 if hour in [8, 9, 17, 18, 19] else 0

def avg_speed(dist):
    return dist / 30 


if st.button("Predict Trip Distance"):

    traffic_map = {"low": 1, "medium": 2, "high": 3}
    traffic_val = traffic_map[traffic]

    dist = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    rush_hour = is_rush_hour(pickup_hour)

    speed = avg_speed(dist)

    features = np.array([[
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
        passenger_count,
        traffic_val,
        dist,
        rush_hour,
        day_of_week,
        speed
    ]])

    
    cat_pred = cat.predict(features)
    xgb_pred = xgb.predict(features)
    lgb_pred = lgb.predict(features)

    final_pred = (
        0.6 * cat_pred +
        0.4 * xgb_pred +
        0.0 * lgb_pred
    )

    st.success(f"🚀 Predicted Distance: {final_pred[0]:.2f} miles")
