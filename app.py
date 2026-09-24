import pickle
import pandas as pd
import streamlit as st

import os

st.set_page_config(page_title="Ship Performance Clustering & Fleet Segmentation System", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'kmeans_ship_model.pkl')
    if not os.path.exists(model_path):
        model_path = 'models/kmeans_ship_model.pkl' if os.path.exists('models/kmeans_ship_model.pkl') else 'kmeans_ship_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

artifacts = load_model()
kmeans = artifacts['kmeans_model']
scaler = artifacts['scaler']
features = artifacts['features']
cluster_desc = artifacts['cluster_descriptions']

st.title("Ship Performance Clustering & Fleet Segmentation System")
st.write("Masukkan parameter kapal di bawah ini untuk memprediksi klaster performa.")

with st.form("ship_form"):
    col1, col2 = st.columns(2)
    with col1:
        speed = st.number_input("Kecepatan (knots)", min_value=5.0, max_value=35.0, value=17.5, step=0.1)
        efficiency = st.number_input("Efisiensi Energi (nm/kWh)", min_value=0.01, max_value=3.0, value=0.80, step=0.01)
        cargo = st.number_input("Berat Kargo (tons)", min_value=10.0, max_value=5000.0, value=1000.0, step=50.0)
        cost = st.number_input("Biaya Operasional (USD)", min_value=1000.0, max_value=2000000.0, value=250000.0, step=5000.0)
    with col2:
        power = st.number_input("Daya Mesin (kW)", min_value=100.0, max_value=5000.0, value=1750.0, step=25.0)
        distance = st.number_input("Jarak Tempuh (nm)", min_value=10.0, max_value=5000.0, value=1000.0, step=50.0)
        load_pct = st.number_input("Rata-rata Muatan (%)", min_value=10.0, max_value=100.0, value=75.0, step=1.0)
        revenue = st.number_input("Pendapatan Pelayaran (USD)", min_value=1000.0, max_value=3000000.0, value=650000.0, step=5000.0)

    submitted = st.form_submit_button("Prediksi Klaster", type="primary", use_container_width=True)

if submitted:
    profit = revenue - cost
    margin = (profit / revenue * 100) if revenue > 0 else 0
    cost_per_nm = (cost / distance) if distance > 0 else 0
    cargo_utilization = cargo * (load_pct / 100.0)
    power_per_speed = (power / speed) if speed > 0 else 0

    input_df = pd.DataFrame([{
        'Speed_Over_Ground_knots': speed,
        'Engine_Power_kW': power,
        'Efficiency_nm_per_kWh': efficiency,
        'Distance_Traveled_nm': distance,
        'Cargo_Weight_tons': cargo,
        'Average_Load_Percentage': load_pct,
        'Operational_Cost_USD': cost,
        'Revenue_per_Voyage_USD': revenue,
        'Profit_USD': profit,
        'Profit_Margin_Pct': margin,
        'Cost_per_nm': cost_per_nm,
        'Cargo_Utilization': cargo_utilization,
        'Power_per_Speed': power_per_speed
    }])

    input_scaled = scaler.transform(input_df[features])
    cluster_id = int(kmeans.predict(input_scaled)[0])
    info = cluster_desc.get(cluster_id, {"title": f"Cluster {cluster_id}", "summary": "", "status": "info"})

    st.divider()
    st.subheader("Hasil Segmentasi")

    # Klaster di bagian atas
    if info["status"] == "success":
        st.success(f"**{info['title']}**\n\n{info['summary']}")
    elif info["status"] == "warning":
        st.warning(f"**{info['title']}**\n\n{info['summary']}")
    else:
        st.info(f"**{info['title']}**\n\n{info['summary']}")

    # Ringkasan Metrik di bawahnya
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Biaya Operasional", f"${cost:,.0f}")
    col_m2.metric("Pendapatan", f"${revenue:,.0f}")
    col_m3.metric("Profit", f"${profit:,.0f}")
    col_m4.metric("Margin", f"{margin:.1f}%")
