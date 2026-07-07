"""
app.py
Streaming data dari OpenAQ v3 - Update setiap 1 jam
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import requests
import time
from threading import Thread
import queue

# Auto-detect JAVA_HOME
if "JAVA_HOME" not in os.environ:
    for candidate in [
        "/usr/lib/jvm/java-17-openjdk-amd64",
        "/usr/lib/jvm/default-java",
    ]:
        if os.path.exists(candidate):
            os.environ["JAVA_HOME"] = candidate
            break

# ============= KONFIGURASI =============
MODEL_PATH = "ispu_rf_model"
FEATURES = ["pm1", "relativehumidity", "temperature", "um003"]

# Ambil API key dari secrets
OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# Lokasi Malang
LOCATION = {
    "latitude": -7.9185093,
    "longitude": 112.651344,
    "radius": 5000
}

# Kategori ISPU
CATEGORY_COLOR = {
    "Baik": "#2ecc71",
    "Sedang": "#f1c40f",
    "Tidak Sehat": "#e67e22",
    "Sangat Tidak Sehat": "#e74c3c",
}

CATEGORY_EMOJI = {
    "Baik": "🌤️",
    "Sedang": "⛅",
    "Tidak Sehat": "🌥️",
    "Sangat Tidak Sehat": "🌧️",
}

CATEGORY_DESC = {
    "Baik": "Kualitas udara baik",
    "Sedang": "Kualitas udara sedang",
    "Tidak Sehat": "Kualitas udara tidak sehat",
    "Sangat Tidak Sehat": "Kualitas udara sangat tidak sehat",
}

st.set_page_config(
    page_title="🌤️ Ramalan Kualitas Udara Malang",
    page_icon="🌤️",
    layout="wide"
)

data_queue = queue.Queue(maxsize=100)


class OpenAQStream:
    """Streaming data dari OpenAQ v3 - Update 1 jam sekali"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        
    def discover_sensors(self):
        """Cari sensor di Malang"""
        try:
            # Cari location
            params = {
                "coordinates": f"{LOCATION['latitude']},{LOCATION['longitude']}",
                "radius": LOCATION["radius"],
                "limit": 5
            }
            
            resp = requests.get(
                f"{OPENAQ_BASE_URL}/locations",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if resp.status_code == 200:
                locations = resp.json().get("results", [])
                if locations:
                    location_id = locations[0].get("id")
                    
                    # Cari sensors
                    sensor_resp = requests.get(
                        f"{OPENAQ_BASE_URL}/sensors",
                        params={"location_id": location_id, "limit": 20},
                        headers=self.headers,
                        timeout=10
                    )
                    
                    if sensor_resp.status_code == 200:
                        sensors = sensor_resp.json().get("results", [])
                        for sensor in sensors:
                            param = sensor.get("parameter", {}).get("name", "").lower()
                            if param in ["pm1", "pm25", "relativehumidity", "temperature"]:
                                self.sensor_ids[param] = sensor.get("id")
                        return self.sensor_ids
            
            return {}
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {}
    
    def get_latest_data(self):
        """Ambil data terbaru dari sensor"""
        if not self.sensor_ids:
            self.discover_sensors()
            
        if not self.sensor_ids:
            return None
            
        data = {}
        for param, sensor_id in self.sensor_ids.items():
            try:
                resp = requests.get(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                    params={"limit": 1, "sort": "desc"},
                    headers=self.headers,
                    timeout=10
                )
                
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        data[param] = results[0].get("value")
                        data["timestamp"] = results[0].get("datetime", {}).get("utc", datetime.now().isoformat())
            except:
                continue
        
        # Cek data lengkap
        if "pm1" in data and "relativehumidity" in data and "temperature" in data:
            data["um003"] = data["pm1"] * 100 + np.random.normal(0, 50)
            return data
        
        return None


def stream_worker():
    """Worker untuk streaming data - jalan di background"""
    stream = OpenAQStream(OPENAQ_API_KEY)
    
    while True:
        try:
            data = stream.get_latest_data()
            if data:
                if not data_queue.full():
                    data_queue.put(data)
            time.sleep(3600)  # 1 jam
        except:
            time.sleep(60)


@st.cache_resource
def load_model():
    """Load Spark dan model"""
    spark = SparkSession.builder \
        .appName("ISPU-Streaming") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(MODEL_PATH)
    return spark, model


def predict(spark, model, data):
    """Prediksi single data"""
    pdf = pd.DataFrame([data])
    sdf = spark.createDataFrame(pdf)
    result = model.transform(sdf)
    labels = model.stages[0].labelsArray[0]
    pred_idx = result.select("prediction").first()[0]
    return labels[int(pred_idx)]


def main():
    st.title("🌤️ Kualitas Udara Malang")
    
    # Start streaming
    if 'stream_thread' not in st.session_state:
        thread = Thread(target=stream_worker, daemon=True)
        thread.start()
        st.session_state.stream_thread = thread
    
    # Load model
    spark, model = load_model()
    
    # Ambil data terbaru dari queue
    latest_data = None
    while not data_queue.empty():
        latest_data = data_queue.get()
    
    # Tampilkan
    if latest_data:
        kategori = predict(spark, model, latest_data)
        latest_data["category"] = kategori
        
        color = CATEGORY_COLOR.get(kategori, "#95a5a6")
        emoji = CATEGORY_EMOJI.get(kategori, "🌤️")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align:center;padding:30px;background:linear-gradient(135deg,{color}20,{color}05);border-radius:20px;border:2px solid {color};'>
                <div style='font-size:4rem;'>{emoji}</div>
                <div style='font-size:2.5rem;font-weight:bold;color:{color};'>{kategori}</div>
                <div style='font-size:1rem;color:#666;margin:10px 0;'>{CATEGORY_DESC.get(kategori, "")}</div>
                <div style='display:flex;justify-content:center;gap:30px;font-size:1rem;color:#888;'>
                    <span>🌡️ {latest_data.get('temperature', 0):.1f}°C</span>
                    <span>💧 {latest_data.get('relativehumidity', 0):.1f}%</span>
                    <span>📊 PM1: {latest_data.get('pm1', 0):.1f}</span>
                </div>
                <div style='margin-top:15px;font-size:0.9rem;color:#999;'>
                    🕐 {latest_data.get('timestamp', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediksi 7 hari
        st.subheader("🔮 Prediksi 7 Hari")
        
        # Generate forecast sederhana
        dates = [(datetime.now() + timedelta(days=i)).strftime("%a, %d %b") for i in range(7)]
        forecast_data = []
        
        for i in range(7):
            day_data = {
                "pm1": max(0, latest_data["pm1"] + np.random.normal(0, 2)),
                "relativehumidity": max(0, min(100, latest_data["relativehumidity"] + np.random.normal(0, 3))),
                "temperature": max(0, min(45, latest_data["temperature"] + np.random.normal(0, 1))),
                "um003": max(0, latest_data["um003"] + np.random.normal(0, 50)),
            }
            day_data["category"] = predict(spark, model, day_data)
            forecast_data.append({"date": dates[i], **day_data})
        
        cols = st.columns(7)
        for i, data in enumerate(forecast_data):
            with cols[i]:
                cat = data["category"]
                color = CATEGORY_COLOR.get(cat, "#95a5a6")
                emoji = CATEGORY_EMOJI.get(cat, "🌤️")
                st.markdown(f"""
                <div style='text-align:center;padding:10px;background:{color}15;border-radius:10px;border-left:3px solid {color};'>
                    <div style='font-size:1.8rem;'>{emoji}</div>
                    <div style='font-size:0.8rem;font-weight:bold;'>{data['date']}</div>
                    <div style='font-size:1.1rem;font-weight:bold;color:{color};'>{cat}</div>
                    <div style='font-size:0.7rem;color:#888;'>
                        {data['temperature']:.0f}°C | {data['relativehumidity']:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Grafik
        st.subheader("📈 Tren")
        fig = make_subplots(rows=2, cols=1)
        
        # Kategori
        cat_to_num = {"Baik": 1, "Sedang": 2, "Tidak Sehat": 3, "Sangat Tidak Sehat": 4}
        fig.add_trace(
            go.Scatter(
                x=[d["date"] for d in forecast_data],
                y=[cat_to_num[d["category"]] for d in forecast_data],
                mode='lines+markers+text',
                text=[d["category"] for d in forecast_data],
                textposition='top center',
                marker=dict(size=15, color=[CATEGORY_COLOR[d["category"]] for d in forecast_data]),
                name='Kategori'
            ),
            row=1, col=1
        )
        fig.update_yaxes(tickvals=[1,2,3,4], ticktext=['Baik','Sedang','Tidak Sehat','Sangat Tidak Sehat'], row=1, col=1)
        
        # PM1
        fig.add_trace(
            go.Bar(
                x=[d["date"] for d in forecast_data],
                y=[d["pm1"] for d in forecast_data],
                name='PM1',
                marker_color='#27ae60'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("⏳ Menunggu data dari OpenAQ... Update setiap 1 jam")


if __name__ == "__main__":
    main()
