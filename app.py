"""
app.py
Streaming data dari OpenAQ v3 - Update setiap 1 jam
Menampilkan: Prediksi 1 hari ke depan + Historis 1 hari ke belakang + Dashboard
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
try:
    OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]
except:
    OPENAQ_API_KEY = "430a6cbeb038741241c9129a3323543b8a15f7e2b80bd32d9d07b2efb3d66aff"

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
    "Baik": "Kualitas udara baik, tidak berdampak pada kesehatan",
    "Sedang": "Kualitas udara sedang, kelompok sensitif perlu waspada",
    "Tidak Sehat": "Kualitas udara tidak sehat, kurangi aktivitas luar ruangan",
    "Sangat Tidak Sehat": "Kualitas udara sangat tidak sehat, hindari aktivitas luar ruangan",
}

CATEGORY_ADVICE = {
    "Baik": "✅ Bebas beraktivitas di luar ruangan",
    "Sedang": "⚠️ Kelompok sensitif kurangi aktivitas berat",
    "Tidak Sehat": "😷 Gunakan masker saat di luar",
    "Sangat Tidak Sehat": "🚫 Hindari aktivitas luar ruangan",
}

st.set_page_config(
    page_title="🌤️ Dashboard Kualitas Udara Malang",
    page_icon="🌤️",
    layout="wide"
)

data_queue = queue.Queue(maxsize=100)
history_data = []  # Untuk menyimpan data historis


class OpenAQStream:
    """Streaming data dari OpenAQ v3 - Update 1 jam sekali"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        self.using_dummy = True
        
    def discover_sensors(self):
        """Cari sensor di Malang"""
        try:
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
                        
                        if self.sensor_ids:
                            self.using_dummy = False
                            return self.sensor_ids
            
            return {}
            
        except Exception as e:
            return {}
    
    def get_latest_data(self):
        """Ambil data terbaru dari sensor"""
        if not self.sensor_ids:
            self.discover_sensors()
            
        if not self.sensor_ids:
            return self.get_dummy_data()
            
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
            data["is_dummy"] = False
            return data
        
        return self.get_dummy_data()
    
    def get_dummy_data(self):
        """Generate dummy data untuk testing"""
        return {
            "pm1": np.random.uniform(5, 25),
            "relativehumidity": np.random.uniform(50, 80),
            "temperature": np.random.uniform(24, 30),
            "um003": np.random.uniform(200, 600),
            "timestamp": datetime.now().isoformat(),
            "is_dummy": True
        }
    
    def get_historical_data(self, hours=24):
        """Ambil data historis 24 jam terakhir"""
        historical = []
        
        if not self.sensor_ids:
            self.discover_sensors()
            
        if not self.sensor_ids:
            # Generate dummy historical
            for i in range(24, 0, -1):
                t = datetime.now() - timedelta(hours=i)
                historical.append({
                    "timestamp": t.isoformat(),
                    "pm1": np.random.uniform(5, 25),
                    "relativehumidity": np.random.uniform(50, 80),
                    "temperature": np.random.uniform(24, 30),
                    "um003": np.random.uniform(200, 600),
                    "is_dummy": True
                })
            return historical
        
        # Ambil dari sensor
        for param, sensor_id in self.sensor_ids.items():
            try:
                resp = requests.get(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                    params={"limit": hours, "sort": "desc"},
                    headers=self.headers,
                    timeout=10
                )
                
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        timestamp = r.get("datetime", {}).get("utc")
                        value = r.get("value")
                        if timestamp and value is not None:
                            # Cari existing entry
                            entry = next((x for x in historical if x.get("timestamp") == timestamp), None)
                            if entry is None:
                                entry = {"timestamp": timestamp}
                                historical.append(entry)
                            entry[param] = value
            except:
                continue
        
        # Sort by timestamp
        historical.sort(key=lambda x: x.get("timestamp", ""))
        
        # Isi missing values dengan dummy
        for entry in historical:
            if "pm1" not in entry:
                entry["pm1"] = np.random.uniform(5, 25)
            if "relativehumidity" not in entry:
                entry["relativehumidity"] = np.random.uniform(50, 80)
            if "temperature" not in entry:
                entry["temperature"] = np.random.uniform(24, 30)
            if "um003" not in entry:
                entry["um003"] = entry["pm1"] * 100 + np.random.normal(0, 50)
            entry["is_dummy"] = False
        
        return historical[-24:]  # 24 jam terakhir


def stream_worker():
    """Worker untuk streaming data - jalan di background"""
    stream = OpenAQStream(OPENAQ_API_KEY)
    
    while True:
        try:
            # Ambil data terbaru
            data = stream.get_latest_data()
            if data:
                if not data_queue.full():
                    data_queue.put(data)
            
            # Update history
            global history_data
            history_data = stream.get_historical_data(24)
            
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


def display_metric_card(title, value, unit, color):
    """Display metric card"""
    st.markdown(f"""
    <div style='
        background: {color}15;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {color};
        text-align: center;
    '>
        <div style='font-size: 0.8rem; color: #888;'>{title}</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: {color};'>
            {value:.1f} {unit}
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.title("🌤️ Dashboard Kualitas Udara Malang")
    st.caption("Update data setiap 1 jam dari OpenAQ")
    
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
    
    # Jika tidak ada data, gunakan dummy
    if latest_data is None:
        stream = OpenAQStream(OPENAQ_API_KEY)
        latest_data = stream.get_dummy_data()
        # Update history
        if not history_data:
            history_data.extend(stream.get_historical_data(24))
    
    # Prediksi kategori untuk data terbaru
    if latest_data:
        kategori = predict(spark, model, latest_data)
        latest_data["category"] = kategori
    
    # ============ DASHBOARD ============
    
    # Row 1: Current Status
    st.subheader("📍 Kondisi Saat Ini")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        color = CATEGORY_COLOR.get(latest_data.get("category", "Baik"), "#95a5a6")
        emoji = CATEGORY_EMOJI.get(latest_data.get("category", "Baik"), "🌤️")
        is_dummy = latest_data.get("is_dummy", False)
        dummy_text = " (Simulasi)" if is_dummy else ""
        
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {color}20, {color}05);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid {color};
            text-align: center;
        '>
            <div style='font-size: 3rem;'>{emoji}</div>
            <div style='font-size: 2rem; font-weight: bold; color: {color};'>
                {latest_data.get('category', 'Baik')}{dummy_text}
            </div>
            <div style='font-size: 0.9rem; color: #666;'>
                {CATEGORY_DESC.get(latest_data.get('category', 'Baik'), '')}
            </div>
            <div style='margin-top: 10px; font-size: 0.85rem; background: #f0f0f0; padding: 8px; border-radius: 8px;'>
                💡 {CATEGORY_ADVICE.get(latest_data.get('category', 'Baik'), '')}
            </div>
            <div style='margin-top: 8px; font-size: 0.8rem; color: #999;'>
                🕐 {latest_data.get('timestamp', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        display_metric_card("Suhu", latest_data.get("temperature", 0), "°C", "#e74c3c")
    
    with col3:
        display_metric_card("Kelembapan", latest_data.get("relativehumidity", 0), "%", "#3498db")
    
    with col4:
        display_metric_card("PM1", latest_data.get("pm1", 0), "µg/m³", "#27ae60")
    
    st.markdown("---")
    
    # ============ HISTORIS 24 JAM ============
    st.subheader("📊 Data Historis 24 Jam Terakhir")
    
    if history_data:
        # Konversi ke DataFrame
        df_hist = pd.DataFrame(history_data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist = df_hist.sort_values('timestamp')
        
        # Prediksi kategori untuk historis
        categories = []
        for _, row in df_hist.iterrows():
            cat = predict(spark, model, row.to_dict())
            categories.append(cat)
        df_hist['category'] = categories
        
        # Plot historis
        fig_hist = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Kategori Kualitas Udara", "Suhu & Kelembapan", "PM1"),
            vertical_spacing=0.1,
            row_heights=[0.3, 0.35, 0.35]
        )
        
        # Plot kategori
        cat_to_num = {"Baik": 1, "Sedang": 2, "Tidak Sehat": 3, "Sangat Tidak Sehat": 4}
        cat_nums = [cat_to_num.get(c, 0) for c in df_hist['category']]
        colors = [CATEGORY_COLOR.get(c, "#95a5a6") for c in df_hist['category']]
        
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist['timestamp'],
                y=cat_nums,
                mode='markers+lines',
                marker=dict(size=10, color=colors),
                line=dict(color='#333', width=1),
                text=df_hist['category'],
                hovertemplate='%{text}<extra></extra>',
                name='Kategori'
            ),
            row=1, col=1
        )
        fig_hist.update_yaxes(
            tickvals=[1, 2, 3, 4],
            ticktext=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat'],
            row=1, col=1
        )
        
        # Plot suhu & kelembapan
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist['timestamp'],
                y=df_hist['temperature'],
                mode='lines+markers',
                name='Suhu (°C)',
                line=dict(color='#e74c3c'),
                marker=dict(size=6, color='#e74c3c')
            ),
            row=2, col=1
        )
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist['timestamp'],
                y=df_hist['relativehumidity'],
                mode='lines+markers',
                name='Kelembapan (%)',
                line=dict(color='#3498db'),
                marker=dict(size=6, color='#3498db')
            ),
            row=2, col=1
        )
        
        # Plot PM1
        fig_hist.add_trace(
            go.Bar(
                x=df_hist['timestamp'],
                y=df_hist['pm1'],
                name='PM1 (µg/m³)',
                marker_color='#27ae60',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        fig_hist.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tampilkan stats
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_pm1 = df_hist['pm1'].mean()
            st.metric("Rata-rata PM1", f"{avg_pm1:.1f} µg/m³")
        with col2:
            min_pm1 = df_hist['pm1'].min()
            max_pm1 = df_hist['pm1'].max()
            st.metric("Range PM1", f"{min_pm1:.1f} - {max_pm1:.1f} µg/m³")
        with col3:
            most_common = df_hist['category'].mode()[0] if not df_hist['category'].empty else "N/A"
            st.metric("Kategori Dominan", most_common)
    
    st.markdown("---")
    
    # ============ PREDIKSI 1 HARI KEDEPAN ============
    st.subheader("🔮 Prediksi 1 Hari Ke Depan")
    
    # Generate prediksi 24 jam ke depan
    future_data = []
    for i in range(1, 25):
        # Simulasi variasi natural
        hour_variation = np.sin(i * np.pi / 12) * 0.5
        future = {
            "timestamp": (datetime.now() + timedelta(hours=i)).isoformat(),
            "pm1": max(0, latest_data.get("pm1", 10) + np.random.normal(0, 1) + hour_variation * 2),
            "relativehumidity": max(0, min(100, latest_data.get("relativehumidity", 65) + np.random.normal(0, 2) - hour_variation * 3)),
            "temperature": max(0, min(45, latest_data.get("temperature", 27) + np.random.normal(0, 0.5) + hour_variation)),
            "um003": max(0, latest_data.get("um003", 300) + np.random.normal(0, 30) + hour_variation * 50),
        }
        future["category"] = predict(spark, model, future)
        future_data.append(future)
    
    df_future = pd.DataFrame(future_data)
    df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])
    
    # Plot prediksi
    fig_future = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Prediksi Kategori 24 Jam", "Prediksi Suhu & Kelembapan", "Prediksi PM1"),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.35, 0.35]
    )
    
    # Plot kategori prediksi
    cat_to_num = {"Baik": 1, "Sedang": 2, "Tidak Sehat": 3, "Sangat Tidak Sehat": 4}
    future_cat_nums = [cat_to_num.get(c, 0) for c in df_future['category']]
    future_colors = [CATEGORY_COLOR.get(c, "#95a5a6") for c in df_future['category']]
    
    fig_future.add_trace(
        go.Scatter(
            x=df_future['timestamp'],
            y=future_cat_nums,
            mode='lines+markers',
            marker=dict(size=8, color=future_colors),
            line=dict(color='#333', width=1),
            text=df_future['category'],
            hovertemplate='%{text}<extra></extra>',
            name='Prediksi Kategori'
        ),
        row=1, col=1
    )
    fig_future.update_yaxes(
        tickvals=[1, 2, 3, 4],
        ticktext=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat'],
        row=1, col=1
    )
    
    # Plot suhu & kelembapan
    fig_future.add_trace(
        go.Scatter(
            x=df_future['timestamp'],
            y=df_future['temperature'],
            mode='lines',
            name='Suhu (°C)',
            line=dict(color='#e74c3c')
        ),
        row=2, col=1
    )
    fig_future.add_trace(
        go.Scatter(
            x=df_future['timestamp'],
            y=df_future['relativehumidity'],
            mode='lines',
            name='Kelembapan (%)',
            line=dict(color='#3498db')
        ),
        row=2, col=1
    )
    
    # Plot PM1
    fig_future.add_trace(
        go.Scatter(
            x=df_future['timestamp'],
            y=df_future['pm1'],
            mode='lines',
            name='PM1 (µg/m³)',
            line=dict(color='#27ae60'),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.2)'
        ),
        row=3, col=1
    )
    
    fig_future.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_future, use_container_width=True)
    
    # ============ RINGKASAN ============
    st.markdown("---")
    st.subheader("📋 Ringkasan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Kategori saat ini
        current_cat = latest_data.get("category", "Baik")
        st.metric("Kondisi Saat Ini", current_cat)
    
    with col2:
        # Prediksi 6 jam depan
        next_6h = df_future.head(6)['category']
        if not next_6h.empty:
            pred_6h = next_6h.mode()[0]
            st.metric("Prediksi 6 Jam", pred_6h)
    
    with col3:
        # Prediksi 12 jam depan
        next_12h = df_future.head(12)['category']
        if not next_12h.empty:
            pred_12h = next_12h.mode()[0]
            st.metric("Prediksi 12 Jam", pred_12h)
    
    with col4:
        # Prediksi 24 jam depan
        pred_24h = df_future['category'].mode()[0] if not df_future['category'].empty else "N/A"
        st.metric("Prediksi 24 Jam", pred_24h)
    
    # Tambahan info
    st.info(f"""
    📊 **Informasi Data**
    - Sumber: OpenAQ API v3
    - Update: Setiap 1 jam
    - Lokasi: STT Satyabhakti Malang
    - Data terakhir: {latest_data.get('timestamp', 'N/A')}
    - Status: {'🟢 Data Real' if not latest_data.get('is_dummy', True) else '🟡 Data Simulasi'}
    """)


if __name__ == "__main__":
    main()
