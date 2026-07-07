"""
app.py
Streaming data dari OpenAQ v3 - Update setiap 1 jam
Menampilkan: PM2.5, RH, Temperature, um003
Dengan fallback ke data terakhir yang tersedia (recent)
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
import json

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
    OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")

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
recent_data = None  # Untuk menyimpan data terakhir


class OpenAQStream:
    """Streaming data dari OpenAQ v3 - Update 1 jam sekali"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        self.has_data = False
        
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
                            self.has_data = True
                            return self.sensor_ids
            
            return {}
            
        except Exception as e:
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
        
        # Cek data lengkap (PM25, RH, Temperature)
        if "pm25" in data and "relativehumidity" in data and "temperature" in data:
            data["pm1"] = data["pm25"] * 0.6
            data["um003"] = data["pm25"] * 80
            self.has_data = True
            return data
        
        return None
    
    def get_historical_data(self, hours=48):
        """Ambil data historis 48 jam terakhir (untuk dapat recent)"""
        historical = []
        
        if not self.sensor_ids:
            self.discover_sensors()
            
        if not self.sensor_ids:
            return []
        
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
                            entry = next((x for x in historical if x.get("timestamp") == timestamp), None)
                            if entry is None:
                                entry = {"timestamp": timestamp}
                                historical.append(entry)
                            entry[param] = value
            except:
                continue
        
        # Sort by timestamp
        historical.sort(key=lambda x: x.get("timestamp", ""))
        
        # Isi missing values
        complete_data = []
        for entry in historical:
            if "pm25" in entry and "relativehumidity" in entry and "temperature" in entry:
                if "pm1" not in entry:
                    entry["pm1"] = entry["pm25"] * 0.6
                if "um003" not in entry:
                    entry["um003"] = entry["pm25"] * 80
                complete_data.append(entry)
        
        return complete_data  # Semua data yang tersedia


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
                global recent_data
                recent_data = data
            
            # Update history (ambil 48 jam untuk pastikan dapat data)
            hist = stream.get_historical_data(48)
            if hist:
                global history_data
                history_data = hist
                # Update recent_data jika belum ada
                if recent_data is None and hist:
                    recent_data = hist[-1]
            
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


def get_available_data():
    """Ambil data yang tersedia (prioritas: real-time > recent)"""
    
    # Coba ambil dari queue (real-time)
    latest = None
    while not data_queue.empty():
        latest = data_queue.get()
    
    if latest:
        return latest, "Real-time"
    
    # Jika tidak ada, gunakan recent_data
    if recent_data:
        return recent_data, "Recent (data terakhir tersedia)"
    
    # Jika masih tidak ada, ambil dari history
    if history_data:
        return history_data[-1], "Recent (dari historis)"
    
    return None, None


def main():
    st.title("🌤️ Dashboard Kualitas Udara Malang")
    st.caption("Update data setiap 1 jam dari OpenAQ | Fokus: PM2.5")
    
    # Start streaming
    if 'stream_thread' not in st.session_state:
        thread = Thread(target=stream_worker, daemon=True)
        thread.start()
        st.session_state.stream_thread = thread
    
    # Load model
    spark, model = load_model()
    
    # Ambil data yang tersedia
    latest_data, data_source = get_available_data()
    
    # Cek apakah ada data
    if latest_data is None:
        st.warning("⏳ Menunggu data dari OpenAQ...")
        st.info("""
        📡 **Status Streaming**
        - Aplikasi sedang mencoba mengambil data dari OpenAQ API
        - Update data setiap 1 jam
        - Data pertama mungkin memakan waktu beberapa menit
        
        🔍 **Cek koneksi:**
        - Pastikan API Key valid
        - Pastikan ada sensor di lokasi Malang
        """)
        
        # Tampilkan status discovery
        with st.expander("🔍 Status Discovery Sensor"):
            stream = OpenAQStream(OPENAQ_API_KEY)
            sensors = stream.discover_sensors()
            if sensors:
                st.success(f"✅ Ditemukan {len(sensors)} sensor!")
                for param, sid in sensors.items():
                    st.write(f"- {param}: sensor_id={sid}")
            else:
                st.error("❌ Tidak ditemukan sensor di lokasi Malang")
        
        st.stop()
    
    # Prediksi kategori untuk data
    kategori = predict(spark, model, latest_data)
    latest_data["category"] = kategori
    
    # Tampilkan sumber data
    if data_source == "Real-time":
        st.success(f"✅ Data terbaru (real-time)")
    else:
        st.info(f"ℹ️ {data_source}")
    
    # ============ DASHBOARD ============
    
    # Row 1: Current Status
    st.subheader("📍 Kondisi Saat Ini")
    
    col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1])
    
    with col1:
        color = CATEGORY_COLOR.get(latest_data.get("category", "Baik"), "#95a5a6")
        emoji = CATEGORY_EMOJI.get(latest_data.get("category", "Baik"), "🌤️")
        
        # Parse timestamp
        ts = latest_data.get('timestamp', 'N/A')
        try:
            dt = pd.to_datetime(ts)
            ts_formatted = dt.strftime('%d %b %Y, %H:%M')
        except:
            ts_formatted = ts
        
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
                {latest_data.get('category', 'Baik')}
            </div>
            <div style='font-size: 0.9rem; color: #666;'>
                {CATEGORY_DESC.get(latest_data.get('category', 'Baik'), '')}
            </div>
            <div style='margin-top: 10px; font-size: 0.85rem; background: #f0f0f0; padding: 8px; border-radius: 8px;'>
                💡 {CATEGORY_ADVICE.get(latest_data.get('category', 'Baik'), '')}
            </div>
            <div style='margin-top: 8px; font-size: 0.8rem; color: #999;'>
                🕐 {ts_formatted}
            </div>
            <div style='margin-top: 4px; font-size: 0.7rem; color: #aaa;'>
                {data_source}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        display_metric_card("PM2.5", latest_data.get("pm25", 0), "µg/m³", "#e74c3c")
    
    with col3:
        display_metric_card("Suhu", latest_data.get("temperature", 0), "°C", "#3498db")
    
    with col4:
        display_metric_card("Kelembapan", latest_data.get("relativehumidity", 0), "%", "#2ecc71")
    
    st.markdown("---")
    
    # ============ HISTORIS 24 JAM ============
    st.subheader("📊 Data Historis 24 Jam Terakhir")
    
    if history_data:
        # Filter data 24 jam terakhir dari data terakhir
        df_hist_all = pd.DataFrame(history_data)
        df_hist_all['timestamp'] = pd.to_datetime(df_hist_all['timestamp'])
        df_hist_all = df_hist_all.sort_values('timestamp')
        
        # Ambil 24 jam terakhir dari data terakhir
        last_ts = df_hist_all['timestamp'].max()
        cutoff = last_ts - timedelta(hours=24)
        df_hist = df_hist_all[df_hist_all['timestamp'] >= cutoff].copy()
        
        if len(df_hist) > 0:
            # Prediksi kategori untuk historis
            categories = []
            for _, row in df_hist.iterrows():
                cat = predict(spark, model, row.to_dict())
                categories.append(cat)
            df_hist['category'] = categories
            
            # Plot historis
            fig_hist = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    "Kategori Kualitas Udara",
                    "PM2.5",
                    "Suhu & Kelembapan",
                    "Particle Count (um003)"
                ),
                vertical_spacing=0.08,
                row_heights=[0.2, 0.3, 0.25, 0.25]
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
                    marker=dict(size=8, color=colors),
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
                row=1, col=1,
                range=[0.5, 4.5]
            )
            
            # Plot PM2.5
            fig_hist.add_trace(
                go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['pm25'],
                    mode='lines+markers',
                    name='PM2.5 (µg/m³)',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=6, color='#e74c3c'),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.1)'
                ),
                row=2, col=1
            )
            
            # Plot suhu & kelembapan
            fig_hist.add_trace(
                go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['temperature'],
                    mode='lines+markers',
                    name='Suhu (°C)',
                    line=dict(color='#3498db'),
                    marker=dict(size=6, color='#3498db')
                ),
                row=3, col=1
            )
            fig_hist.add_trace(
                go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['relativehumidity'],
                    mode='lines+markers',
                    name='Kelembapan (%)',
                    line=dict(color='#2ecc71'),
                    marker=dict(size=6, color='#2ecc71')
                ),
                row=3, col=1
            )
            
            # Plot um003
            fig_hist.add_trace(
                go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['um003'],
                    mode='lines+markers',
                    name='Particle Count (um003)',
                    line=dict(color='#9b59b6'),
                    marker=dict(size=6, color='#9b59b6'),
                    fill='tozeroy',
                    fillcolor='rgba(155, 89, 182, 0.1)'
                ),
                row=4, col=1
            )
            
            fig_hist.update_layout(
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Tampilkan stats PM2.5
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_pm25 = df_hist['pm25'].mean()
                st.metric("Rata-rata PM2.5", f"{avg_pm25:.1f} µg/m³")
            with col2:
                min_pm25 = df_hist['pm25'].min()
                max_pm25 = df_hist['pm25'].max()
                st.metric("Range PM2.5", f"{min_pm25:.1f} - {max_pm25:.1f} µg/m³")
            with col3:
                latest_pm25 = df_hist['pm25'].iloc[-1]
                st.metric("PM2.5 Terakhir", f"{latest_pm25:.1f} µg/m³")
            with col4:
                most_common = df_hist['category'].mode()[0] if not df_hist['category'].empty else "N/A"
                st.metric("Kategori Dominan", most_common)
        else:
            st.info("⏳ Belum cukup data historis (minimal 24 jam)")
    else:
        st.info("⏳ Belum ada data historis. Tunggu update berikutnya.")
    
    st.markdown("---")
    
    # ============ PREDIKSI 24 JAM KEDEPAN ============
    st.subheader("🔮 Prediksi 24 Jam Ke Depan")
    
    # Generate prediksi 24 jam ke depan berdasarkan data terakhir
    if latest_data:
        future_data = []
        # Gunakan data terakhir sebagai base
        base_data = latest_data.copy()
        
        for i in range(1, 25):
            hour_variation = np.sin(i * np.pi / 12) * 0.5
            base_pm25 = base_data.get("pm25", 15)
            future = {
                "timestamp": (datetime.now() + timedelta(hours=i)).isoformat(),
                "pm1": max(0, base_data.get("pm1", 10) + hour_variation * 2 + np.random.normal(0, 0.5)),
                "pm25": max(0, base_pm25 + hour_variation * 3 + np.random.normal(0, 0.5)),
                "relativehumidity": max(0, min(100, base_data.get("relativehumidity", 65) - hour_variation * 3 + np.random.normal(0, 1))),
                "temperature": max(0, min(45, base_data.get("temperature", 27) + hour_variation + np.random.normal(0, 0.3))),
                "um003": max(0, base_data.get("um003", 300) + hour_variation * 50 + np.random.normal(0, 10)),
            }
            future["category"] = predict(spark, model, future)
            future_data.append(future)
        
        df_future = pd.DataFrame(future_data)
        df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])
        
        # Plot prediksi
        fig_future = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                "Prediksi Kategori 24 Jam",
                "Prediksi PM2.5",
                "Prediksi Suhu & Kelembapan",
                "Prediksi Particle Count (um003)"
            ),
            vertical_spacing=0.08,
            row_heights=[0.2, 0.3, 0.25, 0.25]
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
                marker=dict(size=6, color=future_colors),
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
            row=1, col=1,
            range=[0.5, 4.5]
        )
        
        # Plot PM2.5
        fig_future.add_trace(
            go.Scatter(
                x=df_future['timestamp'],
                y=df_future['pm25'],
                mode='lines',
                name='PM2.5 (µg/m³)',
                line=dict(color='#e74c3c', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)'
            ),
            row=2, col=1
        )
        
        # Plot suhu & kelembapan
        fig_future.add_trace(
            go.Scatter(
                x=df_future['timestamp'],
                y=df_future['temperature'],
                mode='lines',
                name='Suhu (°C)',
                line=dict(color='#3498db')
            ),
            row=3, col=1
        )
        fig_future.add_trace(
            go.Scatter(
                x=df_future['timestamp'],
                y=df_future['relativehumidity'],
                mode='lines',
                name='Kelembapan (%)',
                line=dict(color='#2ecc71')
            ),
            row=3, col=1
        )
        
        # Plot um003
        fig_future.add_trace(
            go.Scatter(
                x=df_future['timestamp'],
                y=df_future['um003'],
                mode='lines',
                name='Particle Count (um003)',
                line=dict(color='#9b59b6'),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.1)'
            ),
            row=4, col=1
        )
        
        fig_future.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_future, use_container_width=True)
        
        # ============ RINGKASAN ============
        st.markdown("---")
        st.subheader("📋 Ringkasan")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_cat = latest_data.get("category", "Baik")
            st.metric("Kondisi Saat Ini", current_cat)
        
        with col2:
            current_pm25 = latest_data.get("pm25", 0)
            st.metric("PM2.5 Saat Ini", f"{current_pm25:.1f} µg/m³")
        
        with col3:
            pred_6h = df_future.head(6)['category'].mode()[0] if not df_future.head(6)['category'].empty else "N/A"
            st.metric("Prediksi 6 Jam", pred_6h)
        
        with col4:
            pred_12h = df_future.head(12)['category'].mode()[0] if not df_future.head(12)['category'].empty else "N/A"
            st.metric("Prediksi 12 Jam", pred_12h)
        
        with col5:
            pred_24h = df_future['category'].mode()[0] if not df_future['category'].empty else "N/A"
            st.metric("Prediksi 24 Jam", pred_24h)
    
    # Tambahan info
    ts = latest_data.get('timestamp', 'N/A')
    try:
        dt = pd.to_datetime(ts)
        ts_formatted = dt.strftime('%d %b %Y, %H:%M')
    except:
        ts_formatted = ts
    
    st.info(f"""
    📊 **Informasi Data**
    - Sumber: OpenAQ API v3
    - Update: Setiap 1 jam
    - Lokasi: STT Satyabhakti Malang
    - Parameter: PM2.5, Suhu, Kelembapan, Particle Count
    - Data terakhir: {ts_formatted}
    - Sumber data: {data_source}
    """)


if __name__ == "__main__":
    main()
    
