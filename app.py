"""
app.py
Streaming data dari OpenAQ v3 - Update setiap 1 jam
Menggunakan datetime_from dan datetime_to untuk ambil data historis
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
history_data = []
recent_data = None


class OpenAQStream:
    """Streaming data dari OpenAQ v3 - Update 1 jam sekali"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        
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
                        return self.sensor_ids
            return {}
        except:
            return {}
    
    def get_data_by_time_range(self, sensor_id, datetime_from, datetime_to, limit=100):
        """Ambil data dalam rentang waktu tertentu"""
        try:
            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": limit,
                "sort": "desc"
            }
            
            resp = requests.get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return results
            return []
        except:
            return []
    
    def get_latest_data(self):
        """Ambil data terbaru (1 jam terakhir)"""
        if not self.sensor_ids:
            self.discover_sensors()
        if not self.sensor_ids:
            return None
        
        # Rentang waktu: 1 jam terakhir
        datetime_to = datetime.now().isoformat() + 'Z'
        datetime_from = (datetime.now() - timedelta(hours=1)).isoformat() + 'Z'
        
        data = {}
        for param, sensor_id in self.sensor_ids.items():
            results = self.get_data_by_time_range(sensor_id, datetime_from, datetime_to, limit=1)
            if results:
                data[param] = results[0].get("value")
                data["timestamp"] = results[0].get("datetime", {}).get("utc", datetime.now().isoformat())
        
        if "pm25" in data and "relativehumidity" in data and "temperature" in data:
            data["pm1"] = data["pm25"] * 0.6
            data["um003"] = data["pm25"] * 80
            return data
        return None
    
    def get_historical_data(self, hours=24):
        """Ambil data historis 24 jam terakhir dengan rentang waktu"""
        if not self.sensor_ids:
            self.discover_sensors()
        if not self.sensor_ids:
            return []
        
        # Rentang waktu: 24 jam terakhir
        datetime_to = datetime.now().isoformat() + 'Z'
        datetime_from = (datetime.now() - timedelta(hours=hours)).isoformat() + 'Z'
        
        historical = []
        for param, sensor_id in self.sensor_ids.items():
            results = self.get_data_by_time_range(sensor_id, datetime_from, datetime_to, limit=hours)
            for r in results:
                timestamp = r.get("datetime", {}).get("utc")
                value = r.get("value")
                if timestamp and value is not None:
                    entry = next((x for x in historical if x.get("timestamp") == timestamp), None)
                    if entry is None:
                        entry = {"timestamp": timestamp}
                        historical.append(entry)
                    entry[param] = value
        
        # Sort by timestamp
        historical.sort(key=lambda x: x.get("timestamp", ""))
        
        # Isi missing values
        complete = []
        for entry in historical:
            if "pm25" in entry and "relativehumidity" in entry and "temperature" in entry:
                entry["pm1"] = entry["pm25"] * 0.6
                entry["um003"] = entry["pm25"] * 80
                complete.append(entry)
        
        return complete
    
    def get_recent_data(self):
        """Ambil data terbaru dari 24 jam terakhir (ambil yang paling baru)"""
        historical = self.get_historical_data(24)
        if historical:
            return historical[-1]
        return None


def stream_worker():
    """Worker streaming - jalan di background"""
    stream = OpenAQStream(OPENAQ_API_KEY)
    
    while True:
        try:
            # Ambil data terbaru (1 jam terakhir)
            data = stream.get_latest_data()
            if data:
                if not data_queue.full():
                    data_queue.put(data)
                global recent_data
                recent_data = data
            
            # Ambil data historis (24 jam terakhir)
            hist = stream.get_historical_data(24)
            if hist:
                global history_data
                history_data = hist
                if recent_data is None and hist:
                    recent_data = hist[-1]
            
            time.sleep(3600)  # 1 jam
        except Exception as e:
            time.sleep(60)


@st.cache_resource
def load_model():
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
    pdf = pd.DataFrame([data])
    sdf = spark.createDataFrame(pdf)
    result = model.transform(sdf)
    labels = model.stages[0].labelsArray[0]
    pred_idx = result.select("prediction").first()[0]
    return labels[int(pred_idx)]


def display_metric_card(title, value, unit, color):
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


def get_recent_data():
    """Ambil data terbaru dari queue atau recent_data"""
    latest = None
    while not data_queue.empty():
        latest = data_queue.get()
    if latest:
        return latest
    if recent_data:
        return recent_data
    if history_data:
        return history_data[-1]
    return None


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
    
    # Ambil recent data
    latest_data = get_recent_data()
    
    # Jika belum ada data, fetch sekali
    if latest_data is None:
        with st.spinner("Mengambil data dari OpenAQ..."):
            stream = OpenAQStream(OPENAQ_API_KEY)
            stream.discover_sensors()
            latest_data = stream.get_latest_data()
            if latest_data:
                recent_data = latest_data
                history_data = stream.get_historical_data(24)
    
    # Jika masih tidak ada data, tampilkan pesan
    if latest_data is None:
        st.warning("⚠️ Belum ada data dari OpenAQ. Tunggu update berikutnya.")
        st.stop()
    
    # Prediksi kategori
    kategori = predict(spark, model, latest_data)
    latest_data["category"] = kategori
    
    # ============ RECENT DATA ============
    st.subheader("📍 Data Terbaru")
    
    col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1])
    
    with col1:
        color = CATEGORY_COLOR.get(latest_data.get("category", "Baik"), "#95a5a6")
        emoji = CATEGORY_EMOJI.get(latest_data.get("category", "Baik"), "🌤️")
        
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
    st.subheader("📊 Historis 24 Jam Terakhir")
    
    if history_data:
        df_hist_all = pd.DataFrame(history_data)
        df_hist_all['timestamp'] = pd.to_datetime(df_hist_all['timestamp'])
        df_hist_all = df_hist_all.sort_values('timestamp')
        
        # Filter 24 jam terakhir
        last_ts = df_hist_all['timestamp'].max()
        cutoff = last_ts - timedelta(hours=24)
        df_hist = df_hist_all[df_hist_all['timestamp'] >= cutoff].copy()
        
        if len(df_hist) > 0:
            categories = []
            for _, row in df_hist.iterrows():
                cat = predict(spark, model, row.to_dict())
                categories.append(cat)
            df_hist['category'] = categories
            
            fig_hist = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Kategori", "PM2.5", "Suhu & Kelembapan"),
                vertical_spacing=0.12,
                row_heights=[0.25, 0.35, 0.4]
            )
            
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
                row=1, col=1,
                range=[0.5, 4.5]
            )
            
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
            
            fig_hist.update_layout(height=600, showlegend=True, hovermode='x unified')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata PM2.5", f"{df_hist['pm25'].mean():.1f} µg/m³")
            with col2:
                st.metric("Min - Max PM2.5", f"{df_hist['pm25'].min():.1f} - {df_hist['pm25'].max():.1f} µg/m³")
            with col3:
                most_common = df_hist['category'].mode()[0] if not df_hist['category'].empty else "N/A"
                st.metric("Kategori Dominan", most_common)
        else:
            st.info("⏳ Belum cukup data historis")
    else:
        st.info("⏳ Menunggu data historis")
    
    st.markdown("---")
    
    # ============ PREDIKSI 24 JAM ============
    st.subheader("🔮 Prediksi 24 Jam Ke Depan")
    
    # Generate prediksi berdasarkan data terakhir
    future_data = []
    for i in range(1, 25):
        hour_variation = np.sin(i * np.pi / 12) * 0.5
        future = {
            "timestamp": (datetime.now() + timedelta(hours=i)).isoformat(),
            "pm1": max(0, latest_data.get("pm1", 10) + hour_variation * 2 + np.random.normal(0, 0.3)),
            "pm25": max(0, latest_data.get("pm25", 15) + hour_variation * 3 + np.random.normal(0, 0.3)),
            "relativehumidity": max(0, min(100, latest_data.get("relativehumidity", 65) - hour_variation * 3 + np.random.normal(0, 0.5))),
            "temperature": max(0, min(45, latest_data.get("temperature", 27) + hour_variation + np.random.normal(0, 0.2))),
            "um003": max(0, latest_data.get("um003", 300) + hour_variation * 50 + np.random.normal(0, 5)),
        }
        future["category"] = predict(spark, model, future)
        future_data.append(future)
    
    df_future = pd.DataFrame(future_data)
    df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])
    
    fig_future = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Prediksi Kategori", "Prediksi PM2.5", "Prediksi Suhu & Kelembapan"),
        vertical_spacing=0.12,
        row_heights=[0.25, 0.35, 0.4]
    )
    
    cat_to_num = {"Baik": 1, "Sedang": 2, "Tidak Sehat": 3, "Sangat Tidak Sehat": 4}
    future_cat_nums = [cat_to_num.get(c, 0) for c in df_future['category']]
    future_colors = [CATEGORY_COLOR.get(c, "#95a5a6") for c in df_future['category']]
    
    fig_future.add_trace(
        go.Scatter(
            x=df_future['timestamp'],
            y=future_cat_nums,
            mode='lines+markers',
            marker=dict(size=10, color=future_colors),
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
    
    fig_future.update_layout(height=600, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig_future, use_container_width=True)
    
    # ============ RINGKASAN ============
    st.markdown("---")
    st.subheader("📋 Ringkasan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Kondisi Saat Ini", latest_data.get("category", "Baik"))
    with col2:
        st.metric("PM2.5 Saat Ini", f"{latest_data.get('pm25', 0):.1f} µg/m³")
    with col3:
        pred_12h = df_future.head(12)['category'].mode()[0] if not df_future.head(12)['category'].empty else "N/A"
        st.metric("Prediksi 12 Jam", pred_12h)
    with col4:
        pred_24h = df_future['category'].mode()[0] if not df_future['category'].empty else "N/A"
        st.metric("Prediksi 24 Jam", pred_24h)
    
    ts = latest_data.get('timestamp', 'N/A')
    try:
        dt = pd.to_datetime(ts)
        ts_formatted = dt.strftime('%d %b %Y, %H:%M')
    except:
        ts_formatted = ts
    
    st.info(f"📊 **Data terakhir:** {ts_formatted} | Update berikutnya: +1 jam")


if __name__ == "__main__":
    main()
