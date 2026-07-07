"""
app.py
Dashboard Kualitas Udara Malang - OpenAQ v3
Dengan Optimasi Streaming & Caching
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import requests
import time
from threading import Thread, Lock
import queue
import pickle
from functools import lru_cache

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

try:
    OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]
except:
    OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "430a6cbeb038741241c9129a3323543b8a15f7e2b80bd32d9d07b2efb3d66aff")

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

LOCATION = {
    "name": "STT Satyabhakti",
    "latitude": -7.9185093,
    "longitude": 112.651344,
    "radius": 5000
}

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
    page_title="Dashboard Kualitas Udara Malang",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ DATA CACHE ============
# Simpan data di session_state agar tidak reload setiap kali
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {
        'recent': None,
        'history': [],
        'last_update': None,
        'predicted_history': None,
        'predicted_future': None,
        'cache_version': 0
    }

if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = {}

if 'data_lock' not in st.session_state:
    st.session_state.data_lock = Lock()

data_queue = queue.Queue(maxsize=100)

# ============ STREAMING WORKER ============
class OpenAQStream:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        self.location_id = None
        
    def discover_sensors(self):
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

            if resp.status_code != 200:
                return {}

            locations = resp.json().get("results", [])
            if not locations:
                return {}

            self.location_id = locations[0].get("id")

            sensor_resp = requests.get(
                f"{OPENAQ_BASE_URL}/locations/{self.location_id}/sensors",
                headers=self.headers,
                timeout=10
            )

            if sensor_resp.status_code != 200:
                return {}

            sensors = sensor_resp.json().get("results", [])
            self.sensor_ids = {}
            for sensor in sensors:
                param = sensor.get("parameter", {}).get("name", "").lower()
                self.sensor_ids[param] = sensor.get("id")

            return self.sensor_ids

        except Exception as e:
            return {}
    
    def get_sensor_data(self, sensor_id, datetime_from=None, datetime_to=None, limit=50):
        results = self._try_endpoint("measurements", sensor_id, datetime_from, datetime_to, limit)
        if results:
            return results
        return self._try_endpoint("hours", sensor_id, datetime_from, datetime_to, limit)

    def _try_endpoint(self, endpoint, sensor_id, datetime_from, datetime_to, limit):
        try:
            params = {"limit": limit, "sort": "desc"}
            if datetime_from:
                params["datetime_from"] = datetime_from
            if datetime_to:
                params["datetime_to"] = datetime_to

            resp = requests.get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/{endpoint}",
                params=params,
                headers=self.headers,
                timeout=10
            )

            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
        except Exception as e:
            return []
    
    def fetch_all_data(self):
        if not self.sensor_ids:
            self.discover_sensors()
        if not self.sensor_ids:
            return None, []

        now = datetime.utcnow()
        datetime_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        datetime_from_24h = (now - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")  # Ambil lebih banyak

        per_param_rows = {}

        for param, sensor_id in self.sensor_ids.items():
            results = self.get_sensor_data(sensor_id, datetime_from_24h, datetime_to, limit=200)
            if not results:
                results = self.get_sensor_data(sensor_id, limit=200)

            rows = []
            for r in results:
                timestamp = r.get("datetime", {}).get("utc") if r.get("datetime") else None
                if not timestamp and r.get("period"):
                    timestamp = r.get("period", {}).get("datetimeFrom", {}).get("utc")
                value = r.get("value")
                if timestamp and value is not None:
                    rows.append((timestamp, value))
            per_param_rows[param] = rows

        hourly_frames = {}
        for param, rows in per_param_rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["timestamp", param])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour_bucket"] = df["timestamp"].dt.floor("h")
            hourly_frames[param] = df.groupby("hour_bucket")[param].mean()

        if not hourly_frames:
            return None, []

        combined = pd.DataFrame(hourly_frames).reset_index().rename(columns={"hour_bucket": "timestamp_dt"})

        required = [c for c in ["pm25", "relativehumidity", "temperature"] if c in combined.columns]
        if required:
            combined = combined.dropna(subset=required)

        if combined.empty:
            return None, []

        if "pm1" not in combined.columns:
            combined["pm1"] = combined.get("pm25", 0) * 0.6
        if "um003" not in combined.columns:
            combined["um003"] = combined.get("pm25", 0) * 80

        combined = combined.sort_values("timestamp_dt")
        combined["timestamp"] = combined["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        historical = combined.drop(columns=["timestamp_dt"]).to_dict("records")
        recent = historical[-1] if historical else None

        return recent, historical


def stream_worker():
    """Worker streaming di background - update data setiap jam"""
    stream = OpenAQStream(OPENAQ_API_KEY)
    
    # Initial fetch
    recent, historical = stream.fetch_all_data()
    if recent:
        data_queue.put((recent, historical, datetime.now()))
    
    while True:
        try:
            time.sleep(3600)  # 1 jam
            recent, historical = stream.fetch_all_data()
            if recent:
                # Clear old data from queue
                while not data_queue.empty():
                    try:
                        data_queue.get_nowait()
                    except:
                        break
                data_queue.put((recent, historical, datetime.now()))
        except Exception as e:
            time.sleep(60)


# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    spark = SparkSession.builder \
        .appName("ISPU-Streaming") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(MODEL_PATH)
    return spark, model


def predict_batch(spark, model, data_list):
    """Batch prediction dengan cache"""
    if not data_list:
        return []
    
    # Check cache dulu
    uncached_indices = []
    uncached_data = []
    results = [None] * len(data_list)
    
    for i, data in enumerate(data_list):
        # Buat cache key yang konsisten
        cache_key = tuple(sorted(
            (k, round(v, 4)) if isinstance(v, float) else (k, v) 
            for k, v in data.items() 
            if k not in ['timestamp', 'category']
        ))
        
        if cache_key in st.session_state.prediction_cache:
            results[i] = st.session_state.prediction_cache[cache_key]
        else:
            uncached_indices.append(i)
            uncached_data.append(data)
    
    # Batch predict uncached data
    if uncached_data:
        pdf = pd.DataFrame(uncached_data)
        sdf = spark.createDataFrame(pdf)
        result = model.transform(sdf)
        labels = model.stages[0].labelsArray[0]
        
        preds = result.select("prediction").collect()
        
        for idx, pred in zip(uncached_indices, preds):
            pred_label = labels[int(pred[0])]
            results[idx] = pred_label
            
            # Simpan di cache
            cache_key = tuple(sorted(
                (k, round(v, 4)) if isinstance(v, float) else (k, v) 
                for k, v in data_list[idx].items() 
                if k not in ['timestamp', 'category']
            ))
            st.session_state.prediction_cache[cache_key] = pred_label
    
    return results


def get_hourly_pattern(history_df, feature_cols):
    if history_df.empty:
        return None
    
    df = history_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    pattern = df.groupby("hour")[feature_cols].mean()
    
    full_index = pd.Index(range(24), name="hour")
    pattern = pattern.reindex(full_index)
    pattern = pd.concat([pattern, pattern, pattern])
    pattern = pattern.interpolate(limit_direction="both")
    pattern = pattern.iloc[24:48]
    pattern.index = range(24)
    
    return pattern


# ============ UI FUNCTIONS ============
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa !important; }
    .main { background-color: #f5f7fa !important; }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a2634;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        color: #6b7a8a;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .weather-card {
        background: white;
        border-radius: 24px;
        padding: 32px 36px;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        margin-bottom: 24px;
        border: 1px solid #eef2f6;
        transition: all 0.3s ease;
    }
    
    .weather-card:hover {
        box-shadow: 0 4px 30px rgba(0,0,0,0.08);
    }
    
    .weather-main {
        display: flex;
        align-items: center;
        gap: 40px;
        flex-wrap: wrap;
    }
    
    .weather-icon { font-size: 72px; line-height: 1; }
    
    .weather-temp {
        font-size: 56px;
        font-weight: 700;
        color: #1a2634;
        line-height: 1;
    }
    
    .weather-temp-unit {
        font-size: 24px;
        font-weight: 400;
        color: #6b7a8a;
        margin-left: 2px;
    }
    
    .weather-desc {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 4px;
    }
    
    .weather-location {
        font-size: 16px;
        color: #6b7a8a;
        margin-top: 2px;
    }
    
    .weather-details {
        display: flex;
        flex-wrap: wrap;
        gap: 24px 40px;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #eef2f6;
    }
    
    .weather-detail-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #4a5a6a;
    }
    
    .weather-detail-item .label { color: #8894a0; font-weight: 500; }
    .weather-detail-item .value { font-weight: 600; color: #1a2634; }
    
    .weather-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-left: 12px;
    }
    
    .forecast-section {
        background: white;
        border-radius: 20px;
        padding: 24px 28px;
        margin-top: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
    }
    
    .forecast-header {
        font-size: 16px;
        font-weight: 700;
        color: #1a2634;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .forecast-header .timezone {
        font-weight: 400;
        color: #8894a0;
        font-size: 13px;
    }
    
    .forecast-scroll {
        overflow-x: auto;
        padding-bottom: 8px;
        margin: 0 -8px;
    }
    
    .forecast-table {
        width: 100%;
        border-collapse: collapse;
        min-width: 600px;
    }
    
    .forecast-table td {
        padding: 8px 12px;
        text-align: center;
        font-size: 13px;
        vertical-align: middle;
        border-bottom: 1px solid #f0f4f8;
    }
    
    .forecast-table tr:last-child td { border-bottom: none; }
    
    .forecast-table .time-cell {
        color: #6b7a8a;
        font-size: 12px;
        min-width: 60px;
    }
    
    .forecast-table .icon-cell { font-size: 28px; min-width: 50px; }
    
    .forecast-table .temp-cell {
        font-weight: 700;
        color: #1a2634;
        font-size: 16px;
        min-width: 55px;
    }
    
    .forecast-table .detail-cell {
        color: #6b7a8a;
        font-size: 12px;
        min-width: 60px;
    }
    
    .date-separator {
        background: #f8fafc;
        font-weight: 700;
        color: #1a2634;
        padding: 8px 16px !important;
        text-align: left !important;
        font-size: 13px;
    }
    
    .date-separator td { padding: 8px 16px !important; }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }
    
    .metric-item {
        background: white;
        border-radius: 16px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #eef2f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    .metric-item:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transform: translateY(-2px);
    }
    
    .metric-item .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8894a0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-item .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1a2634;
        margin-top: 4px;
    }
    
    .metric-item .metric-unit {
        font-size: 13px;
        font-weight: 400;
        color: #6b7a8a;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1a2634;
        margin: 28px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .info-box {
        background: #f8fafc;
        border: 1px solid #eef2f6;
        border-radius: 16px;
        padding: 16px 20px;
        margin-top: 20px;
    }
    
    .info-box .title {
        font-weight: 600;
        color: #1a2634;
        margin-bottom: 4px;
        font-size: 14px;
    }
    
    .info-box .content {
        color: #6b7a8a;
        font-size: 13px;
        line-height: 1.6;
    }
    
    .update-badge {
        display: inline-block;
        background: #2ecc71;
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 12px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @media (max-width: 768px) {
        .weather-main { gap: 20px; }
        .weather-temp { font-size: 40px; }
        .weather-card { padding: 20px; }
        .weather-details { gap: 16px 20px; }
        .forecast-section { padding: 16px; }
        .forecast-table td { padding: 6px 8px; font-size: 12px; }
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """, unsafe_allow_html=True)


def render_weather_card(latest_data, ts_formatted):
    category = latest_data.get("category", "Baik")
    color = CATEGORY_COLOR.get(category, "#95a5a6")
    emoji = CATEGORY_EMOJI.get(category, "🌤️")
    pm25 = latest_data.get("pm25", 0)
    temp = latest_data.get("temperature", 0)
    hum = latest_data.get("relativehumidity", 0)
    desc = CATEGORY_DESC.get(category, "")
    
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-main">
            <div class="weather-icon">{emoji}</div>
            <div>
                <div class="weather-temp">{pm25:.0f}<span class="weather-temp-unit"> µg/m³</span></div>
                <div class="weather-desc">
                    {category}
                    <span class="weather-badge" style="background:{color}22; color:{color}; border:1px solid {color}44;">
                        {CATEGORY_ADVICE.get(category, '')}
                    </span>
                </div>
                <div class="weather-location">📍 {LOCATION['name']} • {ts_formatted} WIB</div>
            </div>
        </div>
        <div class="weather-details">
            <div class="weather-detail-item">
                <span class="label">💧 Kelembapan</span>
                <span class="value">{hum:.0f}%</span>
            </div>
            <div class="weather-detail-item">
                <span class="label">🌡️ Suhu</span>
                <span class="value">{temp:.1f}°C</span>
            </div>
            <div class="weather-detail-item">
                <span class="label">🫧 PM1</span>
                <span class="value">{latest_data.get('pm1', 0):.1f} µg/m³</span>
            </div>
            <div class="weather-detail-item">
                <span class="label">🔬 Partikel</span>
                <span class="value">{latest_data.get('um003', 0):.0f}</span>
            </div>
            <div class="weather-detail-item">
                <span class="label">📊 Status</span>
                <span class="value" style="color:{color};">{desc}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_grid(data, category):
    metrics = [
        ("PM2.5", f"{data.get('pm25', 0):.1f}", "µg/m³"),
        ("PM1", f"{data.get('pm1', 0):.1f}", "µg/m³"),
        ("Suhu", f"{data.get('temperature', 0):.1f}", "°C"),
        ("Kelembapan", f"{data.get('relativehumidity', 0):.0f}", "%"),
        ("Kategori", category, ""),
        ("Partikel", f"{data.get('um003', 0):.0f}", ""),
    ]
    
    html = '<div class="metric-grid">'
    for label, value, unit in metrics:
        html += f"""
        <div class="metric-item">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>
        </div>
        """
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_hourly_forecast(df, title, key_prefix, page_size=8):
    if df is None or df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.markdown(f'<div class="forecast-header">📋 {title} <span class="timezone">(WIB)</span></div>', unsafe_allow_html=True)

    df = df.copy().reset_index(drop=True)
    df["timestamp_wib"] = df["timestamp"] + timedelta(hours=7)
    df = df.sort_values("timestamp_wib").reset_index(drop=True)

    total = len(df)
    page_key = f"{key_prefix}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    max_page = max(0, (total - 1) // page_size)
    st.session_state[page_key] = min(st.session_state[page_key], max_page)

    with col2:
        if st.button("‹", key=f"{key_prefix}_prev", disabled=st.session_state[page_key] <= 0, use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
    with col3:
        if st.button("›", key=f"{key_prefix}_next", disabled=st.session_state[page_key] >= max_page, use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()

    start = st.session_state[page_key] * page_size
    end = start + page_size
    window = df.iloc[start:end]

    html = '<div class="forecast-scroll"><table class="forecast-table">'
    
    prev_date = None
    for _, row in window.iterrows():
        ts = row["timestamp_wib"]
        date_str = ts.strftime("%-d %b %Y")
        hour_str = ts.strftime("%H.%M")
        cat = row.get("category", "Baik")
        emoji = CATEGORY_EMOJI.get(cat, "🌤️")
        pm25 = row.get("pm25", 0)
        temp = row.get("temperature", 0)
        hum = row.get("relativehumidity", 0)
        
        if date_str != prev_date:
            html += f'<tr><td colspan="7" class="date-separator">{date_str}</td></tr>'
            prev_date = date_str
        
        html += f"""
        <tr>
            <td class="time-cell">{hour_str}</td>
            <td class="icon-cell">{emoji}</td>
            <td class="temp-cell">{pm25:.0f} µg/m³</td>
            <td class="detail-cell">{temp:.0f}°C</td>
            <td class="detail-cell">{hum:.0f}%</td>
            <td class="detail-cell" style="color:{CATEGORY_COLOR.get(cat, '#95a5a6')};">{cat}</td>
        </tr>
        """
    
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Menampilkan jam ke-{start + 1}–{min(end, total)} dari {total}")


# ============ MAIN ============
def main():
    inject_custom_css()
    
    st.markdown('<div class="main-header">🌤️ Kualitas Udara Malang</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">📍 {LOCATION["name"]} · Update setiap 1 jam dari OpenAQ</div>', unsafe_allow_html=True)
    
    # Start streaming thread
    if 'stream_thread' not in st.session_state:
        thread = Thread(target=stream_worker, daemon=True)
        thread.start()
        st.session_state.stream_thread = thread
        st.session_state.stream_started = datetime.now()
    
    # Process data from queue
    with st.session_state.data_lock:
        if not data_queue.empty():
            try:
                recent, historical, update_time = data_queue.get_nowait()
                if recent:
                    st.session_state.cached_data['recent'] = recent
                    st.session_state.cached_data['history'] = historical
                    st.session_state.cached_data['last_update'] = update_time
                    st.session_state.cached_data['cache_version'] += 1
            except:
                pass
    
    # Load model
    with st.spinner("Memuat model..."):
        spark, model = load_model()
    
    # ============ AMBIL DATA DARI CACHE ============
    recent_data = st.session_state.cached_data.get('recent')
    history_data = st.session_state.cached_data.get('history', [])
    
    # If no cached data, fetch immediately
    if recent_data is None:
        with st.spinner("Mengambil data dari OpenAQ..."):
            stream = OpenAQStream(OPENAQ_API_KEY)
            stream.discover_sensors()
            recent, historical = stream.fetch_all_data()
            if recent:
                recent_data = recent
                history_data = historical
                st.session_state.cached_data['recent'] = recent
                st.session_state.cached_data['history'] = historical
                st.session_state.cached_data['last_update'] = datetime.now()
    
    if recent_data is None:
        st.warning("⚠️ Belum ada data dari OpenAQ. Tunggu update berikutnya.")
        st.stop()
    
    # ============ BATCH PREDICT ============
    with st.spinner("Memproses data..."):
        # Predict latest
        kategori = predict_batch(spark, model, [recent_data])[0]
        recent_data["category"] = kategori
        
        # Process historical
        if history_data and st.session_state.cached_data.get('cache_version', 0) != st.session_state.get('last_cache_version', 0):
            df_hist = pd.DataFrame(history_data)
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True).dt.tz_localize(None)
            df_hist = df_hist.sort_values('timestamp')
            
            anchor_ts = pd.to_datetime(recent_data["timestamp"])
            cutoff = anchor_ts - timedelta(hours=24)
            df_hist = df_hist[(df_hist['timestamp'] >= cutoff) & (df_hist['timestamp'] <= anchor_ts)].copy()
            
            if len(df_hist) > 0:
                hist_data_list = df_hist.to_dict('records')
                hist_categories = predict_batch(spark, model, hist_data_list)
                df_hist['category'] = hist_categories
                st.session_state.cached_data['predicted_history'] = df_hist
            else:
                st.session_state.cached_data['predicted_history'] = pd.DataFrame()
            
            # Process future
            anchor_ts = pd.to_datetime(recent_data["timestamp"])
            feature_cols = ["pm1", "pm25", "relativehumidity", "temperature", "um003"]
            
            hourly_pattern = None
            if history_data:
                df_pattern_src = pd.DataFrame(history_data)
                df_pattern_src["timestamp"] = pd.to_datetime(df_pattern_src["timestamp"], utc=True).dt.tz_localize(None)
                available_cols = [c for c in feature_cols if c in df_pattern_src.columns]
                if available_cols:
                    hourly_pattern = get_hourly_pattern(df_pattern_src, available_cols)
            
            future_data = []
            for i in range(1, 25):
                future_time = anchor_ts + timedelta(hours=i)
                target_hour = future_time.hour
                
                if hourly_pattern is not None:
                    base = hourly_pattern.loc[target_hour]
                    future = {
                        "timestamp": future_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "pm1": max(0, base.get("pm1", recent_data.get("pm1", 10))),
                        "pm25": max(0, base.get("pm25", recent_data.get("pm25", 15))),
                        "relativehumidity": max(0, min(100, base.get("relativehumidity", recent_data.get("relativehumidity", 65)))),
                        "temperature": max(0, min(45, base.get("temperature", recent_data.get("temperature", 27)))),
                        "um003": max(0, base.get("um003", recent_data.get("um003", 300))),
                    }
                else:
                    future = {
                        "timestamp": future_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "pm1": recent_data.get("pm1", 10),
                        "pm25": recent_data.get("pm25", 15),
                        "relativehumidity": recent_data.get("relativehumidity", 65),
                        "temperature": recent_data.get("temperature", 27),
                        "um003": recent_data.get("um003", 300),
                    }
                future_data.append(future)
            
            future_categories = predict_batch(spark, model, future_data)
            for i, cat in enumerate(future_categories):
                future_data[i]["category"] = cat
            
            df_future = pd.DataFrame(future_data)
            df_future['timestamp'] = pd.to_datetime(df_future['timestamp'], utc=True).dt.tz_localize(None)
            st.session_state.cached_data['predicted_future'] = df_future
            st.session_state.last_cache_version = st.session_state.cached_data['cache_version']
        else:
            df_hist = st.session_state.cached_data.get('predicted_history', pd.DataFrame())
            df_future = st.session_state.cached_data.get('predicted_future', pd.DataFrame())
    
    # ============ FORMAT TIMESTAMP ============
    ts = recent_data.get('timestamp', 'N/A')
    try:
        dt_utc = pd.to_datetime(ts)
        dt_wib = dt_utc + timedelta(hours=7)
        ts_formatted = dt_wib.strftime('%d %b %Y, %H:%M')
        ts_display = dt_wib.strftime('%Y-%m-%d %H:%M')
    except Exception:
        ts_formatted = ts
        ts_display = ts
    
    # Show last update time
    last_update = st.session_state.cached_data.get('last_update')
    if last_update:
        time_diff = datetime.now() - last_update
        minutes_ago = int(time_diff.total_seconds() / 60)
        if minutes_ago < 60:
            st.caption(f"🔄 Data terakhir diperbarui {minutes_ago} menit yang lalu")
        else:
            st.caption(f"🔄 Data terakhir diperbarui {int(minutes_ago/60)} jam yang lalu")
    
    # ============ RENDER ============
    render_weather_card(recent_data, ts_formatted)
    render_metric_grid(recent_data, kategori)
    
    # Historical
    st.markdown('<div class="section-title">📊 Historis 24 Jam</div>', unsafe_allow_html=True)
    if not df_hist.empty and len(df_hist) > 0:
        render_hourly_forecast(df_hist, "Historis 24 Jam Terakhir", key_prefix="hist")
    else:
        st.info("⏳ Belum cukup data historis (minimal 24 jam)")
    
    # Future
    st.markdown('<div class="section-title">🔮 Prediksi 24 Jam</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ Tentang Prediksi", expanded=False):
        st.markdown("""
        **Bagaimana prediksi dibuat?**
        
        Fitur (PM2.5, PM1, suhu, kelembapan, um003) untuk tiap jam ke depan diestimasi dari 
        pola nilai per jam-dalam-hari pada data historis yang tersedia. Random Forest kemudian 
        mengklasifikasikan kategori ISPU dari fitur hasil estimasi tersebut.
        """)
    
    if not df_future.empty:
        render_hourly_forecast(df_future, "Prediksi 24 Jam Ke Depan", key_prefix="future", page_size=8)
    
    # Summary
    st.markdown('<div class="section-title">📋 Ringkasan</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Kondisi Saat Ini", recent_data.get("category", "Baik"))
    with col2:
        st.metric("PM2.5 Saat Ini", f"{recent_data.get('pm25', 0):.1f} µg/m³")
    with col3:
        pred_12h = df_future.head(12)['category'].mode()[0] if not df_future.head(12)['category'].empty else "N/A"
        st.metric("Prediksi 12 Jam", pred_12h)
    with col4:
        pred_24h = df_future['category'].mode()[0] if not df_future['category'].empty else "N/A"
        st.metric("Prediksi 24 Jam", pred_24h)
    
    # Cache stats
    cache_size = len(st.session_state.prediction_cache)
    st.caption(f"⚡ Cache: {cache_size} prediksi tersimpan")
    
    st.markdown(f"""
    <div class="info-box">
        <div class="title">📊 Informasi Data</div>
        <div class="content">
            <strong>Sumber:</strong> OpenAQ API v3 &nbsp;·&nbsp; 
            <strong>Lokasi:</strong> {LOCATION['name']} &nbsp;·&nbsp;
            <strong>Parameter:</strong> PM2.5, PM1, Suhu, Kelembapan, um003 &nbsp;·&nbsp;
            <strong>Data terakhir:</strong> {ts_display} WIB &nbsp;·&nbsp;
            <strong>Update berikutnya:</strong> +1 jam
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
