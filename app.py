"""
app.py
Dashboard Kualitas Udara Malang - OpenAQ v3
Menampilkan: Recent, Historis 24 Jam, Prediksi 24 Jam
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

data_queue = queue.Queue(maxsize=100)
history_data = []
recent_data = None


class OpenAQStream:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        self.location_id = None
        self.last_debug = {}
        
    def discover_sensors(self):
        """
        Cari location_id berdasarkan koordinat, lalu ambil daftar sensornya.
        PENTING: endpoint sensor yang benar di OpenAQ v3 adalah nested resource:
            GET /locations/{id}/sensors
        BUKAN /sensors?location_id=... (itu bukan endpoint yang valid, makanya
        sebelumnya selalu gagal walau location_id berhasil ditemukan).
        """
        self.last_debug = {}
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
                timeout=15
            )

            print("DEBUG /locations status:", resp.status_code)
            print("DEBUG /locations body:", resp.text[:2000])

            self.last_debug["locations_status"] = resp.status_code
            self.last_debug["locations_body"] = resp.text[:1000]

            if resp.status_code == 401:
                self.last_debug["reason"] = "401 Unauthorized -> API key salah/kadaluarsa."
                return {}
            if resp.status_code != 200:
                self.last_debug["reason"] = f"Request /locations gagal, status {resp.status_code}."
                return {}

            locations = resp.json().get("results", [])
            self.last_debug["locations_found"] = len(locations)

            if not locations:
                self.last_debug["reason"] = (
                    f"Tidak ada lokasi OpenAQ dalam radius {LOCATION['radius']}m dari koordinat ini."
                )
                return {}

            self.location_id = locations[0].get("id")
            self.last_debug["location_id"] = self.location_id
            self.last_debug["location_name"] = locations[0].get("name")

            # Endpoint sensor yang benar: nested resource per location
            sensor_resp = requests.get(
                f"{OPENAQ_BASE_URL}/locations/{self.location_id}/sensors",
                headers=self.headers,
                timeout=15
            )

            print("DEBUG /locations/{id}/sensors status:", sensor_resp.status_code)
            print("DEBUG /locations/{id}/sensors body:", sensor_resp.text[:2000])

            self.last_debug["sensors_status"] = sensor_resp.status_code
            self.last_debug["sensors_body"] = sensor_resp.text[:1000]

            if sensor_resp.status_code == 404:
                self.last_debug["reason"] = f"404 -> location_id {self.location_id} tidak punya endpoint sensors (mungkin sudah tidak aktif)."
                return {}
            if sensor_resp.status_code != 200:
                self.last_debug["reason"] = f"Request sensors gagal, status {sensor_resp.status_code}."
                return {}

            sensors = sensor_resp.json().get("results", [])
            self.last_debug["sensors_found"] = len(sensors)
            self.last_debug["available_parameters"] = [
                s.get("parameter", {}).get("name", "").lower() for s in sensors
            ]

            self.sensor_ids = {}
            for sensor in sensors:
                param = sensor.get("parameter", {}).get("name", "").lower()
                self.sensor_ids[param] = sensor.get("id")

            required = {"pm1", "pm25", "relativehumidity", "temperature"}
            missing = required - set(self.sensor_ids.keys())
            if missing:
                self.last_debug["missing_required_params"] = list(missing)
                self.last_debug["note"] = (
                    "Lokasi ini tidak punya semua parameter yang dibutuhkan model. "
                    "Jika ini terjadi, model tidak bisa dijalankan penuh untuk lokasi ini."
                )

            return self.sensor_ids

        except Exception as e:
            self.last_debug["reason"] = f"Exception: {e}"
            return {}
    
    def get_sensor_data(self, sensor_id, datetime_from=None, datetime_to=None, limit=50):
        """
        Ambil data sensor dengan rentang waktu.
        Coba /measurements dulu; kalau gagal/kosong, fallback ke /hours
        (sensor AirGradient ini melaporkan data teragregasi per jam berdasarkan
        field coverage.expectedInterval="01:00:00" pada respons /locations/{id}/sensors).
        """
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
                timeout=15
            )

            print(f"DEBUG /sensors/{sensor_id}/{endpoint} params={params} status={resp.status_code}")
            print(f"DEBUG body: {resp.text[:1500]}")

            key = f"{endpoint}_sensor_{sensor_id}"
            self.last_debug[key] = {
                "params": params,
                "status": resp.status_code,
                "body": resp.text[:1000],
            }

            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
        except Exception as e:
            self.last_debug[f"{endpoint}_sensor_{sensor_id}"] = {"exception": str(e)}
            return []
    
    def fetch_all_data(self):
        """
        Fetch semua data: recent + historical.
        Digabung per JAM (bukan exact timestamp match), karena sensor PM1/PM2.5/RH/suhu/um003
        hampir tidak pernah melapor pada detik yang persis sama. Sebelumnya kode ini mensyaratkan
        kombinasi timestamp identik antar sensor -> hampir selalu kosong walau sensor ditemukan.
        """
        if not self.sensor_ids:
            self.discover_sensors()
        if not self.sensor_ids:
            return None, []

        now = datetime.utcnow()
        datetime_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        datetime_from_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

        per_param_rows = {}

        for param, sensor_id in self.sensor_ids.items():
            results = self.get_sensor_data(sensor_id, datetime_from_24h, datetime_to, limit=200)
            if not results:
                results = self.get_sensor_data(sensor_id, limit=200)

            rows = []
            for r in results:
                # /measurements pakai r["datetime"]["utc"], /hours pakai r["period"]["datetimeFrom"]["utc"]
                timestamp = r.get("datetime", {}).get("utc") if r.get("datetime") else None
                if not timestamp and r.get("period"):
                    timestamp = r.get("period", {}).get("datetimeFrom", {}).get("utc")
                value = r.get("value")
                if timestamp and value is not None:
                    rows.append((timestamp, value))
            per_param_rows[param] = rows
            self.last_debug[f"raw_count_{param}"] = len(rows)

        # Bangun rata-rata per jam untuk tiap parameter, lalu gabungkan by jam
        hourly_frames = {}
        for param, rows in per_param_rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["timestamp", param])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour_bucket"] = df["timestamp"].dt.floor("h")
            hourly_frames[param] = df.groupby("hour_bucket")[param].mean()

        if not hourly_frames:
            self.last_debug["fetch_reason"] = "Tidak ada satupun measurement valid yang dikembalikan sensor."
            return None, []

        combined = pd.DataFrame(hourly_frames).reset_index().rename(columns={"hour_bucket": "timestamp_dt"})
        self.last_debug["combined_hours_before_filter"] = len(combined)
        self.last_debug["combined_columns"] = list(combined.columns)

        required = [c for c in ["pm25", "relativehumidity", "temperature"] if c in combined.columns]
        if required:
            combined = combined.dropna(subset=required)
        self.last_debug["combined_hours_after_filter"] = len(combined)

        if combined.empty:
            self.last_debug["fetch_reason"] = (
                "Tiap parameter punya data, tapi tidak ada satu jam pun dengan kombinasi "
                "pm25 + relativehumidity + temperature lengkap dalam 24 jam terakhir."
            )
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
    """Worker streaming - jalan di background"""
    stream = OpenAQStream(OPENAQ_API_KEY)
    
    while True:
        try:
            # Fetch semua data sekaligus
            recent, historical = stream.fetch_all_data()
            
            if recent:
                if not data_queue.full():
                    data_queue.put(recent)
                global recent_data
                recent_data = recent
            
            if historical:
                global history_data
                history_data = historical
            
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


def get_anchor_data(history):
    """
    Mengambil data historis yang paling dekat dengan jam sekarang (dibulatkan ke bawah).
    Contoh: sekarang 10:45 -> cari data jam 10:00.
    Kalau tidak ada -> pakai data terakhir sebelum jam 10:00.
    Kalau tidak ada juga -> pakai data terakhir yang tersedia (paling baru).
    """
    if not history:
        return None

    df = pd.DataFrame(history)
    # utc=True memaksa hasil jadi tz-aware (UTC) walaupun sebagian string ada yang
    # tidak berakhiran "Z"/offset, lalu tz_localize(None) melepas info tz supaya
    # bisa dibandingkan dengan anchor yang juga naive. Semua tetap merepresentasikan UTC.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    # PENTING: pakai UTC, bukan datetime.now() (waktu lokal server), karena semua
    # timestamp dari OpenAQ dalam UTC. Kalau pakai now() lokal, anchor bisa salah
    # beberapa jam tergantung timezone server (mis. WIB = UTC+7).
    now = datetime.utcnow()
    anchor = now.replace(minute=0, second=0, microsecond=0)

    before = df[df["timestamp"] <= anchor]

    if len(before) > 0:
        return before.sort_values("timestamp").iloc[-1].to_dict()

    return df.sort_values("timestamp").iloc[-1].to_dict()


def get_hourly_pattern(history_df, feature_cols):
    """
    Menghitung rata-rata nilai tiap fitur berdasarkan jam-dalam-hari (0-23)
    dari data historis. Dipakai sebagai dasar prediksi fitur masa depan
    (pola diurnal), menggantikan noise acak (np.random.normal).
    Jika suatu jam tidak punya data historis, akan di-interpolasi dari jam terdekat.
    """
    df = history_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    pattern = df.groupby("hour")[feature_cols].mean()

    # Lengkapi 24 jam penuh, isi jam yang kosong dengan interpolasi melingkar
    full_index = pd.Index(range(24), name="hour")
    pattern = pattern.reindex(full_index)
    pattern = pd.concat([pattern, pattern, pattern])
    pattern = pattern.interpolate(limit_direction="both")
    pattern = pattern.iloc[24:48]
    pattern.index = range(24)

    return pattern


def inject_custom_css():
    """CSS untuk kartu 'Saat Ini' dan strip per-jam bergaya widget cuaca."""
    st.markdown("""
    <style>
    /* Global Styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a2634;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        color: #6b7a8a;
        font-size: 1rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #f0f4f8;
        padding-bottom: 1rem;
    }
    
    /* Main Card */
    .wx-card {
        display: flex;
        align-items: center;
        gap: 28px;
        flex-wrap: wrap;
        padding: 28px 32px;
        border-radius: 24px;
        border: 1px solid #e6eef5;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    
    .wx-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .wx-icon-circle {
        width: 96px;
        height: 96px;
        min-width: 96px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 44px;
        background: #f0f4f8;
        border: 3px solid #e0e8ef;
        transition: all 0.3s ease;
    }
    
    .wx-main { flex: 1; min-width: 240px; }
    
    .wx-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #8894a0;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .wx-value { 
        font-size: 44px; 
        font-weight: 800; 
        color: #1b2733; 
        line-height: 1.1;
    }
    
    .wx-unit { 
        font-size: 18px; 
        font-weight: 600; 
        color: #4a5661; 
        margin-left: 4px; 
    }
    
    .wx-cat { 
        font-size: 18px; 
        font-weight: 600; 
        margin-left: 14px; 
    }
    
    .wx-loc { 
        font-size: 15px; 
        color: #6b7684; 
        margin-left: 8px; 
    }
    
    .wx-meta { 
        font-size: 12.5px; 
        color: #8894a0; 
        margin-top: 4px; 
    }
    
    .wx-chip-row { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 10px; 
        margin-top: 16px; 
    }
    
    .wx-chip {
        background: #f4f8fc;
        border: 1px solid #e5edf5;
        border-radius: 24px;
        padding: 7px 16px;
        font-size: 13.5px;
        color: #4a5661;
        white-space: nowrap;
        transition: all 0.2s ease;
    }
    
    .wx-chip:hover {
        background: #e8f0f8;
        transform: scale(1.02);
    }
    
    .wx-chip b { color: #1b2733; }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e6eef5;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transform: translateY(-3px);
    }
    
    .metric-card .label {
        font-size: 0.8rem;
        color: #8894a0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #1b2733;
    }
    
    .metric-card .sub {
        font-size: 0.75rem;
        color: #8894a0;
        margin-top: 4px;
    }
    
    /* Hourly Strip */
    .hourly-wrap { 
        overflow-x: auto; 
        padding-bottom: 6px; 
        margin: 8px 0;
    }
    
    .hourly-table { 
        border-collapse: collapse; 
        min-width: 100%;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .hourly-table td {
        text-align: center;
        padding: 10px 16px;
        font-size: 13px;
        white-space: nowrap;
        vertical-align: middle;
        border-bottom: 1px solid #f0f4f8;
    }
    
    .hourly-table tr:last-child td {
        border-bottom: none;
    }
    
    .hourly-table td:first-child {
        position: sticky;
        left: 0;
        background: #f8fafc;
        text-align: left;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #8894a0;
        text-transform: uppercase;
        min-width: 128px;
        z-index: 2;
        border-right: 1px solid #e6eef5;
    }
    
    .hourly-date { 
        font-weight: 700 !important; 
        font-size: 12.5px !important; 
        color: #1b2733 !important; 
        text-align: left !important; 
    }
    
    .hourly-hour { 
        color: #8894a0; 
        font-size: 12px; 
    }
    
    .hourly-icon { 
        font-size: 24px; 
    }
    
    .hourly-val { 
        font-weight: 700; 
        font-size: 15px; 
        color: #1b2733; 
    }
    
    .hourly-sub { 
        color: #5d6875; 
        font-size: 12.5px; 
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a2634;
        margin-top: 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-header span {
        font-size: 1.8rem;
    }
    
    /* Info Box */
    .info-box {
        background: #f8fafc;
        border: 1px solid #e6eef5;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
    }
    
    .info-box .title {
        font-weight: 600;
        color: #1b2733;
        margin-bottom: 4px;
    }
    
    .info-box .content {
        color: #6b7a8a;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .wx-card {
            flex-direction: column;
            align-items: flex-start;
            padding: 20px;
        }
        
        .wx-value {
            font-size: 32px;
        }
        
        .metric-card .value {
            font-size: 1.5rem;
        }
        
        .hourly-table td {
            padding: 8px 12px;
            font-size: 12px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def render_current_card(latest_data, ts_formatted):
    """Kartu 'Saat Ini' bergaya widget cuaca: ikon kategori besar + chip info."""
    category = latest_data.get("category", "Baik")
    color = CATEGORY_COLOR.get(category, "#95a5a6")
    emoji = CATEGORY_EMOJI.get(category, "🌤️")
    pm25 = latest_data.get("pm25", 0)
    pm1 = latest_data.get("pm1", 0)
    temp = latest_data.get("temperature", 0)
    hum = latest_data.get("relativehumidity", 0)
    um003 = latest_data.get("um003", 0)

    st.markdown(f"""
    <div class="wx-card">
        <div class="wx-icon-circle" style="background:{color}22; border: 3px solid {color};">
            {emoji}
        </div>
        <div class="wx-main">
            <div class="wx-label">📍 Saat Ini</div>
            <div>
                <span class="wx-value">{pm25:.0f}</span><span class="wx-unit">µg/m³ PM2.5</span>
                <span class="wx-cat" style="color:{color};">{category}</span>
                <span class="wx-loc">di {LOCATION['name']}</span>
            </div>
            <div class="wx-meta">🕐 {ts_formatted} WIB &nbsp;•&nbsp; {CATEGORY_DESC.get(category, '')}</div>
            <div class="wx-chip-row">
                <div class="wx-chip">💧 Kelembapan: <b>{hum:.0f}%</b></div>
                <div class="wx-chip">🌡️ Suhu: <b>{temp:.1f}°C</b></div>
                <div class="wx-chip">🫧 PM1: <b>{pm1:.1f} µg/m³</b></div>
                <div class="wx-chip">🔬 Partikel: <b>{um003:.0f}</b></div>
                <div class="wx-chip" style="background:{color}22; border-color:{color};">💡 {CATEGORY_ADVICE.get(category, '')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(data, category):
    """Render metric cards for additional data points."""
    cols = st.columns(5)
    
    metrics = [
        ("PM2.5", f"{data.get('pm25', 0):.1f}", "µg/m³", "#2ecc71"),
        ("PM1", f"{data.get('pm1', 0):.1f}", "µg/m³", "#3498db"),
        ("Suhu", f"{data.get('temperature', 0):.1f}", "°C", "#e67e22"),
        ("Kelembapan", f"{data.get('relativehumidity', 0):.0f}", "%", "#9b59b6"),
        ("Kategori", category, "", CATEGORY_COLOR.get(category, "#95a5a6")),
    ]
    
    for idx, (label, value, unit, color) in enumerate(metrics):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color};">{value} {unit}</div>
            </div>
            """, unsafe_allow_html=True)


def render_hourly_strip(df, title, key_prefix, page_size=8):
    """
    Strip horizontal per jam gaya 'Prakiraan per Jam': kategori, PM2.5, suhu, kelembapan,
    dengan navigasi ‹ ›. Timestamp dikonversi ke WIB (UTC+7) untuk ditampilkan; df["timestamp"]
    input harus tz-naive dan merepresentasikan UTC (lihat get_anchor_data / fetch_all_data).
    """
    header_col, prev_col, next_col = st.columns([8, 1, 1])
    with header_col:
        st.markdown(f'<div style="font-size:1.1rem; font-weight:600; color:#1a2634;">{title}</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    df = df.copy().reset_index(drop=True)
    df["timestamp_wib"] = df["timestamp"] + timedelta(hours=7)
    df = df.sort_values("timestamp_wib").reset_index(drop=True)

    total = len(df)
    page_key = f"{key_prefix}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    max_page = max(0, (total - 1) // page_size)
    st.session_state[page_key] = min(st.session_state[page_key], max_page)

    with prev_col:
        if st.button("‹", key=f"{key_prefix}_prev", disabled=st.session_state[page_key] <= 0, use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
    with next_col:
        if st.button("›", key=f"{key_prefix}_next", disabled=st.session_state[page_key] >= max_page, use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()

    start = st.session_state[page_key] * page_size
    end = start + page_size
    window = df.iloc[start:end]

    date_cells = hour_cells = icon_cells = val_cells = temp_cells = hum_cells = ""
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

        date_html = date_str if date_str != prev_date else ""
        prev_date = date_str

        date_cells += f'<td class="hourly-date">{date_html}</td>'
        hour_cells += f'<td class="hourly-hour">{hour_str}</td>'
        icon_cells += f'<td class="hourly-icon" title="{cat}">{emoji}</td>'
        val_cells += f'<td class="hourly-val">{pm25:.0f}<br><span style="font-weight:400;color:#8894a0;font-size:10.5px;">µg/m³</span></td>'
        temp_cells += f'<td class="hourly-sub">{temp:.0f}°C</td>'
        hum_cells += f'<td class="hourly-sub">{hum:.0f}%</td>'

    html = f"""
    <div class="hourly-wrap">
    <table class="hourly-table">
        <tr><td></td>{date_cells}</tr>
        <tr><td></td>{hour_cells}</tr>
        <tr><td>Kategori</td>{icon_cells}</tr>
        <tr><td>PM2.5</td>{val_cells}</tr>
        <tr><td>Suhu</td>{temp_cells}</tr>
        <tr><td>Kelembapan</td>{hum_cells}</tr>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Menampilkan jam ke-{start + 1}–{min(end, total)} dari {total} · waktu dalam WIB (UTC+7)")


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


def main():
    global recent_data, history_data
    
    # Custom CSS
    inject_custom_css()
    
    # Header
    st.markdown('<div class="main-header">🌤️ Dashboard Kualitas Udara Malang</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">📍 {LOCATION["name"]} · Update data setiap 1 jam dari OpenAQ</div>', unsafe_allow_html=True)
    
    # Start streaming
    if 'stream_thread' not in st.session_state:
        thread = Thread(target=stream_worker, daemon=True)
        thread.start()
        st.session_state.stream_thread = thread
    
    # Load model
    with st.spinner("Memuat model..."):
        spark, model = load_model()
    
    # ============ AMBIL DATA ============
    # Kosongkan antrian queue supaya recent_data/history_data selalu yang terbaru
    while not data_queue.empty():
        data_queue.get()

    # Jika history_data masih kosong, fetch langsung dulu
    if not history_data:
        with st.spinner("Mengambil data dari OpenAQ..."):
            stream = OpenAQStream(OPENAQ_API_KEY)
            stream.discover_sensors()
            recent, historical = stream.fetch_all_data()
            if recent:
                recent_data = recent
                history_data = historical

    # Anchor time: pilih data historis paling dekat dengan jam sekarang
    # (bukan sekadar data[-1] dari API, karena OpenAQ tidak selalu update tepat waktu)
    latest_data = get_anchor_data(history_data)

    # Jika masih tidak ada data
    if latest_data is None:
        st.warning("⚠️ Belum ada data dari OpenAQ. Tunggu update berikutnya.")
        
        # Tampilkan status sensor
        with st.expander
