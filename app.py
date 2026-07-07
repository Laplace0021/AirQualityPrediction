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
    """CSS untuk tampilan seperti aplikasi cuaca dengan light mode."""
    st.markdown("""
    <style>
    /* Force Light Mode */
    .stApp {
        background-color: #f5f7fa !important;
    }
    
    .main {
        background-color: #f5f7fa !important;
    }
    
    /* Global Styles */
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
    
    /* Main Weather Card - Like Weather App */
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
    
    .weather-icon {
        font-size: 72px;
        line-height: 1;
    }
    
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
    
    .weather-detail-item .label {
        color: #8894a0;
        font-weight: 500;
    }
    
    .weather-detail-item .value {
        font-weight: 600;
        color: #1a2634;
    }
    
    .weather-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-left: 12px;
    }
    
    /* Hourly Forecast Section */
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
    
    .forecast-table tr:last-child td {
        border-bottom: none;
    }
    
    .forecast-table .date-cell {
        font-weight: 700;
        color: #1a2634;
        text-align: left;
        font-size: 13px;
        min-width: 100px;
    }
    
    .forecast-table .time-cell {
        color: #6b7a8a;
        font-size: 12px;
        min-width: 60px;
    }
    
    .forecast-table .icon-cell {
        font-size: 28px;
        min-width: 50px;
    }
    
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
    
    .forecast-table .pm25-cell {
        font-weight: 600;
        color: #2c3e50;
        font-size: 13px;
        min-width: 65px;
    }
    
    /* Date separator in forecast */
    .date-separator {
        background: #f8fafc;
        font-weight: 700;
        color: #1a2634;
        padding: 8px 16px !important;
        text-align: left !important;
        font-size: 13px;
    }
    
    .date-separator td {
        padding: 8px 16px !important;
    }
    
    /* Metric cards */
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
    
    /* Section headers */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1a2634;
        margin: 28px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Navigation buttons */
    .nav-button {
        background: white !important;
        border: 1px solid #eef2f6 !important;
        border-radius: 12px !important;
        color: #1a2634 !important;
        font-size: 18px !important;
        padding: 4px 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .nav-button:hover {
        background: #f8fafc !important;
        border-color: #d0d8e0 !important;
    }
    
    .nav-button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }
    
    /* Info box */
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
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .weather-main {
            gap: 20px;
        }
        
        .weather-temp {
            font-size: 40px;
        }
        
        .weather-card {
            padding: 20px;
        }
        
        .weather-details {
            gap: 16px 20px;
        }
        
        .forecast-section {
            padding: 16px;
        }
        
        .forecast-table td {
            padding: 6px 8px;
            font-size: 12px;
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    </style>
    """, unsafe_allow_html=True)


def render_weather_card(latest_data, ts_formatted):
    """Render kartu cuaca utama seperti aplikasi cuaca."""
    category = latest_data.get("category", "Baik")
    color = CATEGORY_COLOR.get(category, "#95a5a6")
    emoji = CATEGORY_EMOJI.get(category, "🌤️")
    pm25 = latest_data.get("pm25", 0)
    temp = latest_data.get("temperature", 0)
    hum = latest_data.get("relativehumidity", 0)
    
    # Tentukan deskripsi berdasarkan kategori
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
    """Render grid metric seperti aplikasi cuaca."""
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
    """
    Render forecast per jam seperti aplikasi cuaca.
    """
    if df is None or df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    # Header with navigation
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

    # Build HTML table
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
        
        # Date separator
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


def main():
    global recent_data, history_data
    
    # Inject custom CSS (light mode)
    inject_custom_css()
    
    # Header
    st.markdown('<div class="main-header">🌤️ Kualitas Udara Malang</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">📍 {LOCATION["name"]} · Update setiap 1 jam dari OpenAQ</div>', unsafe_allow_html=True)
    
    # Start streaming
    if 'stream_thread' not in st.session_state:
        thread = Thread(target=stream_worker, daemon=True)
        thread.start()
        st.session_state.stream_thread = thread
    
    # Load model
    with st.spinner("Memuat model..."):
        spark, model = load_model()
    
    # ============ AMBIL DATA ============
    while not data_queue.empty():
        data_queue.get()

    if not history_data:
        with st.spinner("Mengambil data dari OpenAQ..."):
            stream = OpenAQStream(OPENAQ_API_KEY)
            stream.discover_sensors()
            recent, historical = stream.fetch_all_data()
            if recent:
                recent_data = recent
                history_data = historical

    latest_data = get_anchor_data(history_data)

    if latest_data is None:
        st.warning("⚠️ Belum ada data dari OpenAQ. Tunggu update berikutnya.")
        
        with st.expander("🔍 Status Sensor", expanded=True):
            stream = OpenAQStream(OPENAQ_API_KEY)
            sensors = stream.discover_sensors()
            if sensors:
                st.success(f"✅ Ditemukan {len(sensors)} sensor!")
                for param, sid in sensors.items():
                    st.write(f"- {param}: sensor_id={sid}")
                
                st.write("**Mencoba ambil & gabungkan data measurement...**")
                recent_try, historical_try = stream.fetch_all_data()
                st.write("**Detail debug fetch:**")
                st.json(stream.last_debug)
                
                if not historical_try:
                    st.error("❌ Sensor ditemukan, tapi belum berhasil menggabungkan data measurement jadi baris historis.")
                else:
                    st.success(f"✅ Berhasil menggabungkan {len(historical_try)} baris data per jam.")
            else:
                st.error("❌ Tidak ditemukan sensor di lokasi Malang")
                st.write("**Detail debug:**")
                st.json(stream.last_debug)
        st.stop()
    
    # Prediksi kategori
    kategori = predict(spark, model, latest_data)
    latest_data["category"] = kategori
    
    # ============ FORMAT TIMESTAMP ============
    ts = latest_data.get('timestamp', 'N/A')
    try:
        dt_utc = pd.to_datetime(ts)
        dt_wib = dt_utc + timedelta(hours=7)
        ts_formatted = dt_wib.strftime('%d %b %Y, %H:%M')
        ts_display = dt_wib.strftime('%Y-%m-%d %H:%M')
    except Exception:
        ts_formatted = ts
        ts_display = ts
    
    # ============ WEATHER CARD ============
    render_weather_card(latest_data, ts_formatted)
    
    # Metric Grid
    render_metric_grid(latest_data, kategori)
    
    # ============ HISTORIS PER JAM ============
    st.markdown('<div class="section-title">📊 Historis 24 Jam</div>', unsafe_allow_html=True)

    if history_data and len(history_data) > 0:
        df_hist = pd.DataFrame(history_data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True).dt.tz_localize(None)
        df_hist = df_hist.sort_values('timestamp')

        anchor_ts = pd.to_datetime(latest_data["timestamp"])
        cutoff = anchor_ts - timedelta(hours=24)
        df_hist = df_hist[(df_hist['timestamp'] >= cutoff) & (df_hist['timestamp'] <= anchor_ts)].copy()

        if len(df_hist) > 0:
            categories = []
            for _, row in df_hist.iterrows():
                cat = predict(spark, model, row.to_dict())
                categories.append(cat)
            df_hist['category'] = categories

            render_hourly_forecast(df_hist, "Historis 24 Jam Terakhir", key_prefix="hist")
        else:
            st.info("⏳ Belum cukup data historis (minimal 24 jam)")
    else:
        st.info("⏳ Menunggu data historis...")

    # ============ PREDIKSI PER JAM ============
    st.markdown('<div class="section-title">🔮 Prediksi 24 Jam</div>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Prediksi", expanded=False):
        st.markdown("""
        **Bagaimana prediksi dibuat?**
        
        Fitur (PM2.5, PM1, suhu, kelembapan, um003) untuk tiap jam ke depan diestimasi dari 
        pola nilai per jam-dalam-hari pada data historis yang tersedia. Random Forest kemudian 
        mengklasifikasikan kategori ISPU dari fitur hasil estimasi tersebut.
        """)

    anchor_ts = pd.to_datetime(latest_data["timestamp"])
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
                "pm1": max(0, base.get("pm1", latest_data.get("pm1", 10))),
                "pm25": max(0, base.get("pm25", latest_data.get("pm25", 15))),
                "relativehumidity": max(0, min(100, base.get("relativehumidity", latest_data.get("relativehumidity", 65)))),
                "temperature": max(0, min(45, base.get("temperature", latest_data.get("temperature", 27)))),
                "um003": max(0, base.get("um003", latest_data.get("um003", 300))),
            }
        else:
            future = {
                "timestamp": future_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pm1": latest_data.get("pm1", 10),
                "pm25": latest_data.get("pm25", 15),
                "relativehumidity": latest_data.get("relativehumidity", 65),
                "temperature": latest_data.get("temperature", 27),
                "um003": latest_data.get("um003", 300),
            }

        future["category"] = predict(spark, model, future)
        future_data.append(future)
    
    df_future = pd.DataFrame(future_data)
    df_future['timestamp'] = pd.to_datetime(df_future['timestamp'], utc=True).dt.tz_localize(None)

    render_hourly_forecast(df_future, "Prediksi 24 Jam Ke Depan", key_prefix="future", page_size=8)
    
    # ============ RINGKASAN ============
    st.markdown('<div class="section-title">📋 Ringkasan</div>', unsafe_allow_html=True)
    
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
    
    # Info Box
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
