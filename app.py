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
    page_title="🌤️ Dashboard Kualitas Udara Malang",
    page_icon="🌤️",
    layout="wide"
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
        """Ambil data sensor dengan rentang waktu"""
        try:
            params = {"limit": limit, "sort": "desc"}
            
            if datetime_from:
                params["datetime_from"] = datetime_from
            if datetime_to:
                params["datetime_to"] = datetime_to
            
            resp = requests.get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                params=params,
                headers=self.headers,
                timeout=15
            )
            
            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
        except Exception as e:
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
                timestamp = r.get("datetime", {}).get("utc")
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    now = datetime.now()
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
    
    st.title("🌤️ Dashboard Kualitas Udara Malang")
    st.caption(f"Update data setiap 1 jam dari OpenAQ | Lokasi: {LOCATION['name']}")
    
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
                st.caption(
                    "Cek: status code 401/403 → API key tidak valid atau perlu re-generate di "
                    "explore.openaq.org. Status 200 tapi locations_found=0 → tidak ada lokasi "
                    "OpenAQ dalam radius yang ditentukan (coba perbesar LOCATION['radius'])."
                )
        st.stop()
    
    # Prediksi kategori
    kategori = predict(spark, model, latest_data)
    latest_data["category"] = kategori
    
    # ============ FORMAT TIMESTAMP ============
    ts = latest_data.get('timestamp', 'N/A')
    try:
        dt = pd.to_datetime(ts)
        ts_formatted = dt.strftime('%d %b %Y, %H:%M')
        ts_display = dt.strftime('%Y-%m-%d %H:%M')
    except:
        ts_formatted = ts
        ts_display = ts
    
    # ============ RECENT DATA ============
    st.subheader("📍 Data Terbaru")
    
    col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1])
    
    with col1:
        color = CATEGORY_COLOR.get(latest_data.get("category", "Baik"), "#95a5a6")
        emoji = CATEGORY_EMOJI.get(latest_data.get("category", "Baik"), "🌤️")
        
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
    
    if history_data and len(history_data) > 0:
        df_hist = pd.DataFrame(history_data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist = df_hist.sort_values('timestamp')
        
        # Ambil 24 jam ke belakang berdasarkan anchor time (bukan max data historis)
        anchor_ts = pd.to_datetime(latest_data["timestamp"])
        cutoff = anchor_ts - timedelta(hours=24)
        df_hist = df_hist[(df_hist['timestamp'] >= cutoff) & (df_hist['timestamp'] <= anchor_ts)].copy()
        
        if len(df_hist) > 0:
            # Prediksi kategori untuk historis
            categories = []
            for _, row in df_hist.iterrows():
                cat = predict(spark, model, row.to_dict())
                categories.append(cat)
            df_hist['category'] = categories
            
            # Plot
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
            st.info("⏳ Belum cukup data historis (minimal 24 jam)")
    else:
        st.info("⏳ Menunggu data historis...")
    
    st.markdown("---")
    
    # ============ PREDIKSI 24 JAM ============
    st.subheader("🔮 Prediksi 24 Jam Ke Depan")
    st.caption(
        "Fitur (PM2.5, PM1, suhu, kelembapan, um003) untuk tiap jam ke depan diestimasi dari "
        "pola nilai per jam-dalam-hari pada data historis yang tersedia (bukan angka acak). "
        "Random Forest kemudian mengklasifikasikan kategori ISPU dari fitur hasil estimasi tersebut — "
        "sesuai fungsi aslinya sebagai model klasifikasi, bukan regresi PM2.5."
    )

    anchor_ts = pd.to_datetime(latest_data["timestamp"])
    feature_cols = ["pm1", "pm25", "relativehumidity", "temperature", "um003"]

    hourly_pattern = None
    if history_data:
        df_pattern_src = pd.DataFrame(history_data)
        df_pattern_src["timestamp"] = pd.to_datetime(df_pattern_src["timestamp"])
        available_cols = [c for c in feature_cols if c in df_pattern_src.columns]
        if available_cols:
            hourly_pattern = get_hourly_pattern(df_pattern_src, available_cols)

    future_data = []
    for i in range(1, 25):
        future_time = anchor_ts + timedelta(hours=i)
        target_hour = future_time.hour

        if hourly_pattern is not None:
            # Estimasi fitur dari pola historis pada jam yang sama
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
            # Fallback kalau data historis belum cukup: persistence dari data terakhir
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
    
    st.info(f"""
    📊 **Informasi Data**
    - Sumber: OpenAQ API v3
    - Lokasi: {LOCATION['name']}
    - Parameter: PM2.5, PM1, Suhu, Kelembapan, um003
    - Data terakhir: {ts_display}
    - Update berikutnya: +1 jam
    """)


if __name__ == "__main__":
    main()
