"""
app.py
Aplikasi Prediksi Kualitas Udara dengan Data Streaming Real-time dari OpenAQ v3
Deployed di Streamlit Cloud
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
import warnings
warnings.filterwarnings('ignore')

# ============= AUTO-DETECT JAVA_HOME =============
# Untuk Streamlit Cloud - Java sudah diinstall via packages.txt
if "JAVA_HOME" not in os.environ:
    for candidate in [
        "/usr/lib/jvm/java-17-openjdk-amd64",
        "/usr/lib/jvm/default-java",
        "/usr/lib/jvm/java-11-openjdk-amd64",
    ]:
        if os.path.exists(candidate):
            os.environ["JAVA_HOME"] = candidate
            break

# ============= KONFIGURASI =============
MODEL_PATH = "ispu_rf_model"
FEATURES = ["pm1", "relativehumidity", "temperature", "um003"]

# Ambil API key dari Streamlit secrets atau environment variable
try:
    if hasattr(st, 'secrets') and "OPENAQ_API_KEY" in st.secrets:
        OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]
    else:
        OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")
        
    if not OPENAQ_API_KEY:
        # Fallback untuk testing (tidak aman untuk production)
        OPENAQ_API_KEY = "430a6cbeb038741241c9129a3323543b8a15f7e2b80bd32d9d07b2efb3d66aff"
        st.warning("⚠️ Menggunakan default API key. Untuk production, set di secrets.toml")
except Exception as e:
    st.error(f"Error loading API key: {str(e)}")
    OPENAQ_API_KEY = "430a6cbeb038741241c9129a3323543b8a15f7e2b80bd32d9d07b2efb3d66aff"

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# Lokasi sensor di Malang
LOCATION = {
    "name": "STT Satyabhakti Malang",
    "latitude": -7.9185093,
    "longitude": 112.651344,
    "radius": 5000
}

# ============= KATEGORI ISPU =============
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
    "Baik": "Kualitas udara baik, tidak berdampak pada kesehatan.",
    "Sedang": "Kualitas udara masih dapat diterima, namun bisa berdampak pada kelompok sensitif.",
    "Tidak Sehat": "Kualitas udara tidak sehat, disarankan mengurangi aktivitas luar ruangan.",
    "Sangat Tidak Sehat": "Kualitas udara sangat tidak sehat, hindari aktivitas luar ruangan.",
}

CATEGORY_ADVICE = {
    "Baik": "✅ Bebas beraktivitas di luar ruangan",
    "Sedang": "⚠️ Kelompok sensitif (anak, lansia) sebaiknya kurangi aktivitas berat",
    "Tidak Sehat": "😷 Gunakan masker saat di luar, kurangi aktivitas luar ruangan",
    "Sangat Tidak Sehat": "🚫 Hindari aktivitas luar ruangan, tutup jendela",
}

st.set_page_config(
    page_title="🌤️ Ramalan Kualitas Udara Malang",
    page_icon="🌤️",
    layout="wide"
)

data_queue = queue.Queue(maxsize=1000)

# ============= OPENAQ V3 HANDLER =============
class OpenAQV3Handler:
    """Handler untuk OpenAQ API v3"""
    
    def __init__(self, api_key, location):
        self.api_key = api_key
        self.location = location
        self.base_url = "https://api.openaq.org/v3"
        self.headers = {"X-API-Key": api_key}
        self.sensor_ids = {}
        self.using_dummy = True
        
    def discover_sensors(self):
        """Cari sensor di lokasi tertentu"""
        try:
            # Cari locations terlebih dahulu
            params = {
                "coordinates": f"{self.location['latitude']},{self.location['longitude']}",
                "radius": self.location["radius"],
                "limit": 10
            }
            
            response = requests.get(
                f"{self.base_url}/locations",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                locations = data.get("results", [])
                
                if locations:
                    location_id = locations[0].get("id")
                    
                    # Cari sensors di location tersebut
                    sensor_params = {
                        "location_id": location_id,
                        "limit": 20
                    }
                    
                    sensor_response = requests.get(
                        f"{self.base_url}/sensors",
                        params=sensor_params,
                        headers=self.headers,
                        timeout=10
                    )
                    
                    if sensor_response.status_code == 200:
                        sensor_data = sensor_response.json()
                        sensors = sensor_data.get("results", [])
                        
                        # Mapping parameter ke sensor_id
                        sensor_mapping = {}
                        for sensor in sensors:
                            param = sensor.get("parameter", {}).get("name", "").lower()
                            if param in ["pm1", "pm25", "relativehumidity", "temperature"]:
                                sensor_mapping[param] = sensor.get("id")
                        
                        if sensor_mapping:
                            self.sensor_ids = sensor_mapping
                            self.using_dummy = False
                            return sensor_mapping
                        
            return {}
            
        except Exception as e:
            st.error(f"Error discovering sensors: {str(e)}")
            return {}
    
    def get_measurements(self, sensor_id, limit=10):
        """Get measurements dari sensor tertentu"""
        try:
            params = {
                "limit": limit,
                "order_by": "datetime",
                "sort": "desc"
            }
            
            response = requests.get(
                f"{self.base_url}/sensors/{sensor_id}/measurements",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                return []
                
        except Exception as e:
            return []
    
    def get_latest_data(self):
        """Get data terbaru dari semua sensor"""
        if self.using_dummy or not self.sensor_ids:
            return self.get_dummy_data()
        
        result = {}
        
        for param, sensor_id in self.sensor_ids.items():
            measurements = self.get_measurements(sensor_id, limit=1)
            if measurements and len(measurements) > 0:
                latest = measurements[0]
                value = latest.get("value")
                if value is not None:
                    result[param] = value
                    result["timestamp"] = latest.get("datetime", {}).get("utc", datetime.now().isoformat())
        
        # Cek apakah semua parameter ada
        required = ["pm1", "relativehumidity", "temperature"]
        if not all(k in result for k in required):
            return self.get_dummy_data()
        
        # Tambahkan um003
        if "pm1" in result:
            result["um003"] = result["pm1"] * 100 + np.random.normal(0, 50)
        
        result["is_dummy"] = False
        return result
    
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


# ============= STREAMING HANDLER =============
class StreamDataHandler:
    """Handler untuk streaming data dari OpenAQ v3"""
    
    def __init__(self, api_key, location, interval=60):
        self.api_key = api_key
        self.location = location
        self.interval = interval
        self.running = False
        self.data_buffer = []
        self.last_data = None
        self.openaq = OpenAQV3Handler(api_key, location)
        
    def discover_sensors(self):
        """Discover sensor IDs"""
        sensor_ids = self.openaq.discover_sensors()
        if sensor_ids:
            st.success(f"✅ Ditemukan {len(sensor_ids)} sensor!")
            for param, sid in sensor_ids.items():
                st.info(f"📡 {param}: {sid}")
        else:
            st.warning("⚠️ Sensor tidak ditemukan. Menggunakan dummy data.")
        return sensor_ids
        
    def fetch_data(self):
        """Fetch data dari OpenAQ API"""
        try:
            if not self.api_key:
                return self.openaq.get_dummy_data()
            
            data = self.openaq.get_latest_data()
            return data
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return self.openaq.get_dummy_data()
    
    def stream_data(self):
        """Main loop untuk streaming data"""
        self.running = True
        self.discover_sensors()
        
        while self.running:
            try:
                data = self.fetch_data()
                if data:
                    if not data_queue.full():
                        data_queue.put(data)
                    self.last_data = data
                    
                    self.data_buffer.append(data)
                    if len(self.data_buffer) > 1000:
                        self.data_buffer = self.data_buffer[-1000:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                st.error(f"Stream error: {str(e)}")
                time.sleep(5)
    
    def start_streaming(self):
        """Start streaming thread"""
        if not self.running:
            thread = Thread(target=self.stream_data, daemon=True)
            thread.start()
            return True
        return False
    
    def stop_streaming(self):
        """Stop streaming"""
        self.running = False


# ============= SPARK & MODEL =============
@st.cache_resource
def load_spark_and_model():
    """Load Spark session dan model"""
    spark = (
        SparkSession.builder.appName("ISPU-Streaming-App")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(MODEL_PATH)
    return spark, model

def predict_batch(spark, model, data):
    """Prediksi batch"""
    pdf = pd.DataFrame(data)
    sdf = spark.createDataFrame(pdf)
    result = model.transform(sdf)
    
    labels = model.stages[0].labelsArray[0]
    pred_idx = result.select("prediction").collect()
    predictions = [labels[int(row["prediction"])] for row in pred_idx]
    return predictions

def predict_single(spark, model, data):
    """Prediksi single data point"""
    return predict_batch(spark, model, [data])[0]

# ============= UI FUNCTIONS =============
def display_current_air_quality(category, data):
    """Display current air quality status"""
    color = CATEGORY_COLOR.get(category, "#95a5a6")
    emoji = CATEGORY_EMOJI.get(category, "🌤️")
    
    is_dummy = data.get("is_dummy", False)
    dummy_text = " (Data Simulasi)" if is_dummy else ""
    
    st.markdown(f"""
    <div style='
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, {color}20, {color}05);
        border-radius: 20px;
        border: 2px solid {color};
    '>
        <div style='font-size: 4rem;'>{emoji}</div>
        <div style='font-size: 2.5rem; font-weight: bold; color: {color};'>
            {category}{dummy_text}
        </div>
        <div style='font-size: 1rem; color: #666; margin: 10px 0;'>
            {CATEGORY_DESC.get(category, "")}
        </div>
        <div style='display: flex; justify-content: center; gap: 20px; font-size: 0.9rem; color: #888;'>
            <span>🌡️ {data.get("temperature", 0):.1f}°C</span>
            <span>💧 {data.get("relativehumidity", 0):.1f}%</span>
            <span>📊 PM1: {data.get("pm1", 0):.1f}</span>
        </div>
        <div style='margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 10px;'>
            💡 {CATEGORY_ADVICE.get(category, "")}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_weather_card(category, date, temp, rh, pm1, um003):
    """Tampilkan kartu ramalan cuaca per hari"""
    color = CATEGORY_COLOR.get(category, "#95a5a6")
    emoji = CATEGORY_EMOJI.get(category, "🌤️")
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {color}25, {color}10);
        border-radius: 15px;
        padding: 15px;
        margin: 5px 0;
        border-left: 5px solid {color};
        text-align: center;
        min-height: 180px;
    '>
        <div style='font-size: 2.5rem;'>{emoji}</div>
        <div style='font-weight: bold; font-size: 0.9rem;'>{date}</div>
        <div style='font-size: 1.3rem; font-weight: bold; color: {color};'>
            {category}
        </div>
        <div style='font-size: 0.8rem; color: #666;'>
            🌡️ {temp:.1f}°C &nbsp;|&nbsp; 💧 {rh:.0f}%
        </div>
        <div style='font-size: 0.75rem; color: #888;'>
            PM1: {pm1:.1f} | Particle: {um003:.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

def generate_weekly_forecast(current_data, history_data=None):
    """Generate forecast 7 hari"""
    dates = [(datetime.now() + timedelta(days=i)).strftime("%a, %d %b") for i in range(7)]
    
    if history_data and len(history_data) > 5:
        pm1_trend = np.mean([d.get("pm1", 0) for d in history_data[-10:]])
        temp_trend = np.mean([d.get("temperature", 0) for d in history_data[-10:]])
        rh_trend = np.mean([d.get("relativehumidity", 0) for d in history_data[-10:]])
        um003_trend = np.mean([d.get("um003", 0) for d in history_data[-10:]])
    else:
        pm1_trend = current_data.get("pm1", 10)
        temp_trend = current_data.get("temperature", 27)
        rh_trend = current_data.get("relativehumidity", 65)
        um003_trend = current_data.get("um003", 300)
    
    forecast_data = []
    for i in range(7):
        day_variation = np.sin(i * np.pi / 3) * 0.3
        
        day_data = {
            "date": dates[i],
            "temperature": max(0, min(45, temp_trend + np.random.normal(0, 1) + day_variation * 2)),
            "relativehumidity": max(0, min(100, rh_trend + np.random.normal(0, 3) - day_variation * 5)),
            "pm1": max(0, pm1_trend + np.random.normal(0, 2) + day_variation * 3),
            "um003": max(0, um003_trend + np.random.normal(0, 50) + day_variation * 100),
        }
        forecast_data.append(day_data)
    
    return forecast_data

def create_forecast_plot(forecast_data, current_data, history_data):
    """Buat plot interaktif"""
    forecast_dates = [d["date"] for d in forecast_data]
    forecast_temps = [d["temperature"] for d in forecast_data]
    forecast_rh = [d["relativehumidity"] for d in forecast_data]
    forecast_pm1 = [d["pm1"] for d in forecast_data]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "🌤️ Prediksi Kualitas Udara 7 Hari",
            "🌡️ Prediksi Suhu & Kelembapan",
            "📊 Prediksi PM1 & Particle Count"
        ),
        vertical_spacing=0.12,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    forecast_categories = [d["category"] for d in forecast_data]
    category_to_num = {"Baik": 1, "Sedang": 2, "Tidak Sehat": 3, "Sangat Tidak Sehat": 4}
    forecast_cat_nums = [category_to_num.get(c, 0) for c in forecast_categories]
    forecast_colors = [CATEGORY_COLOR.get(c, "#95a5a6") for c in forecast_categories]
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_cat_nums,
            mode='lines+markers+text',
            name='Prediksi Kategori',
            text=forecast_categories,
            textposition='top center',
            marker=dict(size=20, color=forecast_colors, symbol='star'),
            line=dict(color='#333', width=2, dash='solid'),
            hovertemplate='%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    if current_data:
        current_cat = current_data.get("category", "Baik")
        current_num = category_to_num.get(current_cat, 1)
        fig.add_trace(
            go.Scatter(
                x=["Sekarang"],
                y=[current_num],
                mode='markers',
                name='Kondisi Saat Ini',
                marker=dict(size=25, color=CATEGORY_COLOR.get(current_cat, "#95a5a6"), symbol='circle'),
                text=[current_cat],
                hovertemplate='Sekarang: %{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_temps,
            mode='lines+markers',
            name='Suhu (°C)',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8, color='#e74c3c')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_rh,
            mode='lines+markers',
            name='Kelembapan (%)',
            line=dict(color='#3498db', width=2, dash='dot'),
            marker=dict(size=8, color='#3498db')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=forecast_dates,
            y=forecast_pm1,
            name='PM1 (µg/m³)',
            marker_color='#27ae60',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16
    )
    
    fig.update_yaxes(title_text="Kategori", row=1, col=1)
    fig.update_yaxes(
        range=[0.5, 4.5],
        row=1, col=1,
        tickvals=[1, 2, 3, 4],
        ticktext=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat']
    )
    fig.update_yaxes(title_text="Nilai", row=2, col=1)
    fig.update_yaxes(title_text="PM1 (µg/m³)", row=3, col=1)
    fig.update_xaxes(title_text="Hari", row=3, col=1)
    
    return fig

# ============= MAIN APP =============
def main():
    st.title("🌤️ Ramalan Kualitas Udara Malang")
    st.caption("Real-time monitoring dan prediksi 7 hari ke depan menggunakan OpenAQ v3")
    
    # Load model
    with st.spinner("Memuat model prediksi..."):
        try:
            spark, model = load_spark_and_model()
            st.success("✅ Model siap digunakan!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Kontrol")
        
        # Sensor discovery
        st.subheader("🔍 Discovery Sensor")
        if st.button("🔎 Cari Sensor Terdekat", use_container_width=True):
            with st.spinner("Mencari sensor di Malang..."):
                handler = StreamDataHandler(OPENAQ_API_KEY, LOCATION, interval=30)
                sensor_ids = handler.discover_sensors()
                if sensor_ids:
                    st.success(f"✅ Ditemukan {len(sensor_ids)} sensor!")
                    for param, sid in sensor_ids.items():
                        st.info(f"📡 {param}: {sid}")
                else:
                    st.warning("⚠️ Tidak ditemukan sensor. Menggunakan dummy data.")
                    st.session_state.using_dummy = True
        
        st.divider()
        
        # Streaming control
        st.subheader("📡 Data Streaming")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Mulai", type="primary", use_container_width=True):
                if 'stream_handler' not in st.session_state:
                    st.session_state.stream_handler = StreamDataHandler(
                        OPENAQ_API_KEY,
                        LOCATION,
                        interval=30
                    )
                    st.session_state.stream_handler.start_streaming()
                    st.success("Streaming dimulai!")
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                if 'stream_handler' in st.session_state:
                    st.session_state.stream_handler.stop_streaming()
                    st.warning("Streaming dihentikan")
                    st.rerun()
        
        st.divider()
        
        # Manual input
        st.subheader("✏️ Input Manual")
        st.caption("Gunakan jika streaming tidak tersedia")
        
        manual_temp = st.slider("Suhu (°C)", 0.0, 45.0, 27.0, 0.5)
        manual_rh = st.slider("Kelembapan (%)", 0.0, 100.0, 65.0, 1.0)
        manual_pm1 = st.slider("PM1 (µg/m³)", 0.0, 100.0, 10.0, 0.5)
        manual_um003 = st.slider("Particle Count", 0.0, 5000.0, 300.0, 50.0)
        
        if st.button("🔄 Gunakan Manual", use_container_width=True):
            st.session_state.current_data = {
                "pm1": manual_pm1,
                "relativehumidity": manual_rh,
                "temperature": manual_temp,
                "um003": manual_um003,
                "timestamp": datetime.now().isoformat(),
                "is_dummy": True
            }
            st.success("Data manual diupdate!")
            st.rerun()
        
        st.divider()
        
        # Status
        st.subheader("📊 Status")
        if 'stream_handler' in st.session_state:
            handler = st.session_state.stream_handler
            if handler.running:
                st.success("🟢 Streaming aktif")
                st.info(f"📦 Buffer: {len(handler.data_buffer)} records")
                if handler.openaq.using_dummy:
                    st.warning("⚠️ Menggunakan dummy data")
            else:
                st.warning("🔴 Streaming tidak aktif")
        else:
            st.info("⏸️ Streaming belum dimulai")
    
    # ============= MAIN CONTENT =============
    current_data = None
    history_data = []
    
    # Get data from streaming
    if 'stream_handler' in st.session_state:
        handler = st.session_state.stream_handler
        
        # Process queue
        while not data_queue.empty():
            data = data_queue.get()
            data["category"] = predict_single(spark, model, data)
            handler.data_buffer.append(data)
            current_data = data
        
        if current_data is None and handler.data_buffer:
            current_data = handler.data_buffer[-1]
        
        history_data = handler.data_buffer[-50:]
    
    # Get data from manual input
    if current_data is None and 'current_data' in st.session_state:
        current_data = st.session_state.current_data
        if "category" not in current_data:
            current_data["category"] = predict_single(spark, model, current_data)
    
    # Default data
    if current_data is None:
        current_data = {
            "pm1": 10.0,
            "relativehumidity": 65.0,
            "temperature": 27.0,
            "um003": 300.0,
            "category": "Baik",
            "timestamp": datetime.now().isoformat(),
            "is_dummy": True
        }
    
    # Display current condition
    st.subheader("📍 Kondisi Kualitas Udara Saat Ini")
    st.caption(f"🕐 Update: {current_data.get('timestamp', 'N/A')}")
    display_current_air_quality(current_data.get("category", "Baik"), current_data)
    
    st.markdown("---")
    
    # Generate forecast
    st.subheader("🔮 Prediksi 7 Hari Ke Depan")
    
    with st.spinner("Menghasilkan ramalan..."):
        forecast_data = generate_weekly_forecast(current_data, history_data)
        
        for day_data in forecast_data:
            day_data["category"] = predict_single(spark, model, day_data)
        
        # Weather cards
        cols = st.columns(7)
        for i, data in enumerate(forecast_data):
            with cols[i]:
                display_weather_card(
                    data["category"],
                    data["date"],
                    data["temperature"],
                    data["relativehumidity"],
                    data["pm1"],
                    data["um003"]
                )
        
        st.markdown("---")
        
        # Plot
        st.subheader("📈 Grafik Tren 7 Hari")
        fig = create_forecast_plot(forecast_data, current_data, history_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Rekomendasi")
        
        categories = [d["category"] for d in forecast_data]
        most_common = max(set(categories), key=categories.count)
        
        cat_order = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat"]
        if len(categories) > 1:
            first_idx = cat_order.index(categories[0])
            last_idx = cat_order.index(categories[-1])
            if last_idx > first_idx:
                trend_text = "📈 **Memburuk** - Kualitas udara diperkirakan menurun"
            elif last_idx < first_idx:
                trend_text = "📉 **Membaik** - Kualitas udara diperkirakan meningkat"
            else:
                trend_text = "➡️ **Stabil** - Kualitas udara relatif stabil"
        else:
            trend_text = "➡️ Data belum cukup untuk menentukan tren"
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"""
            **📊 Prediksi Rata-rata 7 Hari:** Kategori **{most_common}**
            
            **📈 Tren:** {trend_text}
            
            **💡 Saran:** {CATEGORY_ADVICE.get(most_common, "Lakukan aktivitas dengan bijak")}
            
            📝 {CATEGORY_DESC.get(most_common, "")}
            """)
        
        with col2:
            dist_data = pd.DataFrame(categories, columns=["Kategori"])
            st.write("**📊 Distribusi Prediksi:**")
            st.bar_chart(dist_data["Kategori"].value_counts())
    
    # Detail data
    with st.expander("📋 Data Detail"):
        df_forecast = pd.DataFrame(forecast_data)
        df_forecast.columns = ["Tanggal", "Suhu (°C)", "Kelembapan (%)", "PM1", "Particle Count", "Kategori"]
        st.dataframe(df_forecast, use_container_width=True)
        
        csv = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Ramalan (CSV)",
            csv,
            f"ramalan_kualitas_udara_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()