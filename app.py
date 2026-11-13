import os
import io
import re
import json
import math
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import soil_zoning as sz

st.set_page_config(page_title="SoilScout AI", page_icon="🌱", layout="wide")

# ---------------- CUSTOM AGRITECH UI STYLING ----------------
st.markdown(
    """
    <style>
    /* Proper top clearance below Streamlit's top bar */
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064E3B 0%, #1B4332 55%, #2D6A4F 100%);
        padding: 1.45rem 1.8rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-top: 0.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(6, 78, 59, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.14);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .hero-sub {
        font-size: 0.96rem;
        color: #D1FAE5;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    /* Step Section Headers */
    .step-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .step-badge {
        background-color: #1B4332;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        letter-spacing: 0.05em;
    }
    .step-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }

    /* Elevated Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(248, 250, 252, 0.95);
        border: 1px solid #CBD5E1;
        border-left: 4px solid #10B981;
        padding: 12px 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.35rem !important;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.35);
        transform: translateY(-1px);
    }

    /* Download Buttons */
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid #10B981;
    }

    /* Expander Polish */
    div[data-testid="stExpander"] {
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    </style>
    <div class="hero-banner">
        <div class="hero-title">🌱 SoilScout AI — Smart Precision Irrigation</div>
        <p class="hero-sub">Field Moisture Zoning • Live Weather & Rain Advisor • Borewell & Drip Valve Scheduler</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "samples" not in st.session_state:
    st.session_state.samples = []
if "gps_coords" not in st.session_state:
    st.session_state.gps_coords = None
if "last_city_text" not in st.session_state:
    st.session_state.last_city_text = "Bengaluru"
if "saved_logs" not in st.session_state:
    st.session_state.saved_logs = []
if "last_saved_sig" not in st.session_state:
    st.session_state.last_saved_sig = ""

# Capture GPS coordinates returned by browser geolocation button via URL query params
qp = st.query_params
if "lat" in qp and "lon" in qp:
    try:
        st.session_state.gps_coords = (float(qp["lat"]), float(qp["lon"]))
    except ValueError:
        pass


# ---------------- INDIAN SOIL PROFILES ----------------
class IndianSoilAdapter:
    """Wraps a base texture from soil_zoning.py with Indian soil agronomic parameters."""
    def __init__(self, label: str, base_key: str, field_capacity: float, wilting_point: float, desc: str):
        keys = list(sz.TEXTURES.keys())
        chosen_key = base_key if base_key in sz.TEXTURES else keys[0]
        self._base = sz.TEXTURES[chosen_key]
        self.label = label
        self.field_capacity = field_capacity
        self.wilting_point = wilting_point
        self.desc = desc

    def __getattr__(self, name):
        return getattr(self._base, name)


INDIAN_SOILS = {
    "normal_alluvial": IndianSoilAdapter(
        "Alluvial / Normal Farm Soil (Balanced Loam)", "loam", 36.0, 16.0,
        "Balanced drainage and moisture retention; standard fertile Indian farmland."
    ),
    "mixed_soil": IndianSoilAdapter(
        "Mixed Soil (Red & Black Soil Blend)", "clay_loam", 40.0, 19.0,
        "Medium-heavy blend with good nutrient and moisture holding capacity."
    ),
    "red_loam": IndianSoilAdapter(
        "Red Loamy Soil (Kempu Mannu)", "sandy_loam", 31.0, 14.0,
        "Common across Karnataka & Deccan plateau; drains moderately fast."
    ),
    "black_cotton": IndianSoilAdapter(
        "Black Cotton Soil (Regur / Heavy Clay)", "clay", 46.0, 23.0,
        "Deep moisture retention; holds water long, avoid over-watering."
    ),
    "sandy_soil": IndianSoilAdapter(
        "Sandy / Coastal Light Soil", "sand", 22.0, 9.0,
        "Fast water drainage; requires smaller, more frequent irrigation cycles."
    ),
    "clay_lowland": IndianSoilAdapter(
        "Clayey / Silty Lowland Soil", "silt_loam", 42.0, 20.0,
        "Slow drainage and high water storage; ideal for paddy and sugarcane."
    ),
}


# ---------------- INDIAN CROP CATEGORIES & AGRONOMIC PROFILES ----------------
INDIAN_CROPS = {
    "🌾 Cereals & Millets": {
        "Paddy (Rice)": {"root_depth_m": 0.20, "target_fc_ratio": 1.05, "kc": 1.15, "notes": "Shallow 20 cm roots; high evaporation & water demand."},
        "Wheat": {"root_depth_m": 0.35, "target_fc_ratio": 0.80, "kc": 0.90, "notes": "35 cm root zone; moderate moisture, avoid waterlogging."},
        "Maize (Corn)": {"root_depth_m": 0.35, "target_fc_ratio": 0.82, "kc": 0.95, "notes": "35 cm root zone; sensitive to dry spells during flowering."},
        "Ragi (Finger Millet)": {"root_depth_m": 0.25, "target_fc_ratio": 0.70, "kc": 0.75, "notes": "Hardy millet; slow water loss, thrives in lower moisture."},
        "Jowar (Sorghum)": {"root_depth_m": 0.30, "target_fc_ratio": 0.68, "kc": 0.72, "notes": "Drought-tolerant; low irrigation demand."},
        "Bajra (Pearl Millet)": {"root_depth_m": 0.30, "target_fc_ratio": 0.65, "kc": 0.70, "notes": "Very low water need; ideal for dryland conditions."},
    },
    "🍅 Vegetables": {
        "Tomato": {"root_depth_m": 0.25, "target_fc_ratio": 0.88, "kc": 0.98, "notes": "25 cm roots; steady moisture prevents fruit cracking."},
        "Potato": {"root_depth_m": 0.20, "target_fc_ratio": 0.90, "kc": 0.95, "notes": "Shallow 20 cm roots; consistent moisture needed for tubers."},
        "Brinjal (Eggplant)": {"root_depth_m": 0.25, "target_fc_ratio": 0.85, "kc": 0.92, "notes": "25 cm roots; responds well to regular drip irrigation."},
        "Lady Finger (Bhindi)": {"root_depth_m": 0.25, "target_fc_ratio": 0.82, "kc": 0.90, "notes": "25 cm roots; moderate-high moisture during pod formation."},
        "Onion": {"root_depth_m": 0.15, "target_fc_ratio": 0.88, "kc": 0.85, "notes": "Very shallow 15 cm roots; dries fast, needs frequent watering."},
        "Chilli / Capsicum": {"root_depth_m": 0.20, "target_fc_ratio": 0.85, "kc": 0.90, "notes": "20 cm roots; sensitive to both drought and excess water."},
        "Cabbage / Cauliflower": {"root_depth_m": 0.20, "target_fc_ratio": 0.90, "kc": 0.95, "notes": "Shallow 20 cm roots; high moisture for tight head growth."},
    },
    "🫘 Pulses & Oilseeds": {
        "Groundnut (Peanut)": {"root_depth_m": 0.30, "target_fc_ratio": 0.76, "kc": 0.80, "notes": "30 cm roots; prefers well-drained sandy/loamy soil."},
        "Tur / Arhar (Pigeon Pea)": {"root_depth_m": 0.40, "target_fc_ratio": 0.72, "kc": 0.75, "notes": "Deep 40 cm taproot; low water requirement."},
        "Chana (Chickpea)": {"root_depth_m": 0.35, "target_fc_ratio": 0.70, "kc": 0.70, "notes": "35 cm roots; minimal irrigation needed."},
        "Soybean": {"root_depth_m": 0.30, "target_fc_ratio": 0.78, "kc": 0.85, "notes": "30 cm roots; moderate moisture during pod filling."},
        "Sunflower": {"root_depth_m": 0.40, "target_fc_ratio": 0.75, "kc": 0.82, "notes": "40 cm roots; drought-hardy with moderate water needs."},
    },
    "🌿 Cash & Commercial Crops": {
        "Sugarcane": {"root_depth_m": 0.50, "target_fc_ratio": 0.88, "kc": 1.12, "notes": "Deep 50 cm root zone; high daily transpiration."},
        "Cotton": {"root_depth_m": 0.45, "target_fc_ratio": 0.78, "kc": 0.92, "notes": "45 cm taproot; avoid over-watering during boll opening."},
        "Turmeric / Ginger": {"root_depth_m": 0.25, "target_fc_ratio": 0.88, "kc": 0.95, "notes": "25 cm rhizome zone; needs consistently moist soil."},
    },
    "🌴 Fruits & Plantation": {
        "Banana": {"root_depth_m": 0.35, "target_fc_ratio": 0.90, "kc": 1.10, "notes": "35 cm roots; broad leaves cause high daily water loss."},
        "Coconut": {"root_depth_m": 0.65, "target_fc_ratio": 0.78, "kc": 0.88, "notes": "Deep 65 cm root basin; steady moderate moisture."},
        "Arecanut": {"root_depth_m": 0.55, "target_fc_ratio": 0.82, "kc": 0.92, "notes": "55 cm root zone; sensitive to prolonged dry spells."},
        "Mango": {"root_depth_m": 0.70, "target_fc_ratio": 0.72, "kc": 0.78, "notes": "Deep 70 cm roots; low irrigation once established."},
        "Coffee / Black Pepper": {"root_depth_m": 0.45, "target_fc_ratio": 0.82, "kc": 0.88, "notes": "45 cm roots; prefers cool, evenly moist soil."},
    },
}


# ---------------- HELPER FUNCTIONS ----------------
def step_header(badge_text: str, title_text: str):
    st.markdown(
        f"""
        <div class="step-header">
            <span class="step-badge">{badge_text}</span>
            <h3 class="step-title">{title_text}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def auto_save_samples(samples_list: list, source_tag: str, crop_name: str):
    """Automatically saves cleaned readings to CSV & TXT formats without prompting the user."""
    if not samples_list:
        return

    df = pd.DataFrame([
        {
            "spot": i + 1,
            "x_m": round(s.x, 2),
            "y_m": round(s.y, 2),
            "moisture_pct": round(s.moisture, 2),
            "temp_c": round(s.temp_c, 2),
            "humidity_pct": round(s.rh, 2),
        }
        for i, s in enumerate(samples_list)
    ])
    csv_str = df.to_csv(index=False)
    txt_lines = [
        f"{int(r['spot'])},{r['moisture_pct']},{r['temp_c']},{r['humidity_pct']}"
        for _, r in df.iterrows()
    ]
    txt_str = "\n".join(txt_lines) + "\n"

    sig = hashlib.md5(csv_str.encode("utf-8")).hexdigest()
    if sig == st.session_state.last_saved_sig:
        return
    st.session_state.last_saved_sig = sig

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    readable_ts = datetime.now().strftime("%d %b %Y, %I:%M %p")
    safe_crop = re.sub(r"[^a-zA-Z0-9]+", "_", crop_name).strip("_")
    base_name = f"soilscout_{safe_crop}_{ts}"

    # Write local backup files automatically (works locally & in writable cloud temp/workspace)
    try:
        os.makedirs("saved_field_logs", exist_ok=True)
        with open(os.path.join("saved_field_logs", f"{base_name}.csv"), "w", encoding="utf-8") as f_csv:
            f_csv.write(csv_str)
        with open(os.path.join("saved_field_logs", f"{base_name}.txt"), "w", encoding="utf-8") as f_txt:
            f_txt.write(txt_str)
    except Exception:
        pass

    st.session_state.saved_logs.insert(0, {
        "title": f"{crop_name} • {len(samples_list)} spots ({source_tag}) — {readable_ts}",
        "base_name": base_name,
        "csv_data": csv_str,
        "txt_data": txt_str,
        "samples": list(samples_list),
    })


def recommended_spot_count(length_m: float, width_m: float) -> int:
    hectares = (length_m * width_m) / 10000.0
    spots = int(round(6.5 + 3.4 * math.sqrt(hectares)))
    return max(5, min(35, spots))


def generate_serpentine_points(length_m: float, width_m: float, n_spots: int) -> np.ndarray:
    n = max(3, min(35, int(n_spots)))
    aspect = width_m / max(length_m, 1.0)
    n_rows = max(2, min(n, int(round(math.sqrt(n * aspect)))))
    base_cols, extra = divmod(n, n_rows)
    points = []
    for r in range(n_rows):
        cols_in_row = base_cols + (1 if r < extra else 0)
        y = width_m * (r + 0.5) / n_rows
        row_xs = [length_m * (c + 0.5) / cols_in_row for c in range(cols_in_row)]
        if r % 2 == 1:
            row_xs.reverse()
        for x in row_xs:
            points.append([round(x, 1), round(y, 1)])
    return np.array(points, dtype=float)


def format_runtime(liters: float, flow_lpm: float) -> str:
    if liters <= 0 or flow_lpm <= 0:
        return "0 mins (Skip)"
    total_mins = int(round(liters / flow_lpm))
    if total_mins < 1:
        return "< 1 min"
    hrs, mins = divmod(total_mins, 60)
    if hrs == 0:
        return f"{mins} mins"
    if mins == 0:
        return f"{hrs} hr" if hrs == 1 else f"{hrs} hrs"
    return f"{hrs} hr {mins} mins"


@st.cache_data(ttl=1800, show_spinner=False)
def reverse_geocode_coords(lat: float, lon: float) -> str:
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat:.5f}&lon={lon:.5f}&zoom=13"
        req = urllib.request.Request(url, headers={"User-Agent": "SoilScoutAI/1.0 (precision-ag)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        addr = data.get("address", {})
        place = (
            addr.get("village")
            or addr.get("town")
            or addr.get("suburb")
            or addr.get("city")
            or addr.get("county")
            or "GPS Location"
        )
        state = addr.get("state", "")
        return f"{place}, {state}".strip(", ")
    except Exception:
        return f"GPS ({lat:.3f}°N, {lon:.3f}°E)"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_by_coords(lat: float, lon: float, label: str):
    try:
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat:.5f}&longitude={lon:.5f}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code"
            f"&daily=precipitation_sum,precipitation_probability_max,temperature_2m_max"
            f"&timezone=auto&forecast_days=2"
        )
        req_wx = urllib.request.Request(wx_url, headers={"User-Agent": "SoilScoutAI/1.0"})
        with urllib.request.urlopen(req_wx, timeout=5) as resp:
            wx = json.loads(resp.read().decode("utf-8"))

        cur = wx.get("current", {})
        daily = wx.get("daily", {})

        temp_now = float(cur.get("temperature_2m", 29.0))
        rh_now = float(cur.get("relative_humidity_2m", 55.0))
        rain_mm = float(daily.get("precipitation_sum", [0.0])[0] or 0.0)
        rain_prob = int(daily.get("precipitation_probability_max", [0])[0] or 0)
        temp_max = float(daily.get("temperature_2m_max", [temp_now])[0] or temp_now)

        return {
            "location": label,
            "temp_now": temp_now,
            "rh_now": rh_now,
            "rain_mm": rain_mm,
            "rain_prob": rain_prob,
            "temp_max": temp_max,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(city_name: str):
    try:
        q = urllib.parse.quote(city_name.strip() or "Bengaluru")
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
        req_geo = urllib.request.Request(geo_url, headers={"User-Agent": "SoilScoutAI/1.0"})
        with urllib.request.urlopen(req_geo, timeout=5) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))

        if not geo_data.get("results"):
            return None

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = f"{loc.get('name', city_name)}, {loc.get('admin1', loc.get('country', ''))}".strip(", ")
        return fetch_weather_by_coords(lat, lon, resolved_name)
    except Exception:
        return None


def evaluate_weather_advice(wx: dict | None):
    if not wx:
        return 1.0, "info", "Offline Weather Mode", "Using sensor temperature/humidity only (100% baseline irrigation)."

    rain_prob = wx["rain_prob"]
    rain_mm = wx["rain_mm"]
    temp_max = wx["temp_max"]

    if rain_prob >= 60 or rain_mm >= 5.0:
        return (
            0.0,
            "rain",
            f"🌧️ Rain Forecasted ({rain_prob}% chance, ~{rain_mm:.1f} mm) — DO NOT WATER NOW",
            "Natural rainfall is expected within 24 hours. Hold irrigation to save water, electricity, and prevent waterlogging."
        )
    elif rain_prob >= 35 or rain_mm >= 1.5:
        return (
            0.60,
            "light_rain",
            f"🌦️ Light Showers Possible ({rain_prob}% chance, ~{rain_mm:.1f} mm) — Reduced by 40%",
            "Partial rainfall expected. Irrigation volume has been reduced by 40% to avoid over-saturating roots."
        )
    elif temp_max >= 34.0 or wx["temp_now"] >= 33.0:
        return (
            1.15,
            "hot",
            f"☀️ High Heat & Strong Sun (Peak {temp_max:.1f}°C) — +15% Evaporation Buffer Added",
            "Hot, sunny conditions will increase soil evaporation. Added 15% extra water to keep roots hydrated."
        )
    else:
        return (
            1.0,
            "clear",
            f"🌤️ Clear & Mild Weather ({wx['temp_now']:.1f}°C, {rain_prob}% rain chance) — Normal Schedule",
            "Standard irrigation schedule applies based on measured soil moisture deficit."
        )


def estimate_evapotranspiration_and_schedule(temp_c: float, rh: float, mean_moisture: float,
                                           texture, crop_cfg: dict, wx_data: dict | None):
    fc = getattr(texture, "field_capacity", 38.0)
    wp = getattr(texture, "wilting_point", 18.0)
    root_depth = crop_cfg["root_depth_m"]
    kc = crop_cfg.get("kc", 0.90)
    target_m = crop_cfg["target_fc_ratio"] * fc
    warn_thresh = wp + 0.72 * max(1.0, (target_m - wp))

    t_eff = wx_data["temp_max"] * 0.6 + wx_data["temp_now"] * 0.4 if wx_data else temp_c
    rh_eff = wx_data["rh_now"] if wx_data else rh
    es = 0.6108 * math.exp((17.27 * t_eff) / (t_eff + 237.3))
    ea = es * (max(10.0, min(100.0, rh_eff)) / 100.0)
    vpd = max(0.2, es - ea)

    et0_mm = max(1.5, (0.14 * t_eff) + (1.35 * vpd) - 0.8)
    etc_mm = et0_mm * kc
    daily_loss_pct = max(0.25, etc_mm / (root_depth * 10.0))

    rain_bonus_mm = wx_data["rain_mm"] if (wx_data and wx_data["rain_prob"] >= 35) else 0.0
    rain_bonus_pct = rain_bonus_mm / (root_depth * 10.0)

    reserve_pct = (mean_moisture + rain_bonus_pct) - warn_thresh
    days_until_trigger = reserve_pct / daily_loss_pct
    cycle_days = max(1, int(round((target_m - warn_thresh) / daily_loss_pct)))

    today = datetime.now()
    if days_until_trigger <= 0.6:
        if rain_bonus_mm >= 5.0:
            headline = "Hold Today (Rain Expected)"
            subtext = f"Next check in {cycle_days} days ({(today + timedelta(days=cycle_days)).strftime('%b %d')})"
            report_line = f"Hold for Rain  |  Next cycle ~{cycle_days} days ({(today + timedelta(days=cycle_days)).strftime('%d %b')})"
        else:
            headline = "Water Today (Now)"
            subtext = f"After watering, next cycle in {cycle_days} days ({(today + timedelta(days=cycle_days)).strftime('%b %d')})"
            report_line = f"Water Today  |  Next cycle in {cycle_days} days ({(today + timedelta(days=cycle_days)).strftime('%d %b')})"
    else:
        wait_days = max(1, int(round(days_until_trigger)))
        next_date = (today + timedelta(days=wait_days)).strftime("%b %d, %Y")
        next_short = (today + timedelta(days=wait_days)).strftime("%d %b")
        headline = f"In {wait_days} Day{'s' if wait_days > 1 else ''}"
        subtext = f"Around {next_date} (Every ~{cycle_days} days)"
        report_line = f"In {wait_days} Day{'s' if wait_days > 1 else ''} ({next_short})  |  Cycle: every {cycle_days}d"

    return {
        "et0_mm": et0_mm,
        "etc_mm": etc_mm,
        "daily_loss_pct": daily_loss_pct,
        "vpd": vpd,
        "cycle_days": cycle_days,
        "headline": headline,
        "subtext": subtext,
        "report_line": report_line,
    }


def parse_bluetooth_log(raw_input: bytes | str, suggested_pts: np.ndarray, default_temp: float, default_rh: float):
    """Smart parser for both pasted text and uploaded files from Serial Bluetooth Terminal."""
    text = raw_input.decode("utf-8", errors="ignore") if isinstance(raw_input, bytes) else str(raw_input)
    lines = text.splitlines()
    samples = []

    ts_pattern = re.compile(r"^\s*(?:\d{4}[-/]\d{2}[-/]\d{2}\s+)?\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*[-:>]*\s*")
    num_pattern = re.compile(r"[-+]?\d*\.?\d+")

    for raw_line in lines:
        cleaned = ts_pattern.sub("", raw_line).strip()
        if not cleaned:
            continue

        lower = cleaned.lower()
        if any(word in lower for word in ["moisture", "connect", "disconnect", "pong", "ok,", "err,", "bluetooth", "terminal"]):
            continue

        nums = [float(x) for x in num_pattern.findall(cleaned)]
        if len(nums) < 1:
            continue

        idx = len(samples)
        if len(nums) >= 5:
            px, py, moist, t_c, rh = nums[0], nums[1], nums[2], nums[3], nums[4]
        elif len(nums) == 4:
            spot_num = int(round(nums[0])) - 1
            spot_idx = max(0, min(len(suggested_pts) - 1, spot_num if spot_num >= 0 else idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[1], nums[2], nums[3]
        elif len(nums) == 3:
            if nums[0] == float(idx + 1) and nums[0] <= 50:
                spot_idx = max(0, min(len(suggested_pts) - 1, idx))
                px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
                moist, t_c, rh = nums[1], nums[2], default_rh
            else:
                spot_idx = max(0, min(len(suggested_pts) - 1, idx))
                px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
                moist, t_c, rh = nums[0], nums[1], nums[2]
        elif len(nums) == 2:
            spot_idx = max(0, min(len(suggested_pts) - 1, idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[1], default_temp, default_rh
        else:
            spot_idx = max(0, min(len(suggested_pts) - 1, idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[0], default_temp, default_rh

        moist = max(0.0, min(100.0, moist))
        t_c = max(-10.0, min(65.0, t_c))
        rh = max(0.0, min(100.0, rh))

        samples.append(sz.Sample(x=px, y=py, moisture=moist, temp_c=t_c, rh=rh, source="csv"))

    return samples


def draw_walking_map(length_m: float, width_m: float, points: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F1F8F4")

    rect = plt.Rectangle(
        (0, 0), length_m, width_m,
        fill=True, facecolor="#E6F4EA", edgecolor="#1B4332", linewidth=2.5, zorder=1
    )
    ax.add_patch(rect)
    ax.grid(True, linestyle="--", alpha=0.35, color="#64748B", zorder=2)

    xs, ys = points[:, 0], points[:, 1]
    ax.plot(xs, ys, linestyle="--", color="#D97706", linewidth=1.8, zorder=3)
    for i in range(len(points) - 1):
        ax.annotate(
            "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.4, shrinkA=8, shrinkB=8),
            zorder=3
        )

    marker_size = 220 if len(points) > 20 else 300
    font_size = 7.5 if len(points) > 20 else 8.5
    ax.scatter(xs, ys, s=marker_size, color="#1B4332", edgecolors="white", linewidth=1.8, zorder=4)
    for i, (x, y) in enumerate(points):
        ax.text(x, y, str(i + 1), color="white", fontsize=font_size, fontweight="bold",
                ha="center", va="center", zorder=5)

    ax.scatter([0], [0], s=200, marker="s", color="#DC2626", edgecolors="white", linewidth=1.5, zorder=6)
    ax.text(length_m * 0.015, -width_m * 0.065, "🚩 START CORNER (0, 0)",
            color="#DC2626", fontsize=9.5, fontweight="bold")

    pad_x = length_m * 0.06
    pad_y = width_m * 0.09
    ax.set_xlim(-pad_x, length_m + pad_x)
    ax.set_ylim(-pad_y, width_m + pad_y)
    ax.set_xlabel("Field Length X (meters)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Field Width Y (meters)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title(
        f"Field Sampling Guide  •  {length_m:,.0f}m × {width_m:,.0f}m  ({len(points)} Checkpoints)",
        fontsize=12.5, fontweight="bold", color="#0F172A", pad=12
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def extract_zone_summary(moisture: np.ndarray, cell_area_m2: float, total_area_m2: float,
                         texture, crop_cfg: dict, wx_multiplier: float, pump_lpm: float, drip_lpm: float, pump_kw: float):
    fc = getattr(texture, "field_capacity", 38.0)
    wp = getattr(texture, "wilting_point", 18.0)
    root_depth = crop_cfg["root_depth_m"]
    target_m = crop_cfg["target_fc_ratio"] * fc

    crit_thresh = wp + 0.30 * max(1.0, (target_m - wp))
    warn_thresh = wp + 0.72 * max(1.0, (target_m - wp))
    over_thresh = max(fc + 4.0, target_m + 3.0)

    tiers = [
        ("Critical Dry", "#DC2626", moisture < crit_thresh),
        ("Needs Water", "#F59E0B", (moisture >= crit_thresh) & (moisture < warn_thresh)),
        ("Optimal Zone", "#10B981", (moisture >= warn_thresh) & (moisture <= over_thresh)),
        ("Over-Watered", "#2563EB", moisture > over_thresh),
    ]

    total_cells = moisture.size
    summary = []
    max_deficit_pct = 0.0

    for label, color, mask in tiers:
        count = int(np.sum(mask))
        pct_area = 100.0 * count / total_cells
        area_m2 = count * cell_area_m2
        avg_m = float(np.mean(moisture[mask])) if count > 0 else 0.0
        deficit_pct = max(0.0, target_m - avg_m) if count > 0 else 0.0
        if count > 0:
            max_deficit_pct = max(max_deficit_pct, deficit_pct)

        base_liters = area_m2 * root_depth * (deficit_pct / 100.0) * 1000.0 / 0.85
        adj_liters = base_liters * wx_multiplier
        summary.append({
            "label": label,
            "color": color,
            "pct_area": pct_area,
            "area_m2": area_m2,
            "avg_moisture": avg_m,
            "base_liters": base_liters,
            "liters": adj_liters,
            "pump_time": format_runtime(adj_liters, pump_lpm),
            "drip_time": format_runtime(adj_liters, drip_lpm),
        })

    precision_liters = sum(r["liters"] for r in summary)
    if max_deficit_pct > 0:
        blind_base_liters = total_area_m2 * root_depth * (max_deficit_pct / 100.0) * 1000.0 / 0.85
    else:
        blind_base_liters = total_area_m2 * 0.025 * 1000.0

    saved_liters = max(0.0, blind_base_liters - precision_liters)
    saved_pct = (100.0 * saved_liters / blind_base_liters) if blind_base_liters > 0 else 0.0
    saved_pump_hrs = saved_liters / (max(pump_lpm, 1.0) * 60.0)
    saved_kwh = saved_pump_hrs * pump_kw
    saved_time_str = format_runtime(saved_liters, pump_lpm)

    roi = {
        "blind_liters": blind_base_liters,
        "precision_liters": precision_liters,
        "saved_liters": saved_liters,
        "saved_pct": saved_pct,
        "saved_kwh": saved_kwh,
        "saved_time_str": saved_time_str,
    }
    return summary, wp, fc, target_m, crit_thresh, warn_thresh, over_thresh, roi


def draw_premium_zoning_report(length_m, width_m, xc, yc, moisture, samples, info, texture,
                               crop_name, crop_cfg, wx_multiplier, wx_headline, pump_lpm, drip_lpm, pump_kw, et_info):
    cell_area = (xc[1] - xc[0]) * (yc[1] - yc[0]) if len(xc) > 1 and len(yc) > 1 else 1.0
    total_area_m2 = length_m * width_m
    zone_rows, wp, fc, target_m, t1, t2, t3, roi = extract_zone_summary(
        moisture, cell_area, total_area_m2, texture, crop_cfg, wx_multiplier, pump_lpm, drip_lpm, pump_kw
    )
    total_liters = roi["precision_liters"]
    equiv_mm = (total_liters / total_area_m2) if total_area_m2 > 0 else 0.0

    pump_time_str = format_runtime(total_liters, pump_lpm)
    drip_time_str = format_runtime(total_liters, drip_lpm)

    fig = plt.figure(figsize=(16.0, 9.8), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")

    gs = fig.add_gridspec(2, 2, width_ratios=[1.52, 1.12], height_ratios=[0.82, 1.28],
                          wspace=0.16, hspace=0.25, left=0.05, right=0.975, top=0.86, bottom=0.07)

    fig.text(0.05, 0.948, "SOILSCOUT AI  |  ZONE-BY-ZONE IRRIGATION & RESOURCE SAVINGS REPORT",
             fontsize=15, fontweight="bold", color="#0F172A")
    fig.text(
        0.05, 0.918,
        f"Field: {length_m:,.0f}m × {width_m:,.0f}m ({total_area_m2/10000:,.2f} ha)   •   "
        f"Soil: {texture.label}   •   Crop: {crop_name} (Root {int(crop_cfg['root_depth_m']*100)} cm)",
        fontsize=10.0, color="#1E293B", fontweight="bold"
    )
    fig.text(
        0.05, 0.892,
        f"Sensor Avg: {info['temp']:.1f}°C, {info['rh']:.0f}% RH   •   {wx_headline}",
        fontsize=9.5, color="#475569", fontweight="bold"
    )

    # ---------------- LEFT: MAIN SPATIAL MOISTURE & ZONE MAP ----------------
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor("#FFFFFF")

    cmap = LinearSegmentedColormap.from_list(
        "soil_premium",
        ["#B91C1C", "#EF4444", "#F59E0B", "#84CC16", "#10B981", "#0EA5E9", "#1D4ED8"],
        N=256
    )

    vmin = max(0.0, min(float(moisture.min()) - 2.0, wp - 3.0))
    vmax = min(100.0, max(float(moisture.max()) + 2.0, fc + 8.0))
    levels = np.linspace(vmin, vmax, 36)

    cf = ax_map.contourf(xc, yc, moisture, levels=levels, cmap=cmap, extend="both", antialiased=True)

    valid_bounds = [b for b in [t1, t2, t3] if vmin < b < vmax]
    if valid_bounds:
        cs = ax_map.contour(xc, yc, moisture, levels=valid_bounds, colors="#0F172A",
                            linewidths=1.8, linestyles="--", alpha=0.85)
        ax_map.clabel(cs, inline=True, fontsize=8.5, fmt="%.0f%%")

    ax_map.grid(True, linestyle=":", alpha=0.35, color="#0F172A")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    dense = len(samples) > 18
    badge_font = 6.8 if dense else 8.0
    dot_size = 70 if dense else 95
    for i, s in enumerate(samples):
        ax_map.scatter([s.x], [s.y], s=dot_size, color="#0F172A", edgecolors="white", linewidth=1.5, zorder=6)
        label_txt = f"{i+1}:{s.moisture:.0f}%" if dense else f"S{i+1}: {s.moisture:.0f}%"
        ax_map.annotate(
            label_txt,
            xy=(s.x, s.y), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=badge_font, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#94A3B8", alpha=0.90),
            zorder=7
        )

    ax_map.scatter([0], [0], s=140, marker="s", color="#DC2626", edgecolors="white", linewidth=1.5, zorder=8, clip_on=False)
    ax_map.annotate("🚩 Start (0,0)", xy=(0, 0), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold", color="#DC2626",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#FCA5A5", alpha=0.92), zorder=9)

    legend_handles = [
        Patch(facecolor="#DC2626", label=f"Critical Dry (<{t1:.0f}%)"),
        Patch(facecolor="#F59E0B", label=f"Needs Water ({t1:.0f}–{t2:.0f}%)"),
        Patch(facecolor="#10B981", label=f"Optimal ({t2:.0f}–{t3:.0f}%)"),
        Patch(facecolor="#2563EB", label=f"Over-Watered (>{t3:.0f}%)"),
    ]
    ax_map.legend(handles=legend_handles, loc="upper right", fontsize=7.8, framealpha=0.92, edgecolor="#CBD5E1")

    ax_map.set_xlim(0, length_m)
    ax_map.set_ylim(0, width_m)
    ax_map.set_xlabel("Field Length X (meters)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax_map.set_ylabel("Field Width Y (meters)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax_map.set_title(f"Spatial Soil Moisture & {crop_name} Zone Divisions",
                     fontsize=12, fontweight="bold", color="#0F172A", pad=10)

    cbar = fig.colorbar(cf, ax=ax_map, orientation="horizontal", fraction=0.046, pad=0.11, aspect=35)
    cbar.set_label(f"Volumetric Soil Moisture (%)  •  Ideal Target for {crop_name}: {target_m:.0f}%",
                   fontsize=9.2, fontweight="bold", color="#1E293B")
    cbar.outline.set_edgecolor("#CBD5E1")
    if vmin <= target_m <= vmax:
        cbar.ax.axvline(target_m, color="#0F172A", linewidth=2.2)

    # ---------------- TOP RIGHT: ZONE AREA & SAVINGS BANNER ----------------
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor("#FFFFFF")
    labels = [r["label"] for r in zone_rows][::-1]
    pcts = [r["pct_area"] for r in zone_rows][::-1]
    colors = [r["color"] for r in zone_rows][::-1]

    bars = ax_bar.barh(labels, pcts, color=colors, height=0.55, edgecolor="none")
    ax_bar.set_xlim(0, 118)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")
    ax_bar.set_xlabel("Share of Total Field Area (%)", fontsize=9.2, fontweight="bold", color="#334155")
    ax_bar.set_title(
        f"Zone Coverage  •  Saved {roi['saved_pct']:.0f}% Water ({roi['saved_liters']:,.0f} L) & {roi['saved_kwh']:.1f} kWh",
        fontsize=10.5, fontweight="bold", color="#047857", pad=10
    )

    for bar, pct in zip(bars, pcts):
        ax_bar.text(pct + 2.0, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                    va="center", ha="left", fontsize=9.2, fontweight="bold", color="#1E293B")

    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#CBD5E1")
    ax_bar.spines["bottom"].set_color("#CBD5E1")

    # ---------------- BOTTOM RIGHT: ZONE-BY-ZONE VALVE RUNTIME & SCHEDULE ----------------
    ax_card = fig.add_subplot(gs[1, 1])
    ax_card.set_facecolor("#FFFFFF")
    ax_card.set_xticks([])
    ax_card.set_yticks([])
    for spine in ax_card.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    ax_card.text(0.04, 0.93, f"Zone-by-Zone Water & Valve Runtime ({crop_name})",
                 fontsize=10.8, fontweight="bold", color="#0F172A", transform=ax_card.transAxes)

    ax_card.text(0.04, 0.85, "ZONE (AREA)", fontsize=8.0, fontweight="bold", color="#64748B", transform=ax_card.transAxes)
    ax_card.text(0.54, 0.85, "WATER", fontsize=8.0, fontweight="bold", color="#64748B", ha="right", transform=ax_card.transAxes)
    ax_card.text(0.75, 0.85, "PUMP", fontsize=8.0, fontweight="bold", color="#64748B", ha="right", transform=ax_card.transAxes)
    ax_card.text(0.96, 0.85, "DRIP", fontsize=8.0, fontweight="bold", color="#64748B", ha="right", transform=ax_card.transAxes)

    y_pos = 0.76
    for r in zone_rows:
        ax_card.scatter([0.055], [y_pos + 0.012], s=70, color=r["color"], transform=ax_card.transAxes)
        ax_card.text(0.095, y_pos, f"{r['label']} ({r['pct_area']:.0f}%)",
                     fontsize=8.6, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
        water_txt = f"{r['liters']:,.0f} L" if r["liters"] > 0 else "0 L"
        p_txt = r["pump_time"].replace(" (Skip)", "") if r["liters"] > 0 else "Skip"
        d_txt = r["drip_time"].replace(" (Skip)", "") if r["liters"] > 0 else "Skip"

        ax_card.text(0.54, y_pos, water_txt, fontsize=8.5, fontweight="bold", color="#0F172A", ha="right", transform=ax_card.transAxes)
        ax_card.text(0.75, y_pos, p_txt, fontsize=8.5, fontweight="bold", color="#1D4ED8", ha="right", transform=ax_card.transAxes)
        ax_card.text(0.96, y_pos, d_txt, fontsize=8.5, fontweight="bold", color="#0369A1", ha="right", transform=ax_card.transAxes)
        y_pos -= 0.095

    ax_card.plot([0.04, 0.96], [0.36, 0.36], color="#E2E8F0", linewidth=1.2, transform=ax_card.transAxes)

    ax_card.text(0.04, 0.295, f"TOTAL PRECISION WATER ({equiv_mm:.1f} mm):",
                 fontsize=8.6, fontweight="bold", color="#475569", transform=ax_card.transAxes)
    ax_card.text(0.96, 0.295, f"{total_liters:,.0f} L  (Pump: {pump_time_str} | Drip: {drip_time_str})",
                 fontsize=8.6, fontweight="bold", color="#047857", ha="right", transform=ax_card.transAxes)

    ax_card.text(0.04, 0.22, "💰 SAVED VS. FULL-FIELD WATERING:",
                 fontsize=8.6, fontweight="bold", color="#065F46", transform=ax_card.transAxes)
    ax_card.text(0.96, 0.22, f"{roi['saved_liters']:,.0f} L ({roi['saved_pct']:.0f}%)  •  {roi['saved_kwh']:.1f} kWh ({roi['saved_time_str']})",
                 fontsize=8.6, fontweight="bold", color="#047857", ha="right", transform=ax_card.transAxes)

    ax_card.plot([0.04, 0.96], [0.16, 0.16], color="#E2E8F0", linewidth=1.2, transform=ax_card.transAxes)

    ax_card.text(0.04, 0.095, "☀️ Evaporation Rate:",
                 fontsize=8.6, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
    ax_card.text(0.96, 0.095, f"{et_info['etc_mm']:.1f} mm/day (-{et_info['daily_loss_pct']:.1f}%/day)",
                 fontsize=8.6, fontweight="bold", color="#B45309", ha="right", transform=ax_card.transAxes)

    ax_card.text(0.04, 0.03, "📅 Next Watering:",
                 fontsize=8.6, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
    ax_card.text(0.96, 0.03, et_info["report_line"],
                 fontsize=8.4, fontweight="bold", color="#0F766E", ha="right", transform=ax_card.transAxes)

    return fig, total_liters, equiv_mm, zone_rows, pump_time_str, drip_time_str, roi


# ---------------- COLLAPSIBLE DROPDOWN: ENTER FIELD, CROP & PUMP DETAILS ----------------
with st.expander("⚙️ Enter Field, Soil, Crop & Pump Details (Tap to Open / Close)", expanded=False):
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("**1. Farm Location & Dimensions**")
        city_input = st.text_input("Farm Location (City / Town)", value="Bengaluru")
        if city_input != st.session_state.last_city_text:
            st.session_state.last_city_text = city_input
            st.session_state.gps_coords = None
            if "lat" in st.query_params:
                st.query_params.clear()

        components.html(
            """
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2px 0 6px 0;">
              <button id="gps-btn" onclick="detectGPS()" style="
                width: 100%; padding: 8px 12px; background-color: #1B4332; color: white;
                border: none; border-radius: 6px; font-size: 13.5px; font-weight: 600;
                cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;
              ">📍 Detect My GPS Location</button>
              <div id="gps-status" style="font-size: 11.5px; color: #475569; margin-top: 4px;"></div>
            </div>
            <script>
              function detectGPS() {
                const btn = document.getElementById('gps-btn');
                const status = document.getElementById('gps-status');
                if (!navigator.geolocation) {
                  status.innerText = 'GPS not supported on this browser.';
                  return;
                }
                btn.innerText = '📡 Detecting GPS...';
                btn.disabled = true;
                navigator.geolocation.getCurrentPosition(
                  function(pos) {
                    const lat = pos.coords.latitude.toFixed(5);
                    const lon = pos.coords.longitude.toFixed(5);
                    status.innerText = 'Found: ' + lat + ', ' + lon + ' — Updating weather...';
                    try {
                      const url = new URL(window.parent.location.href);
                      url.searchParams.set('lat', lat);
                      url.searchParams.set('lon', lon);
                      window.parent.location.href = url.toString();
                    } catch (e) {
                      const url = new URL(window.top.location.href);
                      url.searchParams.set('lat', lat);
                      url.searchParams.set('lon', lon);
                      window.top.location.href = url.toString();
                    }
                  },
                  function(err) {
                    btn.innerText = '📍 Detect My GPS Location';
                    btn.disabled = false;
                    status.innerText = 'Permission denied or GPS off. Enable location and try again.';
                  },
                  { enableHighAccuracy: true, timeout: 10000 }
                );
              }
            </script>
            """,
            height=64,
        )

        if st.session_state.gps_coords:
            g_lat, g_lon = st.session_state.gps_coords
            st.caption(f"✅ Locked to GPS: **{g_lat:.4f}°N, {g_lon:.4f}°E**")
            if st.button("Use Typed City Name Instead", use_container_width=True):
                st.session_state.gps_coords = None
                st.query_params.clear()
                st.rerun()

        length = st.number_input("Field Length X (meters)", min_value=5.0, max_value=10000.0, value=60.0, step=10.0)
        width = st.number_input("Field Width Y (meters)", min_value=5.0, max_value=10000.0, value=40.0, step=10.0)
        area_ha = (length * width) / 10000.0
        st.caption(f"Total Area: **{length * width:,.0f} m²** ({area_ha:,.2f} hectares)")

    with col_f2:
        st.markdown("**2. Indian Soil & Crop Selection**")
        soil_choice = st.selectbox(
            "Soil Type",
            options=list(INDIAN_SOILS.keys()),
            format_func=lambda k: INDIAN_SOILS[k].label
        )
        st.caption(f"🪨 {INDIAN_SOILS[soil_choice].desc}")

        crop_category = st.selectbox("Crop Category", options=list(INDIAN_CROPS.keys()))
        crop_name = st.selectbox("Select Crop / Plant", options=list(INDIAN_CROPS[crop_category].keys()))
        crop_cfg = INDIAN_CROPS[crop_category][crop_name]
        st.caption(f"🌱 **{crop_name}:** {crop_cfg['notes']}")

    with col_f3:
        st.markdown("**3. Pump, Drip & Checkpoints**")
        pump_preset = st.selectbox(
            "Borewell Pump Size",
            ["5 HP Borewell (~400 L/min)", "3 HP Borewell (~250 L/min)", "7.5 HP Borewell (~600 L/min)", "Custom"]
        )
        if "5 HP" in pump_preset:
            pump_lpm = 400.0
            pump_kw = 4.4
        elif "3 HP" in pump_preset:
            pump_lpm = 250.0
            pump_kw = 2.6
        elif "7.5 HP" in pump_preset:
            pump_lpm = 600.0
            pump_kw = 6.5
        else:
            pump_lpm = st.number_input("Custom Pump Flow (Liters/min)", min_value=10.0, max_value=5000.0, value=400.0, step=25.0)
            pump_kw = (pump_lpm / 400.0) * 4.4

        drip_lpm = st.number_input("Drip System Flow (Liters/min)", min_value=10.0, max_value=3000.0, value=150.0, step=10.0,
                                   help="Typical field drip lateral delivery is 100–250 L/min depending on emitter count.")

        auto_spots = recommended_spot_count(length, width)
        use_auto = st.checkbox(f"Auto-scale checkpoints ({auto_spots} spots)", value=True)
        n_spots = auto_spots if use_auto else st.slider("Manual Checkpoint Count", min_value=5, max_value=35, value=auto_spots)

field = sz.FieldSpec(length, width)
texture = INDIAN_SOILS[soil_choice]
suggested = generate_serpentine_points(length, width, n_spots)

# ---------------- LIVE WEATHER BANNER (GPS OR TYPED CITY) ----------------
if st.session_state.gps_coords:
    g_lat, g_lon = st.session_state.gps_coords
    gps_place = reverse_geocode_coords(g_lat, g_lon)
    wx_data = fetch_weather_by_coords(g_lat, g_lon, f"📍 {gps_place}")
else:
    wx_data = fetch_live_weather(city_input)

wx_mult, wx_status, wx_title, wx_desc = evaluate_weather_advice(wx_data)

if wx_data:
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    w_col1.metric("📍 Farm Location", wx_data["location"])
    w_col2.metric("🌡️ Live Temp / Peak", f"{wx_data['temp_now']:.1f}°C", f"Today Max: {wx_data['temp_max']:.1f}°C")
    w_col3.metric("💧 Air Humidity", f"{wx_data['rh_now']:.0f}%")
    w_col4.metric("🌧️ 24h Rain Forecast", f"{wx_data['rain_prob']}%", f"Expected: {wx_data['rain_mm']:.1f} mm")

if wx_status == "rain":
    st.success(f"**{wx_title}** — {wx_desc}")
elif wx_status == "light_rain":
    st.info(f"**{wx_title}** — {wx_desc}")
elif wx_status == "hot":
    st.warning(f"**{wx_title}** — {wx_desc}")
else:
    st.info(f"**{wx_title}** — {wx_desc}")

# ---------------- STEP 1: VISUAL WALKING MAP & COORDINATES ----------------
st.divider()
step_header("STEP 1", "Field Walking Map & Checkpoints")
map_col, table_col = st.columns([1.45, 1])

with map_col:
    walk_fig = draw_walking_map(length, width, suggested)
    st.pyplot(walk_fig, use_container_width=True)
    walk_buf = io.BytesIO()
    walk_fig.savefig(walk_buf, format="png", dpi=200)
    plt.close(walk_fig)
    st.download_button("📲 Save Walking Map to Phone (PNG)", walk_buf.getvalue(),
                       file_name="walking_checkpoints.png", mime="image/png", use_container_width=True)

with table_col:
    st.markdown("**Checkpoint Coordinates (Meters from Start Corner):**")
    sug_df = pd.DataFrame(suggested, columns=["X (Length m)", "Y (Width m)"])
    sug_df.index = [f"Spot {i+1}" for i in range(len(sug_df))]
    st.dataframe(sug_df.style.format("{:.1f}"), use_container_width=True, height=360)

# ---------------- STEP 2: PASTE, UPLOAD CSV, OR DEMO SIMULATOR (AUTO-SAVED) ----------------
st.divider()
step_header("STEP 2", "Load Wand Data (Auto-Saves CSV & TXT Automatically)")

tab_paste, tab_file, tab_demo = st.tabs([
    "📋 Paste Bluetooth Log",
    "📁 Upload CSV / TXT File",
    "🎲 Offline Demo Simulator"
])

def_t = wx_data["temp_now"] if wx_data else 28.0
def_h = wx_data["rh_now"] if wx_data else 55.0

with tab_paste:
    st.caption("Copy the readings directly from **Serial Bluetooth Terminal** and paste below—data is automatically cleaned and saved as CSV & TXT.")
    pasted_text = st.text_area(
        "Paste copied Bluetooth log here:",
        height=130,
        placeholder="Example:\n10:14:02 Connected to HC-05\n10:14:15 1,24.5,31.2,48.0\n10:15:02 2,36.8,31.0,49.0\n10:15:48 3,41.2,30.8,50.0"
    )
    if pasted_text.strip():
        parsed_paste = parse_bluetooth_log(pasted_text, suggested, def_t, def_h)
        if parsed_paste:
            st.session_state.samples = parsed_paste
            auto_save_samples(parsed_paste, "Bluetooth Paste", crop_name)
        else:
            st.warning("Could not find valid sensor numbers in the pasted text yet.")

with tab_file:
    st.caption("Upload the saved `.csv` or `.txt` log file from your phone or computer—automatically cleaned and archived.")
    uploaded = st.file_uploader("Choose your saved field log file (.csv or .txt)", type=["csv", "txt", "log"])
    if uploaded is not None:
        parsed_file = parse_bluetooth_log(uploaded.getvalue(), suggested, def_t, def_h)
        if parsed_file:
            st.session_state.samples = parsed_file
            auto_save_samples(parsed_file, "File Upload", crop_name)
        else:
            st.error("No valid sensor readings found in that file.")

with tab_demo:
    st.caption("Test the zoning map, water savings, and pump runtime schedule instantly without hardware connected.")
    if st.button("🎲 Generate Realistic Demo Readings for This Field", type="primary", use_container_width=True):
        sim = sz.SimulatedSampleSource(field, texture, seed=7)
        demo_samples = [sim.read_sample(float(x), float(y)) for x, y in suggested]
        st.session_state.samples = demo_samples
        auto_save_samples(demo_samples, "Demo Scan", crop_name)

if st.session_state.samples:
    latest_log = st.session_state.saved_logs[0] if st.session_state.saved_logs else None
    if latest_log:
        st.success(f"✅ **Auto-Saved ({len(st.session_state.samples)} readings)** — `{latest_log['base_name']}.csv` & `.txt` saved automatically.")
        dl_c1, dl_c2, dl_c3 = st.columns([1, 1, 0.8])
        dl_c1.download_button("📄 Export Auto-Saved CSV", latest_log["csv_data"],
                              file_name=f"{latest_log['base_name']}.csv", mime="text/csv", use_container_width=True)
        dl_c2.download_button("📝 Export Auto-Saved TXT", latest_log["txt_data"],
                              file_name=f"{latest_log['base_name']}.txt", mime="text/plain", use_container_width=True)
        if dl_c3.button("🗑️ Clear Active Readings", use_container_width=True):
            st.session_state.samples = []
            st.rerun()

    rec_df = pd.DataFrame([
        {"Spot": f"Spot {i+1}", "X (m)": s.x, "Y (m)": s.y, "Moisture (%)": s.moisture, "Temp (°C)": s.temp_c, "Humidity (%)": s.rh}
        for i, s in enumerate(st.session_state.samples)
    ])
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

if len(st.session_state.saved_logs) > 1:
    with st.expander(f"🗂️ Auto-Saved Field Log History ({len(st.session_state.saved_logs)} scans saved)", expanded=False):
        log_titles = [entry["title"] for entry in st.session_state.saved_logs]
        chosen_title = st.selectbox("Switch to a previously auto-saved field log:", options=log_titles)
        if st.button("🔄 Load Selected Saved Log"):
            for entry in st.session_state.saved_logs:
                if entry["title"] == chosen_title:
                    st.session_state.samples = entry["samples"]
                    st.rerun()

# ---------------- STEP 3: AUTOMATICALLY GENERATE PREMIUM AI MAP, SAVINGS & SCHEDULE ----------------
st.divider()
step_header("STEP 3", "Zone-by-Zone Irrigation, Savings & Schedule")

if len(st.session_state.samples) < 3:
    st.info("Paste, upload, or generate at least **3 samples** in Step 2—your irrigation map & schedule will appear here automatically.")
else:
    with st.spinner("Generating Zone-by-Zone Map, Pump Runtimes & Resource Savings..."):
        clean_samples = sz.sanitize_samples(st.session_state.samples, field)
        model = sz.HybridMoistureModel(field, model="auto").fit(clean_samples)

        xc, yc, dx, dy = field.make_grid(280)
        XX, YY = np.meshgrid(xc, yc)
        pred = model.predict(np.column_stack([XX.ravel(), YY.ravel()]))
        moisture = pred["final"].reshape(XX.shape)
        vigor = sz.vigor_index(moisture, pred["temp"].reshape(XX.shape), pred["vpd"].reshape(XX.shape), texture)

        info = {
            "temp": float(np.mean(pred["temp"])),
            "rh": float(np.mean(pred["rh"])),
            "mean_moisture": float(moisture.mean()),
            "min_moisture": float(moisture.min()),
            "max_moisture": float(moisture.max()),
            "vigor": float(vigor.mean()),
        }

        et_info = estimate_evapotranspiration_and_schedule(
            info["temp"], info["rh"], info["mean_moisture"], texture, crop_cfg, wx_data
        )

        fig, total_liters, equiv_mm, zone_rows, pump_str, drip_str, roi = draw_premium_zoning_report(
            length, width, xc, yc, moisture, clean_samples, info, texture,
            crop_name, crop_cfg, wx_mult, wx_title, pump_lpm, drip_lpm, pump_kw, et_info
        )

        # Row 1: Water & Electricity Savings ROI Strip
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("💧 Water Saved vs. Uniform", f"{roi['saved_liters']:,.0f} L",
                  f"{roi['saved_pct']:.1f}% less water used")
        s2.metric("⚡ Electricity Saved", f"{roi['saved_kwh']:.1f} kWh",
                  f"Pump time saved: {roi['saved_time_str']}")
        s3.metric(f"Total {crop_name} Water", f"{total_liters:,.0f} L",
                  "Hold for rain" if wx_mult == 0 else f"~{equiv_mm:.1f} mm avg depth")
        s4.metric("📅 Next Watering Day", et_info["headline"], et_info["subtext"])

        # Row 2: Evaporation & Total Runtime Strip
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Avg Soil Moisture", f"{info['mean_moisture']:.1f}%",
                  f"Range: {info['min_moisture']:.0f}% – {info['max_moisture']:.0f}%")
        e2.metric("☀️ Crop Evaporation (ETc)", f"{et_info['etc_mm']:.1f} mm / day",
                  f"Dry-down: -{et_info['daily_loss_pct']:.1f}% / day")
        e3.metric(f"⚡ Total Borewell ({pump_lpm:.0f} L/m)", pump_str, "Combined across dry zones")
        e4.metric(f"💧 Total Drip ({drip_lpm:.0f} L/m)", drip_str, "Combined across dry zones")

        # Zone-by-Zone Valve Runtime Table
        st.markdown("#### 🚰 Zone-by-Zone Water & Valve Runtime Schedule")
        zone_table = pd.DataFrame([
            {
                "Management Zone": r["label"],
                "Field Share": f"{r['pct_area']:.1f}% ({r['area_m2']:,.0f} m²)",
                "Avg Moisture": f"{r['avg_moisture']:.1f}%" if r["pct_area"] > 0 else "—",
                "Water Needed": f"{r['liters']:,.0f} L" if r["liters"] > 0 else "0 L (Skip)",
                f"⚡ Borewell Pump ({pump_lpm:.0f} L/m)": r["pump_time"],
                f"💧 Drip Valve ({drip_lpm:.0f} L/m)": r["drip_time"],
            }
            for r in zone_rows
        ])
        st.dataframe(zone_table, use_container_width=True, hide_index=True)

        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=260)
        plt.close(fig)
        st.download_button("📥 Download Full Zone-by-Zone & Savings Report (PNG)", buf.getvalue(),
                           file_name="soilscout_precision_report.png", mime="image/png", use_container_width=True)
