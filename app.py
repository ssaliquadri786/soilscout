import io
import re
import json
import math
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import soil_zoning as sz

st.set_page_config(page_title="SoilScout AI", page_icon="🌱", layout="wide")
st.title("🌱 SoilScout AI — Precision Irrigation & Weather Advisor")

if "samples" not in st.session_state:
    st.session_state.samples = []
if "bt_conn" not in st.session_state:
    st.session_state.bt_conn = None


# ---------------- HELPER FUNCTIONS ----------------
def recommended_spot_count(length_m: float, width_m: float) -> int:
    """Scales checkpoint count smoothly with field area (hectares)."""
    hectares = (length_m * width_m) / 10000.0
    spots = int(round(6 + 4.0 * math.sqrt(hectares)))
    return max(5, min(30, spots))


def format_runtime(liters: float, flow_lpm: float) -> str:
    """Converts water volume (L) and flow rate (L/min) into readable hours & mins."""
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
def fetch_live_weather(city_name: str):
    """Fetches real-time weather & 24h rain forecast via Open-Meteo (no API key needed)."""
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

        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
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
            "location": resolved_name,
            "temp_now": temp_now,
            "rh_now": rh_now,
            "rain_mm": rain_mm,
            "rain_prob": rain_prob,
            "temp_max": temp_max,
        }
    except Exception:
        return None


def evaluate_weather_advice(wx: dict | None):
    """Determines irrigation multiplier and farmer advisory based on forecast."""
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


def parse_bluetooth_log(raw_bytes: bytes, suggested_pts: np.ndarray, default_temp: float, default_rh: float):
    """
    Smart parser for Serial Bluetooth Terminal logs (.txt or .csv).
    Strips timestamps, ignores 'Connected/Disconnected' status lines, and maps spots to (x, y).
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    samples = []

    # Regex strips leading timestamps like "10:14:22 " or "2026-11-13 10:14:22.120 "
    ts_pattern = re.compile(r"^\s*(?:\d{4}[-/]\d{2}[-/]\d{2}\s+)?\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*[-:>]*\s*")
    # Matches comma/semicolon/tab separated numbers
    num_pattern = re.compile(r"[-+]?\d*\.?\d+")

    for raw_line in lines:
        cleaned = ts_pattern.sub("", raw_line).strip()
        if not cleaned:
            continue

        lower = cleaned.lower()
        # Skip header lines or Bluetooth terminal status messages
        if any(word in lower for word in ["moisture", "connect", "disconnect", "pong", "ok,", "err,", "bluetooth", "terminal"]):
            continue

        nums = [float(x) for x in num_pattern.findall(cleaned)]
        if len(nums) < 1:
            continue

        idx = len(samples)
        if len(nums) >= 5:
            # Format: x, y, moisture, temp_c, rh
            px, py, moist, t_c, rh = nums[0], nums[1], nums[2], nums[3], nums[4]
        elif len(nums) == 4:
            # Format: spot, moisture, temp_c, rh
            spot_num = int(round(nums[0])) - 1
            spot_idx = max(0, min(len(suggested_pts) - 1, spot_num if spot_num >= 0 else idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[1], nums[2], nums[3]
        elif len(nums) == 3:
            # Format: spot, moisture, temp_c  OR  moisture, temp_c, rh
            if nums[0] == float(idx + 1) and nums[0] <= 50:
                spot_idx = max(0, min(len(suggested_pts) - 1, idx))
                px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
                moist, t_c, rh = nums[1], nums[2], default_rh
            else:
                spot_idx = max(0, min(len(suggested_pts) - 1, idx))
                px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
                moist, t_c, rh = nums[0], nums[1], nums[2]
        elif len(nums) == 2:
            # Format: spot, moisture
            spot_idx = max(0, min(len(suggested_pts) - 1, idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[1], default_temp, default_rh
        else:
            # Single number: moisture only
            spot_idx = max(0, min(len(suggested_pts) - 1, idx))
            px, py = float(suggested_pts[spot_idx][0]), float(suggested_pts[spot_idx][1])
            moist, t_c, rh = nums[0], default_temp, default_rh

        # Sanity clamp physical bounds
        moist = max(0.0, min(100.0, moist))
        t_c = max(-10.0, min(65.0, t_c))
        rh = max(0.0, min(100.0, rh))

        samples.append(sz.Sample(x=px, y=py, moisture=moist, temp_c=t_c, rh=rh, source="csv"))

    return samples


def draw_walking_map(length_m: float, width_m: float, points: np.ndarray):
    """Generates a clean visual checkpoint & zig-zag route map."""
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
    ax.plot(xs, ys, linestyle="--", color="#D97706", linewidth=2.0, zorder=3)
    for i in range(len(points) - 1):
        ax.annotate(
            "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.6, shrinkA=9, shrinkB=9),
            zorder=3
        )

    ax.scatter(xs, ys, s=300, color="#1B4332", edgecolors="white", linewidth=2.0, zorder=4)
    for i, (x, y) in enumerate(points):
        ax.text(x, y, str(i + 1), color="white", fontsize=8.5, fontweight="bold",
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
        f"Field Sampling Guide  •  {length_m:.0f}m × {width_m:.0f}m  ({len(points)} Checkpoints)",
        fontsize=12.5, fontweight="bold", color="#0F172A", pad=12
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def extract_zone_summary(moisture: np.ndarray, cell_area_m2: float, texture, wx_multiplier: float):
    """Calculates zone area, moisture averages, and weather-adjusted water liters."""
    fc = getattr(texture, "field_capacity", 38.0)
    wp = getattr(texture, "wilting_point", 18.0)

    tiers = [
        ("Critical Dry", "#DC2626", moisture < (wp + 0.25 * (fc - wp))),
        ("Needs Water", "#F59E0B", (moisture >= (wp + 0.25 * (fc - wp))) & (moisture < (wp + 0.60 * (fc - wp)))),
        ("Optimal Zone", "#10B981", (moisture >= (wp + 0.60 * (fc - wp))) & (moisture <= (fc + 4.0))),
        ("Over-Watered", "#2563EB", moisture > (fc + 4.0)),
    ]

    total_cells = moisture.size
    summary = []
    for label, color, mask in tiers:
        count = int(np.sum(mask))
        pct_area = 100.0 * count / total_cells
        area_m2 = count * cell_area_m2
        avg_m = float(np.mean(moisture[mask])) if count > 0 else 0.0
        target_m = 0.85 * fc
        deficit_pct = max(0.0, target_m - avg_m) if count > 0 else 0.0
        base_liters = area_m2 * 0.30 * (deficit_pct / 100.0) * 1000.0 / 0.85
        adj_liters = base_liters * wx_multiplier
        summary.append({
            "label": label,
            "color": color,
            "pct_area": pct_area,
            "area_m2": area_m2,
            "avg_moisture": avg_m,
            "base_liters": base_liters,
            "liters": adj_liters,
        })
    return summary, wp, fc


def draw_premium_zoning_report(length_m, width_m, xc, yc, moisture, samples, info, texture,
                               wx_multiplier, wx_headline, pump_lpm, drip_lpm):
    """Builds a high-resolution, executive-grade irrigation zoning map & pump/drip runtime report."""
    cell_area = (xc[1] - xc[0]) * (yc[1] - yc[0]) if len(xc) > 1 and len(yc) > 1 else 1.0
    zone_rows, wp, fc = extract_zone_summary(moisture, cell_area, texture, wx_multiplier)
    total_liters = sum(r["liters"] for r in zone_rows)

    pump_time_str = format_runtime(total_liters, pump_lpm)
    drip_time_str = format_runtime(total_liters, drip_lpm)

    fig = plt.figure(figsize=(15.5, 9.2), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")

    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[0.95, 1.15],
                          wspace=0.18, hspace=0.28, left=0.055, right=0.97, top=0.87, bottom=0.07)

    fig.text(0.055, 0.945, "SOILSCOUT AI  |  PRECISION IRRIGATION & MOISTURE ZONING REPORT",
             fontsize=15, fontweight="bold", color="#0F172A")
    fig.text(
        0.055, 0.915,
        f"Field: {length_m:.0f}m × {width_m:.0f}m ({(length_m * width_m)/10000:.2f} ha)   •   "
        f"Soil: {texture.label}   •   Sensor Avg: {info['temp']:.1f}°C, {info['rh']:.0f}% RH   •   "
        f"{wx_headline}",
        fontsize=9.8, color="#334155", fontweight="bold"
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

    t1 = wp + 0.25 * (fc - wp)
    t2 = wp + 0.60 * (fc - wp)
    t3 = fc + 4.0
    valid_bounds = [b for b in [t1, t2, t3] if vmin < b < vmax]
    if valid_bounds:
        cs = ax_map.contour(xc, yc, moisture, levels=valid_bounds, colors="#0F172A",
                            linewidths=1.4, linestyles="--", alpha=0.75)
        ax_map.clabel(cs, inline=True, fontsize=8, fmt="%.0f%%")

    ax_map.grid(True, linestyle=":", alpha=0.35, color="#0F172A")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    for i, s in enumerate(samples):
        ax_map.scatter([s.x], [s.y], s=110, color="#0F172A", edgecolors="white", linewidth=1.8, zorder=6)
        ax_map.annotate(
            f"S{i+1}: {s.moisture:.0f}%",
            xy=(s.x, s.y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#94A3B8", alpha=0.92),
            zorder=7
        )

    ax_map.set_xlim(0, length_m)
    ax_map.set_ylim(0, width_m)
    ax_map.set_xlabel("Field Length X (meters)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax_map.set_ylabel("Field Width Y (meters)", fontsize=10.5, fontweight="bold", color="#1E293B", labelpad=8)
    ax_map.set_title("Spatial Soil Moisture Distribution & Zone Boundaries",
                     fontsize=12, fontweight="bold", color="#0F172A", pad=10)

    cbar = fig.colorbar(cf, ax=ax_map, orientation="horizontal", fraction=0.046, pad=0.11, aspect=35)
    cbar.set_label("Volumetric Soil Moisture (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
    cbar.outline.set_edgecolor("#CBD5E1")

    # ---------------- TOP RIGHT: ZONE AREA DISTRIBUTION ----------------
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor("#FFFFFF")
    labels = [r["label"] for r in zone_rows][::-1]
    pcts = [r["pct_area"] for r in zone_rows][::-1]
    colors = [r["color"] for r in zone_rows][::-1]

    bars = ax_bar.barh(labels, pcts, color=colors, height=0.55, edgecolor="none")
    ax_bar.set_xlim(0, 115)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")
    ax_bar.set_xlabel("Share of Total Field Area (%)", fontsize=9.5, fontweight="bold", color="#334155")
    ax_bar.set_title("Management Zone Coverage", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)

    for bar, pct in zip(bars, pcts):
        ax_bar.text(pct + 2.0, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%",
                    va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1E293B")

    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#CBD5E1")
    ax_bar.spines["bottom"].set_color("#CBD5E1")

    # ---------------- BOTTOM RIGHT: IRRIGATION & PUMP/DRIP CARD ----------------
    ax_card = fig.add_subplot(gs[1, 1])
    ax_card.set_facecolor("#FFFFFF")
    ax_card.set_xticks([])
    ax_card.set_yticks([])
    for spine in ax_card.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    ax_card.text(0.05, 0.90, "Irrigation Prescription & Equipment Runtime",
                 fontsize=11.5, fontweight="bold", color="#0F172A", transform=ax_card.transAxes)

    y_pos = 0.76
    for r in zone_rows:
        ax_card.scatter([0.07], [y_pos + 0.015], s=85, color=r["color"], transform=ax_card.transAxes)
        ax_card.text(0.12, y_pos, f"{r['label']} ({r['area_m2']:,.0f} m²)",
                     fontsize=9.2, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
        water_txt = f"{r['liters']:,.0f} L" if r["liters"] > 0 else "0 L (Skip)"
        ax_card.text(0.95, y_pos, water_txt, fontsize=9.2, fontweight="bold",
                     color="#0F172A", ha="right", transform=ax_card.transAxes)
        y_pos -= 0.125

    ax_card.plot([0.05, 0.95], [0.26, 0.26], color="#E2E8F0", linewidth=1.2, transform=ax_card.transAxes)

    ax_card.text(0.05, 0.18, "TOTAL WATER NEEDED:",
                 fontsize=9.2, fontweight="bold", color="#475569", transform=ax_card.transAxes)
    ax_card.text(0.95, 0.18, f"{total_liters:,.0f} Liters",
                 fontsize=10.5, fontweight="bold", color="#047857", ha="right", transform=ax_card.transAxes)

    ax_card.text(0.05, 0.10, f"⚡ Borewell Pump ({pump_lpm:.0f} L/min):",
                 fontsize=9.2, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
    ax_card.text(0.95, 0.10, pump_time_str,
                 fontsize=9.5, fontweight="bold", color="#1D4ED8", ha="right", transform=ax_card.transAxes)

    ax_card.text(0.05, 0.03, f"💧 Drip System ({drip_lpm:.0f} L/min):",
                 fontsize=9.2, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
    ax_card.text(0.95, 0.03, drip_time_str,
                 fontsize=9.5, fontweight="bold", color="#0369A1", ha="right", transform=ax_card.transAxes)

    return fig, total_liters, zone_rows, pump_time_str, drip_time_str


# ---------------- SIDEBAR: FIELD, WEATHER & EQUIPMENT SETUP ----------------
with st.sidebar:
    st.header("1. Field & Location")
    city_input = st.text_input("Farm Location (City / Town)", value="Bengaluru")
    length = st.number_input("Field Length X (meters)", min_value=5.0, max_value=10000.0, value=60.0, step=10.0)
    width = st.number_input("Field Width Y (meters)", min_value=5.0, max_value=10000.0, value=40.0, step=10.0)

    area_ha = (length * width) / 10000.0
    st.caption(f"Total Area: **{length * width:,.0f} m²** ({area_ha:.2f} hectares)")

    soil_choice = st.selectbox(
        "Soil Texture",
        options=list(sz.TEXTURES.keys()),
        format_func=lambda k: sz.TEXTURES[k].label
    )

    auto_spots = recommended_spot_count(length, width)
    use_auto = st.checkbox(f"Auto-scale checkpoints by area ({auto_spots} spots)", value=True)
    n_spots = auto_spots if use_auto else st.slider("Manual Checkpoint Count", min_value=5, max_value=40, value=auto_spots)

    st.divider()
    st.header("2. Pump & Drip Flow Rates")
    pump_preset = st.selectbox(
        "Borewell Pump Size",
        ["5 HP Borewell (~400 L/min)", "3 HP Borewell (~250 L/min)", "7.5 HP Borewell (~600 L/min)", "Custom"]
    )
    if "5 HP" in pump_preset:
        pump_lpm = 400.0
    elif "3 HP" in pump_preset:
        pump_lpm = 250.0
    elif "7.5 HP" in pump_preset:
        pump_lpm = 600.0
    else:
        pump_lpm = st.number_input("Custom Pump Flow (Liters/min)", min_value=10.0, max_value=5000.0, value=400.0, step=25.0)

    drip_lpm = st.number_input("Drip System Flow (Liters/min)", min_value=10.0, max_value=3000.0, value=150.0, step=10.0,
                               help="Typical field drip lateral delivery is 100–250 L/min depending on emitter count.")

    st.divider()
    st.header("3. Data Input Mode")
    mode = st.radio(
        "How are you collecting data?",
        ["Upload CSV / Bluetooth Log", "Live Bluetooth Wand (HC-05)", "Offline Demo Simulator"]
    )

    if mode == "Live Bluetooth Wand (HC-05)":
        ports = sz.SerialSampleSource.candidate_ports()
        port_input = st.text_input("Bluetooth COM Port", value=ports[0] if ports else "COM5")
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("🔗 Connect", use_container_width=True):
            conn = sz.SerialSampleSource.auto_connect(port_input, baud=9600)
            if conn:
                st.session_state.bt_conn = conn
                st.success(f"Connected to {port_input}")
            else:
                st.error(f"Could not connect to {port_input}.")
        if col_c2.button("❌ Disconnect", use_container_width=True):
            if st.session_state.bt_conn:
                st.session_state.bt_conn.close()
                st.session_state.bt_conn = None
                st.info("Disconnected")

field = sz.FieldSpec(length, width)
texture = sz.TEXTURES[soil_choice]
suggested = sz.recommend_sampling_points(field, n_spots)

# ---------------- LIVE WEATHER BANNER ----------------
wx_data = fetch_live_weather(city_input)
wx_mult, wx_status, wx_title, wx_desc = evaluate_weather_advice(wx_data)

if wx_data:
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    w_col1.metric("📍 Location", wx_data["location"])
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
st.subheader("Step 1: Field Walking Map & Checkpoints")
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

# ---------------- STEP 2: RECORD OR UPLOAD READINGS ----------------
st.divider()
st.subheader("Step 2: Upload Phone Log or Record Readings")

idx = min(len(st.session_state.samples), len(suggested) - 1)
default_x, default_y = float(suggested[idx][0]), float(suggested[idx][1])

if mode == "Upload CSV / Bluetooth Log":
    st.info("Upload your raw log saved from **Serial Bluetooth Terminal** (`.csv` or `.txt`). Timestamps and status lines are automatically cleaned!")
    uploaded = st.file_uploader("Choose your field log file", type=["csv", "txt", "log"])
    if uploaded is not None:
        def_t = wx_data["temp_now"] if wx_data else 28.0
        def_h = wx_data["rh_now"] if wx_data else 55.0
        parsed = parse_bluetooth_log(uploaded.getvalue(), suggested, def_t, def_h)
        if parsed:
            st.session_state.samples = parsed
            st.success(f"Cleaned and matched {len(parsed)} sensor readings to your field map!")
        else:
            st.error("No valid sensor readings found in that file. Make sure it contains lines like `1,42.5,29.0,54.0`.")

elif mode == "Live Bluetooth Wand (HC-05)":
    c_x, c_y = st.columns(2)
    cur_x = c_x.number_input("Your Current X (m)", min_value=0.0, max_value=float(length), value=default_x)
    cur_y = c_y.number_input("Your Current Y (m)", min_value=0.0, max_value=float(width), value=default_y)

    if st.button("📡 Trigger Wand & Save Reading", type="primary", use_container_width=True):
        if st.session_state.bt_conn is None:
            st.warning("Click 'Connect' in the left sidebar first!")
        else:
            smp = st.session_state.bt_conn.read_sample(cur_x, cur_y)
            if smp:
                st.session_state.samples.append(smp)
                st.success(f"Saved: {smp.moisture:.1f}% moisture at ({cur_x}m, {cur_y}m)")
            else:
                st.error("Sensor read timed out. Try again.")

else:
    if st.button("🎲 Auto-Fill Simulated Readings for This Field", type="primary", use_container_width=True):
        sim = sz.SimulatedSampleSource(field, texture, seed=7)
        st.session_state.samples = [sim.read_sample(float(x), float(y)) for x, y in suggested]
        st.success(f"Generated {len(st.session_state.samples)} realistic field readings!")

if st.session_state.samples:
    rec_df = pd.DataFrame([
        {"Spot": f"Spot {i+1}", "X (m)": s.x, "Y (m)": s.y, "Moisture (%)": s.moisture, "Temp (°C)": s.temp_c, "Humidity (%)": s.rh}
        for i, s in enumerate(st.session_state.samples)
    ])
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear Readings"):
        st.session_state.samples = []
        st.rerun()

# ---------------- STEP 3: GENERATE PREMIUM AI MAP ----------------
st.divider()
st.subheader("Step 3: Precision AI Irrigation & Pump/Drip Runtime")

if len(st.session_state.samples) < 3:
    st.info("Record or upload at least **3 samples** to generate the irrigation zoning map.")
else:
    if st.button("🗺️ Generate Precision Zoning & Runtime Report", type="primary", use_container_width=True):
        with st.spinner("Running High-Resolution Spatial AI & Weather Adjustment..."):
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

            fig, total_liters, zone_rows, pump_str, drip_str = draw_premium_zoning_report(
                length, width, xc, yc, moisture, clean_samples, info, texture,
                wx_mult, wx_title, pump_lpm, drip_lpm
            )

            # Top KPI & Equipment Runtime Strip
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Soil Moisture", f"{info['mean_moisture']:.1f}%",
                      f"Range: {info['min_moisture']:.0f}% – {info['max_moisture']:.0f}%")
            k2.metric("Weather-Adjusted Water", f"{total_liters:,.0f} L",
                      "Hold for rain" if wx_mult == 0 else f"{int(wx_mult*100)}% of base need")
            k3.metric(f"⚡ Borewell ({pump_lpm:.0f} L/m)", pump_str, "Direct pump runtime")
            k4.metric(f"💧 Drip ({drip_lpm:.0f} L/m)", drip_str, "Drip valve runtime")

            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=260)
            plt.close(fig)
            st.download_button("📥 Download Irrigation & Runtime Report (PNG)", buf.getvalue(),
                               file_name="soilscout_precision_report.png", mime="image/png", use_container_width=True)
