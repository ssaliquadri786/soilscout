import io
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

import soil_zoning as sz

st.set_page_config(page_title="SoilScout AI", page_icon="🌱", layout="wide")
st.title("🌱 SoilScout AI — Precision Irrigation Zoning")

if "samples" not in st.session_state:
    st.session_state.samples = []
if "bt_conn" not in st.session_state:
    st.session_state.bt_conn = None


def recommended_spot_count(length_m: float, width_m: float) -> int:
    """Scales checkpoint count smoothly with field area (hectares)."""
    hectares = (length_m * width_m) / 10000.0
    spots = int(round(6 + 4.0 * math.sqrt(hectares)))
    return max(5, min(30, spots))


def draw_walking_map(field: sz.FieldSpec, points: np.ndarray):
    """Generates a clean visual checkpoint & zig-zag route map."""
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F1F8F4")

    rect = plt.Rectangle(
        (0, 0), field.length_m, field.width_m,
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
    ax.text(field.length_m * 0.015, -field.width_m * 0.065, "🚩 START CORNER (0, 0)",
            color="#DC2626", fontsize=9.5, fontweight="bold")

    pad_x = field.length_m * 0.06
    pad_y = field.width_m * 0.09
    ax.set_xlim(-pad_x, field.length_m + pad_x)
    ax.set_ylim(-pad_y, field.width_m + pad_y)
    ax.set_xlabel("Field Length X (meters)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Field Width Y (meters)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title(
        f"Field Sampling Guide  •  {field.length_m:.0f}m × {field.width_m:.0f}m  ({len(points)} Checkpoints)",
        fontsize=12.5, fontweight="bold", color="#0F172A", pad=12
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def extract_zone_summary(zones: np.ndarray, moisture: np.ndarray, stats, cell_area_m2: float, texture):
    """Safely extracts area, moisture, and water liters per zone."""
    fc = getattr(texture, "field_capacity", 38.0)
    wp = getattr(texture, "wilting_point", 18.0)

    # Default 4 agronomic tiers based on moisture thresholds
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
        # Water needed to bring dry zones up to 85% of field capacity over 0.30m root depth
        target_m = 0.85 * fc
        deficit_pct = max(0.0, target_m - avg_m) if count > 0 else 0.0
        liters = area_m2 * 0.30 * (deficit_pct / 100.0) * 1000.0 / 0.85
        summary.append({
            "label": label,
            "color": color,
            "pct_area": pct_area,
            "area_m2": area_m2,
            "avg_moisture": avg_m,
            "liters": liters,
        })
    return summary, wp, fc


def draw_premium_zoning_report(field, xc, yc, moisture, vigor, zones, stats, samples, info, texture):
    """Builds a high-resolution, executive-grade irrigation zoning map & analytics card."""
    cell_area = (xc[1] - xc[0]) * (yc[1] - yc[0]) if len(xc) > 1 and len(yc) > 1 else 1.0
    zone_rows, wp, fc = extract_zone_summary(zones, moisture, stats, cell_area, texture)
    total_liters = sum(r["liters"] for r in zone_rows)

    fig = plt.figure(figsize=(15.5, 8.8), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")

    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.05, 1.0],
                          wspace=0.18, hspace=0.28, left=0.055, right=0.97, top=0.88, bottom=0.08)

    # Top Banner Header
    fig.text(0.055, 0.945, "SOILSCOUT AI  |  PRECISION IRRIGATION & MOISTURE ZONING MAP",
             fontsize=15, fontweight="bold", color="#0F172A")
    fig.text(
        0.055, 0.915,
        f"Field: {field.length_m:.0f}m × {field.width_m:.0f}m ({(field.length_m*field.width_m)/10000:.2f} ha)   •   "
        f"Soil: {texture.label}   •   Ambient: {info['temp']:.1f}°C, {info['rh']:.0f}% RH   •   "
        f"Samples: {info['n']}",
        fontsize=10.5, color="#475569"
    )

    # ---------------- LEFT: MAIN SPATIAL MOISTURE & ZONE MAP ----------------
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor("#FFFFFF")

    # Refined agronomic colormap: Terracotta Red -> Warm Amber -> Lush Emerald -> Deep Sky Blue
    cmap = LinearSegmentedColormap.from_list(
        "soil_premium",
        ["#B91C1C", "#EF4444", "#F59E0B", "#84CC16", "#10B981", "#0EA5E9", "#1D4ED8"],
        N=256
    )

    vmin = max(0.0, min(float(moisture.min()) - 2.0, wp - 3.0))
    vmax = min(100.0, max(float(moisture.max()) + 2.0, fc + 8.0))
    levels = np.linspace(vmin, vmax, 36)

    cf = ax_map.contourf(xc, yc, moisture, levels=levels, cmap=cmap, extend="both", antialiased=True)

    # Crisp management zone boundary lines
    t1 = wp + 0.25 * (fc - wp)
    t2 = wp + 0.60 * (fc - wp)
    t3 = fc + 4.0
    valid_bounds = [b for b in [t1, t2, t3] if vmin < b < vmax]
    if valid_bounds:
        cs = ax_map.contour(xc, yc, moisture, levels=valid_bounds, colors="#0F172A",
                            linewidths=1.4, linestyles="--", alpha=0.75)
        ax_map.clabel(cs, inline=True, fontsize=8, fmt="%.0f%%")

    # Subtle grid & border
    ax_map.grid(True, linestyle=":", alpha=0.35, color="#0F172A")
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    # Plot Sampled Checkpoints with clean badges
    for i, s in enumerate(samples):
        ax_map.scatter([s.x], [s.y], s=110, color="#0F172A", edgecolors="white", linewidth=1.8, zorder=6)
        ax_map.annotate(
            f"S{i+1}: {s.moisture:.0f}%",
            xy=(s.x, s.y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#94A3B8", alpha=0.92),
            zorder=7
        )

    ax_map.set_xlim(0, field.length_m)
    ax_map.set_ylim(0, field.width_m)
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

    # ---------------- BOTTOM RIGHT: IRRIGATION PRESCRIPTION CARD ----------------
    ax_card = fig.add_subplot(gs[1, 1])
    ax_card.set_facecolor("#FFFFFF")
    ax_card.set_xticks([])
    ax_card.set_yticks([])
    for spine in ax_card.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.5)

    ax_card.text(0.05, 0.89, "Irrigation Prescription & Water Budget",
                 fontsize=11.5, fontweight="bold", color="#0F172A", transform=ax_card.transAxes)

    y_pos = 0.72
    for r in zone_rows:
        ax_card.scatter([0.07], [y_pos + 0.02], s=90, color=r["color"], transform=ax_card.transAxes)
        ax_card.text(0.12, y_pos, f"{r['label']} ({r['area_m2']:,.0f} m²)",
                     fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_card.transAxes)
        water_txt = f"{r['liters']:,.0f} L" if r["liters"] > 0 else "0 L (Skip)"
        ax_card.text(0.95, y_pos, water_txt, fontsize=9.5, fontweight="bold",
                     color="#0F172A", ha="right", transform=ax_card.transAxes)
        y_pos -= 0.145

    ax_card.plot([0.05, 0.95], [0.18, 0.18], color="#E2E8F0", linewidth=1.2, transform=ax_card.transAxes)
    ax_card.text(0.05, 0.07, "TOTAL RECOMMENDED IRRIGATION:",
                 fontsize=9.5, fontweight="bold", color="#475569", transform=ax_card.transAxes)
    ax_card.text(0.95, 0.07, f"{total_liters:,.0f} Liters",
                 fontsize=11, fontweight="bold", color="#047857", ha="right", transform=ax_card.transAxes)

    return fig, total_liters, zone_rows


# ---------------- SIDEBAR: FIELD & CONNECTION SETUP ----------------
with st.sidebar:
    st.header("1. Field Details")
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
    st.header("2. Data Input Mode")
    mode = st.radio(
        "How are you collecting data?",
        ["Upload CSV File (From Phone)", "Live Bluetooth Wand (HC-05)", "Offline Demo Simulator"]
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

# ---------------- STEP 1: VISUAL WALKING MAP & COORDINATES ----------------
st.subheader("Step 1: Field Walking Map & Checkpoints")
map_col, table_col = st.columns([1.45, 1])

with map_col:
    walk_fig = draw_walking_map(field, suggested)
    st.pyplot(walk_fig, use_container_width=True)
    walk_buf = io.BytesIO()
    walk_fig.savefig(walk_buf, format="png", dpi=200)
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

if mode == "Upload CSV File (From Phone)":
    st.info("Upload the CSV saved from **Serial Bluetooth Terminal** (`spot,moisture,temp_c,rh`). Spots 1, 2, 3... automatically match the walking map above.")
    uploaded = st.file_uploader("Choose your field CSV file", type=["csv", "txt"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = [c.strip().lower() for c in df_up.columns]
            loaded = []
            for i, row in df_up.iterrows():
                if "x" in df_up.columns and "y" in df_up.columns:
                    px, py = float(row["x"]), float(row["y"])
                else:
                    spot_idx = int(row["spot"]) - 1 if "spot" in df_up.columns else i
                    spot_idx = max(0, min(len(suggested) - 1, spot_idx))
                    px, py = float(suggested[spot_idx][0]), float(suggested[spot_idx][1])

                loaded.append(sz.Sample(
                    x=px,
                    y=py,
                    moisture=float(row["moisture"]),
                    temp_c=float(row.get("temp_c", 28.0)),
                    rh=float(row.get("rh", 55.0)),
                    source="csv"
                ))
            st.session_state.samples = loaded
            st.success(f"Matched {len(loaded)} sensor readings to your field map!")
        except Exception as e:
            st.error(f"Could not read file: {e}")

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
st.subheader("Step 3: Precision AI Irrigation & Zoning Map")

if len(st.session_state.samples) < 3:
    st.info("Record or upload at least **3 samples** to generate the irrigation zoning map.")
else:
    if st.button("🗺️ Generate Precision Zoning Map", type="primary", use_container_width=True):
        with st.spinner("Running High-Resolution Spatial AI Interpolation..."):
            clean_samples = sz.sanitize_samples(st.session_state.samples, field)
            model = sz.HybridMoistureModel(field, model="auto").fit(clean_samples)
            loo = model.loo_cv()

            # Higher grid resolution (280x280) for ultra-smooth contours
            xc, yc, dx, dy = field.make_grid(280)
            XX, YY = np.meshgrid(xc, yc)
            pred = model.predict(np.column_stack([XX.ravel(), YY.ravel()]))
            moisture = pred["final"].reshape(XX.shape)
            vigor = sz.vigor_index(moisture, pred["temp"].reshape(XX.shape), pred["vpd"].reshape(XX.shape), texture)

            zones, _ = sz.delineate_zones(moisture, method="threshold")
            stats = sz.compute_zone_statistics(zones, moisture, vigor, field, dx * dy, texture, 0.30, 0.85)

            info = {
                "kind": model.kind or "mean",
                "n": len(clean_samples),
                "source": clean_samples[0].source,
                "loo": loo,
                "temp": float(np.mean(pred["temp"])),
                "rh": float(np.mean(pred["rh"])),
                "vpd": float(np.mean(pred["vpd"])),
                "mean_moisture": float(moisture.mean()),
                "min_moisture": float(moisture.min()),
                "max_moisture": float(moisture.max()),
                "vigor": float(vigor.mean()),
            }

            fig, total_liters, zone_rows = draw_premium_zoning_report(
                field, xc, yc, moisture, vigor, zones, stats, clean_samples, info, texture
            )

            # Top KPI Metrics Strip
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Soil Moisture", f"{info['mean_moisture']:.1f}%",
                      f"Range: {info['min_moisture']:.0f}% – {info['max_moisture']:.0f}%")
            k2.metric("Total Water Needed", f"{total_liters:,.0f} L", "To restore optimal moisture")
            dry_pct = zone_rows[0]["pct_area"] + zone_rows[1]["pct_area"]
            k3.metric("Dry Area Requiring Water", f"{dry_pct:.1f}%", f"{(dry_pct/100)*length*width:,.0f} m²")
            k4.metric("Crop Vigor Index", f"{info['vigor']*100:.0f} / 100", f"Temp {info['temp']:.1f}°C | RH {info['rh']:.0f}%")

            st.pyplot(fig, use_container_width=True)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=260)
            st.download_button("📥 Download High-Res Irrigation Report (PNG)", buf.getvalue(),
                               file_name="soilscout_precision_report.png", mime="image/png", use_container_width=True)
