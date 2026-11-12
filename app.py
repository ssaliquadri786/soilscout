import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")

import soil_zoning as sz

st.set_page_config(page_title="SoilScout AI", page_icon="🌱", layout="wide")
st.title("🌱 SoilScout AI — Smart Irrigation Zoning")

if "samples" not in st.session_state:
    st.session_state.samples = []
if "bt_conn" not in st.session_state:
    st.session_state.bt_conn = None

with st.sidebar:
    st.header("1. Field Details")
    length = st.number_input("Field Length X (meters)", min_value=5.0, value=60.0, step=5.0)
    width = st.number_input("Field Width Y (meters)", min_value=5.0, value=40.0, step=5.0)
    
    soil_choice = st.selectbox(
        "Soil Texture",
        options=list(sz.TEXTURES.keys()),
        format_func=lambda k: sz.TEXTURES[k].label
    )
    n_spots = st.slider("Suggested Checkpoints", min_value=5, max_value=10, value=8)

    st.divider()
    st.header("2. Data Input Mode")
    mode = st.radio(
        "How are you collecting data?",
        ["Live Bluetooth Wand (HC-05)", "Upload CSV File (From Phone)", "Offline Demo Simulator"]
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
                st.error(f"Could not connect to {port_input}. Check Bluetooth pairing.")
        if col_c2.button("❌ Disconnect", use_container_width=True):
            if st.session_state.bt_conn:
                st.session_state.bt_conn.close()
                st.session_state.bt_conn = None
                st.info("Disconnected")

field = sz.FieldSpec(length, width)
texture = sz.TEXTURES[soil_choice]
suggested = sz.recommend_sampling_points(field, n_spots)

st.subheader("Step 1: Walk to Checkpoints & Record Soil Data")
col1, col2 = st.columns([1, 1.3])

with col1:
    st.markdown("**Suggested Walking Spots (Serpentine Order):**")
    sug_df = pd.DataFrame(suggested, columns=["Target X (m)", "Target Y (m)"])
    sug_df.index = [f"Spot {i+1}" for i in range(len(sug_df))]
    st.dataframe(sug_df.style.format("{:.1f}"), use_container_width=True, height=250)

with col2:
    idx = min(len(st.session_state.samples), len(suggested) - 1)
    default_x, default_y = float(suggested[idx][0]), float(suggested[idx][1])

    if mode == "Live Bluetooth Wand (HC-05)":
        st.markdown("**Record Live Wand Reading:**")
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

    elif mode == "Upload CSV File (From Phone)":
        st.markdown("**Upload Phone CSV Data:**")
        template = "x,y,moisture,temp_c,rh\n7.5,10.0,32.0,29.0,52\n22.5,10.0,28.5,29.5,50\n37.5,10.0,51.0,28.0,56\n52.5,10.0,58.0,28.0,58\n52.5,30.0,82.0,27.5,64\n"
        st.download_button("⬇️ Download Sample CSV Template", template, file_name="field_samples.csv", mime="text/csv")
        
        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            loaded = []
            for _, row in df_up.iterrows():
                loaded.append(sz.Sample(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    moisture=float(row["moisture"]),
                    temp_c=float(row.get("temp_c", 28.0)),
                    rh=float(row.get("rh", 55.0)),
                    source="csv"
                ))
            st.session_state.samples = loaded
            st.success(f"Loaded {len(loaded)} readings from CSV!")

    else:
        st.markdown("**Test with Simulated Field Data:**")
        if st.button("🎲 Auto-Fill Simulated Readings", type="primary", use_container_width=True):
            sim = sz.SimulatedSampleSource(field, texture, seed=7)
            st.session_state.samples = [sim.read_sample(float(x), float(y)) for x, y in suggested]
            st.success(f"Generated {len(st.session_state.samples)} realistic field readings!")

    if st.session_state.samples:
        if st.button("🗑️ Clear All Readings"):
            st.session_state.samples = []
            st.rerun()

if st.session_state.samples:
    st.markdown("### Recorded Field Samples")
    rec_df = pd.DataFrame([
        {"Sample": f"S{i+1}", "X (m)": s.x, "Y (m)": s.y, "Moisture (%)": s.moisture, "Temp (°C)": s.temp_c, "Humidity (%)": s.rh}
        for i, s in enumerate(st.session_state.samples)
    ])
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Step 2: Generate AI Irrigation & Zoning Map")

if len(st.session_state.samples) < 3:
    st.info("Record or upload at least **3 samples** (5 to 8 recommended) to unlock the AI map.")
else:
    if st.button("🗺️ Generate AI Zoning Map", type="primary", use_container_width=True):
        with st.spinner("Running Gaussian Process + Residual IDW Spatial AI..."):
            clean_samples = sz.sanitize_samples(st.session_state.samples, field)
            model = sz.HybridMoistureModel(field, model="auto").fit(clean_samples)
            loo = model.loo_cv()

            xc, yc, dx, dy = field.make_grid(200)
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
                "root_depth_m": 0.30,
                "truth_rmse": None,
            }

            fig = sz.plot_report(field, xc, yc, zones, stats, clean_samples, suggested, info, texture, "soil_zones.png", show=False)
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            st.download_button("📥 Download Irrigation Report (PNG)", buf.getvalue(), file_name="irrigation_report.png", mime="image/png")