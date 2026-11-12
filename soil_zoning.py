#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soil_zoning.py - AI-assisted handheld soil moisture mapping and irrigation zoning
=================================================================================

Pipeline
--------
 1. Field size (L x W) -> 5-10 stratified sampling suggestions (equal sub-quadrant
    centroids, serpentine walking order).  Any arbitrary (x, y) is accepted.
 2. Samples arrive from the Arduino Mega over an HC-05 (pyserial) or, when no
    hardware is found, from a built-in field simulator (offline mode).
 3. Hybrid model:   Final = clip( ML_trend + IDW(residuals), 0, 100 )
        features : x/L, y/W, distance to centroid, ambient T, ambient RH, VPD
        baseline : Gaussian Process (ARD Matern + noise) or Random Forest
        residual : in-sample (actual - predicted), smoothed Inverse Distance Weighting
 4. Discrete agronomic zoning (thresholds or K-Means) + uniform/median filtering
    + removal of tiny islands -> clean polygon-like regions.
 5. Matplotlib report: zone map, boundaries, sample markers, area table, water advice.

Usage
-----
  python soil_zoning.py                       # demo: auto-detect HC-05, else simulation
  python soil_zoning.py --simulate --no-show  # force offline demo, save PNG only
  python soil_zoning.py --port COM5 --length 80 --width 50 --soil black_cotton
  python soil_zoning.py --interactive         # type your own (x y) for each sample
  python soil_zoning.py --list-ports

Serial protocol (see the .ino):  host sends "READ\\n", the Mega answers
  $SOIL,<seq>,<rawMedian>,<rawMean>,<moisture%>,<tempC>,<rh%>,<dhtOk>*<XOR-hex>

Assumptions you should know about
---------------------------------
 * The sensor percentage is treated as "% of saturation" (0 = air-dry, 100 = in water),
   i.e. relative wetness after the dry/wet calibration stored on the Mega.
 * There is no pH probe.  The "Soil Health / Vigor Index" is an agronomic heuristic from
   moisture (vs. a texture-specific optimum band), air temperature and VPD.  It is a
   screening indicator, not a laboratory measurement.
 * Irrigation volumes bring the root zone to 62.5 % (middle of the optimal band) and
   use texture porosity x root depth; divide by application efficiency for gross water.

Dependencies:  pip install numpy scipy scikit-learn matplotlib pyserial
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # pyserial is optional: without it the script simply runs in simulation mode
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore

    SERIAL_AVAILABLE = hasattr(serial, "Serial")  # guards against the wrong "serial" package
except Exception:  # pragma: no cover - environment dependent
    serial = None  # type: ignore
    list_ports = None  # type: ignore
    SERIAL_AVAILABLE = False

LOG = logging.getLogger("soil")

# --------------------------------------------------------------------------------------
# Constants / agronomic configuration
# --------------------------------------------------------------------------------------
DEFAULT_BAUD = 9600
MIN_SAMPLES = 5
MAX_SAMPLES = 10          # recommended range for the sampling plan
HARD_MAX_SAMPLES = 60     # safety cap for interactive sessions
N_FEATURES = 6

ZONE_BINS = (30.0, 50.0, 75.0)       # % moisture thresholds between the four zones
IRRIGATION_TARGET_PCT = 62.5         # centre of the optimal root zone (50-75 %)


@dataclass(frozen=True)
class ZoneDef:
    zid: int
    name: str
    range_label: str
    action: str
    color: str
    text_color: str


ZONES: Tuple[ZoneDef, ...] = (
    ZoneDef(0, "Critical Dry", "< 30 %", "Heavy irrigation needed", "#e8452c", "white"),
    ZoneDef(1, "Moderate Moisture", "30-50 %", "Light irrigation needed", "#f5d33d", "black"),
    ZoneDef(2, "Optimal Root Zone", "50-75 %", "No action required", "#43a047", "white"),
    ZoneDef(3, "Waterlogged / Saturated", "> 75 %", "Drainage / aeration needed", "#1e78d0", "white"),
)


@dataclass(frozen=True)
class SoilTexture:
    key: str
    label: str
    porosity: float                      # volumetric water content at saturation (m3/m3)
    optimum: Tuple[float, float]         # optimum relative-moisture band (%) for vigor


# Heuristic values: light soils drain fast (higher relative band), heavy soils go
# anaerobic earlier (lower band).  Tune them to your crop / local agronomy advice.
TEXTURES: Dict[str, SoilTexture] = {
    "sandy_loam": SoilTexture("sandy_loam", "Sandy Loam", 0.43, (50.0, 72.0)),
    "loam": SoilTexture("loam", "Loam", 0.47, (48.0, 72.0)),
    "clay": SoilTexture("clay", "Clay", 0.50, (45.0, 68.0)),
    "black_cotton": SoilTexture("black_cotton", "Black Cotton (Vertisol)", 0.52, (48.0, 70.0)),
}


def resolve_texture(name: str) -> SoilTexture:
    """Case/space/hyphen-insensitive lookup with prefix matching."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Soil texture must be a non-empty string.")
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    if key in TEXTURES:
        return TEXTURES[key]
    matches = [k for k in TEXTURES if k.startswith(key)]
    if len(matches) == 1:
        return TEXTURES[matches[0]]
    raise ValueError(f"Unknown soil texture '{name}'. Choose one of: {', '.join(TEXTURES)}")


# --------------------------------------------------------------------------------------
# Physics / agronomy helpers
# --------------------------------------------------------------------------------------
def saturation_vapor_pressure_kpa(temp_c):
    """Tetens equation (kPa)."""
    t = np.clip(np.asarray(temp_c, dtype=float), -40.0, 70.0)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def compute_vpd_kpa(temp_c, rh_pct):
    """Vapor pressure deficit in kPa; RH is clipped to [0, 100]."""
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    return np.maximum(saturation_vapor_pressure_kpa(temp_c) * (1.0 - rh / 100.0), 0.0)


def _trapezoid(v, lo0, lo1, hi1, hi0):
    """Suitability in [0, 1]: 0 below lo0, ramps to 1 at lo1, flat to hi1, ramps to 0 at hi0."""
    v = np.asarray(v, dtype=float)
    up = np.clip((v - lo0) / max(lo1 - lo0, 1e-9), 0.0, 1.0)
    down = np.clip((hi0 - v) / max(hi0 - hi1, 1e-9), 0.0, 1.0)
    return np.minimum(up, down)


def vigor_index(moisture, temp_c, vpd_kpa, texture: SoilTexture):
    """
    Soil Health / Vigor Index (0-100) - weighted geometric mean (limiting-factor style) of
      moisture suitability (texture-specific band), temperature suitability and VPD suitability.
    """
    lo, hi = texture.optimum
    s_m = _trapezoid(moisture, 10.0, lo, hi, 98.0)
    s_t = _trapezoid(temp_c, 5.0, 20.0, 30.0, 42.0)
    s_v = _trapezoid(vpd_kpa, -0.2, 0.4, 1.2, 3.0)
    floor = 0.01
    idx = (np.maximum(s_m, floor) ** 0.55) * (np.maximum(s_t, floor) ** 0.20) * (np.maximum(s_v, floor) ** 0.25)
    return 100.0 * idx


def vigor_label(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    if v >= 75:
        return "High"
    if v >= 50:
        return "Moderate"
    if v >= 25:
        return "Low"
    return "Poor"


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class Sample:
    x: float
    y: float
    moisture: float                 # %
    temp_c: float = float("nan")
    rh: float = float("nan")
    raw: Optional[float] = None
    source: str = "unknown"


@dataclass(frozen=True)
class FieldSpec:
    length: float   # x-direction (m)
    width: float    # y-direction (m)

    def __post_init__(self):
        for nm, v in (("length", self.length), ("width", self.width)):
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Field {nm} must be a positive, finite number of metres (got {v!r}).")

    @property
    def area(self) -> float:
        return float(self.length * self.width)

    @property
    def half_diag(self) -> float:
        return 0.5 * math.hypot(self.length, self.width)

    @property
    def centroid(self) -> Tuple[float, float]:
        return 0.5 * self.length, 0.5 * self.width

    def clip_point(self, x: float, y: float) -> Tuple[float, float]:
        return float(min(max(x, 0.0), self.length)), float(min(max(y, 0.0), self.width))

    def make_grid(self, cells_long: int = 200):
        """Cell-centre coordinates (xc, yc) and cell sizes (dx, dy)."""
        cells_long = int(max(20, min(cells_long, 600)))
        res = max(self.length, self.width) / cells_long
        nx = max(2, int(round(self.length / res)))
        ny = max(2, int(round(self.width / res)))
        dx, dy = self.length / nx, self.width / ny
        xc = (np.arange(nx) + 0.5) * dx
        yc = (np.arange(ny) + 0.5) * dy
        return xc, yc, dx, dy


# --------------------------------------------------------------------------------------
# 1) Sampling recommendation
# --------------------------------------------------------------------------------------
def recommend_sampling_points(field: FieldSpec, n: Optional[int] = None) -> np.ndarray:
    """
    Stratified sampling suggestions: the field is split into equal rectangular sub-quadrants and
    the centroids are returned in a serpentine walking order.

    When n has no sensible factorisation (5, 7, ...) the grid with the most square cells and at
    most 3 spare cells is used, and the n most widely spread centroids are kept (farthest-point
    selection), so the points never collapse onto a single line.
    """
    if n is None:
        n = int(round(3.0 + 1.5 * math.log10(max(field.area, 1.0))))
    n = int(n)
    clamped = int(min(max(n, MIN_SAMPLES), MAX_SAMPLES))
    if clamped != n:
        LOG.warning("Requested %d sampling points; using %d (supported range %d-%d).",
                    n, clamped, MIN_SAMPLES, MAX_SAMPLES)
    n = clamped

    best = None
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            cells = r * c
            if cells < n or cells > n + 3:
                continue
            cw, ch = field.length / c, field.width / r
            score = abs(math.log(cw / ch)) + 0.2 * (cells - n)
            if best is None or score < best[0] - 1e-12:
                best = (score, r, c)
    assert best is not None
    _, rows, cols = best
    cw, ch = field.length / cols, field.width / rows

    cells_ij = [(i, j) for i in range(rows) for j in range(cols)]
    cent = np.array([[(j + 0.5) * cw, (i + 0.5) * ch] for i, j in cells_ij], dtype=float)

    if len(cells_ij) > n:
        centre = np.array(field.centroid)
        chosen = [int(np.argmin(np.sum((cent - centre) ** 2, axis=1)))]
        dmin = np.sum((cent - cent[chosen[0]]) ** 2, axis=1)
        while len(chosen) < n:
            nxt = int(np.argmax(dmin))
            chosen.append(nxt)
            dmin = np.minimum(dmin, np.sum((cent - cent[nxt]) ** 2, axis=1))
    else:
        chosen = list(range(len(cells_ij)))

    chosen.sort(key=lambda k: (cells_ij[k][0], cells_ij[k][1] if cells_ij[k][0] % 2 == 0 else -cells_ij[k][1]))
    return cent[chosen]


# --------------------------------------------------------------------------------------
# 2) Data sources: HC-05 serial link and simulator
# --------------------------------------------------------------------------------------
_SOIL_RE = re.compile(r"^\$(?P<payload>SOIL,[^*]+)\*(?P<cs>[0-9A-Fa-f]{2})\s*$")


def _to_float(token: str) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return float("nan")


def parse_soil_line(line: str) -> Optional[dict]:
    """Validate checksum and ranges of a '$SOIL,...*HH' line. Returns None if invalid."""
    if not line:
        return None
    m = _SOIL_RE.match(line.strip())
    if not m:
        return None
    payload = m.group("payload")
    expected = 0
    for ch in payload:
        expected ^= ord(ch)
    if expected != int(m.group("cs"), 16):
        LOG.debug("Checksum mismatch on line %r", line)
        return None
    parts = payload.split(",")
    if len(parts) != 8:
        return None
    moisture = _to_float(parts[4])
    if not math.isfinite(moisture):
        return None
    temp, rh = _to_float(parts[5]), _to_float(parts[6])
    if not (-30.0 <= temp <= 70.0):
        temp = float("nan")
    if not (0.0 <= rh <= 100.0):
        rh = float("nan")
    return {
        "seq": int(_to_float(parts[1])) if math.isfinite(_to_float(parts[1])) else -1,
        "raw_median": _to_float(parts[2]),
        "raw_mean": _to_float(parts[3]),
        "moisture": float(min(max(moisture, 0.0), 100.0)),
        "temp": temp,
        "rh": rh,
        "dht_ok": parts[7].strip() == "1",
    }


class SerialSampleSource:
    """Polls the Arduino Mega over the HC-05 serial port."""

    name = "HC-05 serial"

    def __init__(self, ser, port: str):
        self.ser = ser
        self.port = port
        self._last_temp = float("nan")
        self._last_rh = float("nan")

    # ----- connection -----
    @staticmethod
    def candidate_ports() -> List[str]:
        if not SERIAL_AVAILABLE:
            return []
        scored: List[Tuple[int, str]] = []
        try:
            for p in list_ports.comports():
                text = f"{p.device} {p.description} {p.hwid}".lower()
                if any(k in text for k in ("hc-05", "hc05", "rfcomm")):
                    scored.append((0, p.device))
                elif any(k in text for k in ("bluetooth", "standard serial")):
                    scored.append((1, p.device))
                elif any(k in text for k in ("arduino", "ch340", "wch", "usb serial", "ttyacm", "ttyusb")):
                    scored.append((2, p.device))
        except Exception as exc:  # pragma: no cover
            LOG.debug("Port enumeration failed: %s", exc)
        try:
            import glob
            for pat in ("/dev/rfcomm*", "/dev/tty.HC-05*", "/dev/cu.HC-05*"):
                scored.extend((0, g) for g in glob.glob(pat))
        except Exception:  # pragma: no cover
            pass
        seen, ordered = set(), []
        for _, dev in sorted(scored):
            if dev not in seen:
                seen.add(dev)
                ordered.append(dev)
        return ordered

    @classmethod
    def auto_connect(cls, port: Optional[str], baud: int = DEFAULT_BAUD) -> Optional["SerialSampleSource"]:
        if not SERIAL_AVAILABLE:
            LOG.warning("pyserial is not installed (pip install pyserial).")
            return None
        candidates = [port] if port and port.lower() != "auto" else cls.candidate_ports()
        if not candidates:
            LOG.warning("No Bluetooth/serial candidates found.")
            return None
        for dev in candidates:
            try:
                ser = serial.Serial(dev, baudrate=baud, timeout=0.5, write_timeout=2.0)
            except (serial.SerialException, OSError, ValueError) as exc:
                LOG.warning("Cannot open %s: %s", dev, exc)
                continue
            src = cls(ser, dev)
            LOG.info("Opened %s @ %d baud, waiting for the soil node ...", dev, baud)
            if src._handshake():
                LOG.info("Soil node answered on %s.", dev)
                return src
            LOG.warning("No answer to PING on %s.", dev)
            src.close()
        return None

    def _handshake(self, tries: int = 4) -> bool:
        try:
            for _ in range(tries):
                self.ser.reset_input_buffer()
                self.ser.write(b"PING\n")
                self.ser.flush()
                deadline = time.monotonic() + 1.6
                while time.monotonic() < deadline:
                    raw = self.ser.readline()
                    if raw and b"PONG" in raw.upper():
                        return True
        except (serial.SerialException, OSError) as exc:
            LOG.warning("Serial error during handshake: %s", exc)
        return False

    # ----- sampling -----
    def read_sample(self, x: float, y: float) -> Optional[Sample]:
        for attempt in range(1, 4):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b"READ\n")
                self.ser.flush()
            except (serial.SerialException, OSError) as exc:
                LOG.error("Serial write failed: %s", exc)
                return None
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                try:
                    raw = self.ser.readline()
                except (serial.SerialException, OSError) as exc:
                    LOG.error("Serial read failed: %s", exc)
                    return None
                if not raw:
                    continue
                rec = parse_soil_line(raw.decode("ascii", errors="ignore"))
                if rec is None:
                    continue
                if math.isfinite(rec["temp"]) and math.isfinite(rec["rh"]):
                    self._last_temp, self._last_rh = rec["temp"], rec["rh"]
                temp = rec["temp"] if math.isfinite(rec["temp"]) else self._last_temp
                rh = rec["rh"] if math.isfinite(rec["rh"]) else self._last_rh
                if not rec["dht_ok"]:
                    LOG.warning("DHT11 flagged stale/invalid; using last known ambient values.")
                return Sample(x, y, rec["moisture"], temp, rh, rec["raw_mean"], "serial")
            LOG.warning("No valid reply from the node (attempt %d/3).", attempt)
        return None

    def close(self) -> None:
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:  # pragma: no cover
            pass


class SimulatedSampleSource:
    """Offline mode: a hidden, realistic moisture field + DHT11-like ambient readings."""

    name = "SIMULATION"

    def __init__(self, field: FieldSpec, texture: SoilTexture, seed: int = 7):
        self.field = field
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.base = float(r.uniform(46.0, 54.0))
        ang = float(r.uniform(0, 2 * math.pi))
        mag = float(r.uniform(8.0, 14.0))
        self.grad = (mag * math.cos(ang), mag * math.sin(ang))
        self.dry_c = r.uniform(0.15, 0.85, size=2)
        self.dry_amp = -float(r.uniform(38.0, 48.0))
        self.dry_sig = float(r.uniform(0.14, 0.22))
        self.wet_c = r.uniform(0.15, 0.85, size=2)
        self.wet_amp = float(r.uniform(38.0, 48.0))
        self.wet_sig = float(r.uniform(0.14, 0.22))
        self.t0 = float(r.uniform(27.0, 32.0))
        self.rh0 = float(r.uniform(48.0, 60.0))

    def truth_moisture(self, x, y):
        xn = np.asarray(x, dtype=float) / self.field.length
        yn = np.asarray(y, dtype=float) / self.field.width
        m = self.base + self.grad[0] * (xn - 0.5) + self.grad[1] * (yn - 0.5)
        m = m + self.dry_amp * np.exp(-((xn - self.dry_c[0]) ** 2 + (yn - self.dry_c[1]) ** 2) / (2 * self.dry_sig ** 2))
        m = m + self.wet_amp * np.exp(-((xn - self.wet_c[0]) ** 2 + (yn - self.wet_c[1]) ** 2) / (2 * self.wet_sig ** 2))
        return np.clip(m, 0.0, 100.0)

    def read_sample(self, x: float, y: float) -> Sample:
        m = float(self.truth_moisture(x, y)) + float(self.rng.normal(0.0, 1.0))
        yn = y / self.field.width
        temp = self.t0 + 1.2 * (yn - 0.5) + float(self.rng.normal(0.0, 0.4))
        rh = self.rh0 + 0.12 * (m - 50.0) + float(self.rng.normal(0.0, 1.0))
        # DHT11 resolution is 1 degC / 1 %RH
        return Sample(x, y, float(np.clip(m, 0.0, 100.0)), float(round(temp)), float(np.clip(round(rh), 0, 100)),
                      None, "simulation")

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# 3) Hybrid ML + residual IDW model
# --------------------------------------------------------------------------------------
def idw_interpolate(query_xy, sample_xy, values, power: float = 2.0, smoothing: float = 1e-6,
                    chunk: int = 20000) -> np.ndarray:
    """
    Smoothed inverse distance weighting:  w_i = 1 / (d_i^2 + s^2)^(p/2).
    The smoothing length s removes the singularity at the sample locations (no division by zero)
    and prevents bulls-eyes.  `values` may be (n,) or (n, k).
    """
    q = np.atleast_2d(np.asarray(query_xy, dtype=float))
    s = np.atleast_2d(np.asarray(sample_xy, dtype=float))
    v = np.asarray(values, dtype=float)
    one_d = v.ndim == 1
    if one_d:
        v = v[:, None]
    if q.ndim != 2 or q.shape[1] != 2 or s.ndim != 2 or s.shape[1] != 2:
        raise ValueError("Coordinates must have shape (m, 2) and (n, 2).")
    if s.shape[0] == 0:
        raise ValueError("IDW needs at least one sample point.")
    if v.shape[0] != s.shape[0]:
        raise ValueError(f"Dimension mismatch: {s.shape[0]} sample points but {v.shape[0]} values.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(s)):
        raise ValueError("Non-finite values passed to IDW.")
    smoothing = max(float(smoothing), 1e-9)
    power = max(float(power), 0.1)

    out = np.empty((q.shape[0], v.shape[1]), dtype=float)
    for a in range(0, q.shape[0], chunk):
        qq = q[a:a + chunk]
        d2 = np.sum((qq[:, None, :] - s[None, :, :]) ** 2, axis=2)
        w = (d2 + smoothing ** 2) ** (-power / 2.0)
        out[a:a + chunk] = (w @ v) / np.sum(w, axis=1, keepdims=True)
    return out[:, 0] if one_d else out


class HybridMoistureModel:
    """ML trend on environmental covariates + smoothed-IDW interpolation of the residuals."""

    DEFAULT_T = 27.0
    DEFAULT_RH = 55.0
    gp_ls_space = (1.0, 20.0)          # length-scale bounds (standardised units) for x/L, y/W, distance
    gp_ls_env = (5.0, 1000.0)          # ... and for T, RH, VPD
    gp_noise = (0.3, (0.15, 1.0))      # (initial, bounds) of the white-noise term (normalised target)

    def __init__(self, field: FieldSpec, model: str = "auto", idw_power: float = 2.0,
                 idw_smoothing_frac: float = 0.03, random_state: int = 42, gp_restarts: int = 3):
        if model not in {"auto", "gpr", "rf", "mean"}:
            raise ValueError("model must be one of: auto, gpr, rf")
        self.field = field
        self.model_choice = model
        self.idw_power = idw_power
        self.idw_smoothing_frac = idw_smoothing_frac
        self.random_state = random_state
        self.gp_restarts = gp_restarts
        self.kind: Optional[str] = None
        self.estimator = None
        self._ml_mean = 0.0
        self.samples: List[Sample] = []
        self.residuals = np.zeros(0)

    # ----- features -----
    def _features(self, xy: np.ndarray, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        L, W = self.field.length, self.field.width
        cx, cy = self.field.centroid
        dist = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) / max(self.field.half_diag, 1e-9)
        vpd = compute_vpd_kpa(temp, rh)
        return np.column_stack([xy[:, 0] / L, xy[:, 1] / W, dist, temp, rh, vpd])

    @staticmethod
    def _fill(arr: np.ndarray, default: float, lo: float, hi: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).copy()
        bad = ~np.isfinite(arr) | (arr < lo) | (arr > hi)
        if bad.all():
            arr[:] = default
        elif bad.any():
            arr[bad] = float(np.median(arr[~bad]))
        return arr

    # ----- estimator construction -----
    def _build(self, kind: str, n: int):
        if kind == "gpr":
            # ARD Matern: spatial features (x/L, y/W, dist) may use short length scales, the ambient
            # features (T, RH, VPD - nearly constant over one field) are forced to act only as gentle trends.
            ls_bounds = np.array([self.gp_ls_space] * 3 + [self.gp_ls_env] * 3, dtype=float)
            kernel = (C(1.0, (1e-2, 1e2))
                      * Matern(length_scale=np.array([1.5, 1.5, 1.5, 10.0, 10.0, 10.0]),
                               length_scale_bounds=ls_bounds, nu=1.5)
                      + WhiteKernel(noise_level=self.gp_noise[0], noise_level_bounds=self.gp_noise[1]))
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=self.gp_restarts,
                                           random_state=self.random_state)
            return make_pipeline(StandardScaler(), gpr)
        if kind == "rf":
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=1 if n < 6 else 2,
                                         random_state=self.random_state, n_jobs=1)
        raise ValueError(kind)

    def _fit_estimator(self, X: np.ndarray, y: np.ndarray) -> str:
        n = len(y)
        chosen = self.model_choice
        if chosen == "auto":
            chosen = "gpr" if n <= 12 else "rf"
        if n < 3:
            chosen = "mean"
        order = {"gpr": ["gpr", "rf", "mean"], "rf": ["rf", "mean"], "mean": ["mean"]}[chosen]
        for kind in order:
            if kind == "mean":
                self.estimator = None
                self._ml_mean = float(np.mean(y))
                if chosen != "mean":
                    LOG.warning("Falling back to a constant-mean trend.")
                elif n < 3:
                    LOG.warning("Fewer than 3 samples: ML trend replaced by the sample mean.")
                return "mean"
            try:
                est = self._build(kind, n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est.fit(X, y)
                    check = est.predict(X)
                if not np.all(np.isfinite(check)):
                    raise FloatingPointError("non-finite predictions")
                self.estimator = est
                return kind
            except Exception as exc:  # robust fallback chain
                LOG.warning("ML model '%s' failed (%s); trying fallback.", kind, exc)
        return "mean"  # pragma: no cover

    def _ml_predict(self, F: np.ndarray) -> np.ndarray:
        if self.kind == "mean" or self.estimator is None:
            return np.full(F.shape[0], self._ml_mean, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.estimator.predict(F), dtype=float)

    # ----- public API -----
    def fit(self, samples: Sequence[Sample]) -> "HybridMoistureModel":
        samples = list(samples)
        if not samples:
            raise ValueError("At least one valid sample is required to fit the model.")
        self.samples = samples
        xy = np.array([[s.x, s.y] for s in samples], dtype=float)
        y = np.clip(np.array([s.moisture for s in samples], dtype=float), 0.0, 100.0)
        t = self._fill(np.array([s.temp_c for s in samples]), self.DEFAULT_T, -30.0, 70.0)
        rh = self._fill(np.array([s.rh for s in samples]), self.DEFAULT_RH, 0.0, 100.0)
        if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(y))):
            raise ValueError("Samples contain non-finite coordinates or moisture values.")
        self._xy, self._y, self._t, self._rh = xy, y, t, rh
        self.idw_smoothing_m = max(self.idw_smoothing_frac * self.field.half_diag, 1e-6)

        X = self._features(xy, t, rh)
        self.kind = self._fit_estimator(X, y)
        self.residuals = y - self._ml_predict(X)          # Actual - Predicted
        return self

    def predict(self, xy_query) -> Dict[str, np.ndarray]:
        q = np.atleast_2d(np.asarray(xy_query, dtype=float))
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("Query coordinates must have shape (m, 2).")
        stacked = np.column_stack([self._t, self._rh, self.residuals])
        interp = idw_interpolate(q, self._xy, stacked, self.idw_power, self.idw_smoothing_m)
        t_q, rh_q, res_q = interp[:, 0], interp[:, 1], interp[:, 2]
        ml = self._ml_predict(self._features(q, t_q, rh_q))
        final = np.clip(ml + res_q, 0.0, 100.0)
        final = np.nan_to_num(final, nan=float(np.mean(self._y)))
        return {"final": final, "ml": ml, "residual": res_q, "temp": t_q, "rh": rh_q,
                "vpd": compute_vpd_kpa(t_q, rh_q)}

    def loo_cv(self) -> Optional[Dict[str, float]]:
        """Leave-one-out RMSE of the hybrid surface and of the ML trend alone (needs >= 4 samples)."""
        n = len(self.samples)
        if n < 4:
            return None
        e_h, e_m = [], []
        for i in range(n):
            sub = HybridMoistureModel(self.field, model=self.kind or "mean", idw_power=self.idw_power,
                                      idw_smoothing_frac=self.idw_smoothing_frac,
                                      random_state=self.random_state, gp_restarts=0)
            sub.fit(self.samples[:i] + self.samples[i + 1:])
            s = self.samples[i]
            out = sub.predict([[s.x, s.y]])
            e_h.append(float(out["final"][0] - s.moisture))
            e_m.append(float(out["ml"][0] - s.moisture))
        return {"rmse_hybrid": float(np.sqrt(np.mean(np.square(e_h)))),
                "rmse_ml": float(np.sqrt(np.mean(np.square(e_m))))}


# --------------------------------------------------------------------------------------
# 4) Discrete zoning
# --------------------------------------------------------------------------------------
def _odd(n: int) -> int:
    return int(n) | 1


def _remove_small_regions(labels: np.ndarray, min_cells: int) -> np.ndarray:
    """Re-assign connected components smaller than min_cells to the nearest surviving label."""
    small = np.zeros(labels.shape, dtype=bool)
    for z in np.unique(labels):
        cc, n = ndimage.label(labels == z)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        ids = np.where(sizes < min_cells)[0] + 1
        if ids.size:
            small |= np.isin(cc, ids)
    if not small.any() or small.all():
        return labels
    idx = ndimage.distance_transform_edt(small, return_distances=False, return_indices=True)
    return labels[tuple(idx)]


def _kmeans_zone_labels(smooth: np.ndarray, seed: int) -> np.ndarray:
    """K-Means on (moisture, x, y) -> clusters named by the agronomic class of their mean moisture."""
    ny, nx = smooth.shape
    k = int(min(4, len(np.unique(np.round(smooth, 3)))))
    if k < 2:
        return np.digitize(smooth, ZONE_BINS).astype(np.int16)
    yy, xx = np.mgrid[0:ny, 0:nx]
    feats = np.column_stack([smooth.ravel() / 100.0,
                             0.25 * xx.ravel() / max(nx - 1, 1),
                             0.25 * yy.ravel() / max(ny - 1, 1)])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
    flat = smooth.ravel()
    means = np.array([flat[km.labels_ == c].mean() if np.any(km.labels_ == c) else 50.0 for c in range(k)])
    zone_of_cluster = np.digitize(means, ZONE_BINS).astype(np.int16)
    return zone_of_cluster[km.labels_].reshape(ny, nx)


def delineate_zones(moisture: np.ndarray, method: str = "threshold", min_region_frac: float = 0.003,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (zone label map with values 0..3, smoothed moisture surface)."""
    if moisture.ndim != 2 or moisture.size == 0:
        raise ValueError("Moisture grid must be a non-empty 2-D array.")
    moisture = np.nan_to_num(moisture, nan=float(np.nanmean(moisture)) if np.isfinite(moisture).any() else 50.0)
    longest = max(moisture.shape)
    smooth_cells = max(3, _odd(round(0.03 * longest)))
    median_cells = max(3, _odd(round(0.04 * longest)))

    smooth = ndimage.uniform_filter(moisture, size=smooth_cells, mode="nearest")
    if method == "kmeans":
        labels = _kmeans_zone_labels(smooth, seed)
    elif method == "threshold":
        labels = np.digitize(smooth, ZONE_BINS).astype(np.int16)
    else:
        raise ValueError("zoning method must be 'threshold' or 'kmeans'")

    labels = ndimage.median_filter(labels, size=median_cells, mode="nearest")
    min_cells = max(4, int(min_region_frac * labels.size))
    for _ in range(2):
        labels = _remove_small_regions(labels, min_cells)
    return labels.astype(np.int16), smooth


def compute_zone_statistics(zones, moisture, vigor, field: FieldSpec, cell_area: float,
                            texture: SoilTexture, root_depth_m: float, efficiency: float) -> dict:
    """Area / coverage / water per zone.  1 mm of water on 1 m2 = 1 litre."""
    mm_per_pct = texture.porosity * root_depth_m * 10.0       # (pct/100) * porosity * depth_m * 1000
    efficiency = min(max(efficiency, 0.1), 1.0)
    rows, total_irrigation_l, total_net_l = [], 0.0, 0.0
    for z in ZONES:
        mask = zones == z.zid
        cnt = int(mask.sum())
        row = {"zid": z.zid, "area": cnt * cell_area, "pct": 100.0 * cnt / zones.size,
               "mean_moisture": float(moisture[mask].mean()) if cnt else float("nan"),
               "mean_vigor": float(vigor[mask].mean()) if cnt else float("nan"),
               "water_l": 0.0, "kind": "none", "depth_mm": 0.0}
        if cnt and z.zid in (0, 1):
            deficit = np.maximum(IRRIGATION_TARGET_PCT - moisture[mask], 0.0)
            net_mm = deficit * mm_per_pct
            net_l = float(net_mm.sum() * cell_area)
            gross_l = net_l / efficiency
            row.update(water_l=gross_l, kind="irrigate", depth_mm=float(net_mm.mean() / efficiency))
            total_irrigation_l += gross_l
            total_net_l += net_l
        elif cnt and z.zid == 3:
            excess = np.maximum(moisture[mask] - ZONE_BINS[2], 0.0)
            drain_mm = excess * mm_per_pct
            row.update(water_l=float(drain_mm.sum() * cell_area), kind="drain", depth_mm=float(drain_mm.mean()))
        rows.append(row)
    return {"zones": rows, "total_irrigation_l": total_irrigation_l, "total_net_l": total_net_l,
            "mean_depth_mm_field": total_irrigation_l / field.area, "efficiency": efficiency}


# --------------------------------------------------------------------------------------
# 5) Visualisation
# --------------------------------------------------------------------------------------
def plot_report(field: FieldSpec, xc, yc, zones, stats: dict, samples: Sequence[Sample],
                suggested: np.ndarray, info: dict, texture: SoilTexture, out_path: Optional[str],
                show: bool):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.25, 1.0],
                          left=0.055, right=0.985, top=0.90, bottom=0.17, wspace=0.10, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax_tab = fig.add_subplot(gs[0, 1])
    ax_inf = fig.add_subplot(gs[1, 1])

    cmap = ListedColormap([z.color for z in ZONES])
    ax.imshow(zones, origin="lower", extent=[0, field.length, 0, field.width], cmap=cmap,
              vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="equal", alpha=0.92)
    if len(np.unique(zones)) > 1:
        try:
            ax.contour(xc, yc, zones.astype(float), levels=[0.5, 1.5, 2.5], colors="black", linewidths=1.6)
        except Exception as exc:  # pragma: no cover
            LOG.debug("Contour failed: %s", exc)

    if suggested is not None and len(suggested):
        ax.scatter(suggested[:, 0], suggested[:, 1], marker="D", s=70, facecolors="none",
                   edgecolors="black", linewidths=1.2, zorder=4)
    sx = np.array([s.x for s in samples])
    sy = np.array([s.y for s in samples])
    ax.scatter(sx, sy, s=95, c="white", edgecolors="black", linewidths=1.6, zorder=5)
    for i, s in enumerate(samples, 1):
        ax.annotate(f"S{i}: {s.moisture:.0f}%", (s.x, s.y), xytext=(7, 7), textcoords="offset points",
                    fontsize=8, zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xlim(0, field.length)
    ax.set_ylim(0, field.width)
    ax.set_xlabel("Field length  X (m)")
    ax.set_ylabel("Field width  Y (m)")
    ax.set_title(f"Soil moisture management zones - {field.length:g} m x {field.width:g} m  |  {texture.label}",
                 fontsize=12, fontweight="bold")

    handles = [Patch(facecolor=z.color, edgecolor="black",
                     label=f"Zone {z.zid + 1}  {z.name} ({z.range_label}): {z.action}") for z in ZONES]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black",
                       markersize=9, label="Actual sample (S#: moisture %)"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor="none", markeredgecolor="black",
                       markersize=8, label="Suggested stratified spot")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5,
              frameon=True)

    # ---- zone table ----
    ax_tab.axis("off")
    col_labels = ["Zone", "Management action", "Area\n(m\u00b2)", "Cover\n(%)", "Water\n(L)"]
    rows = []
    for st in stats["zones"]:
        z = ZONES[st["zid"]]
        if st["kind"] == "irrigate":
            water = f"{st['water_l']:,.0f}"
        elif st["kind"] == "drain":
            water = f"drain\n~{st['water_l']:,.0f}"
        else:
            water = "-"
        rows.append([f"{z.zid + 1}. {z.name}\n({z.range_label})", textwrap.fill(z.action, 16),
                     f"{st['area']:,.0f}", f"{st['pct']:.1f}", water])
    rows.append(["Total field", "", f"{field.area:,.0f}", "100.0", f"{stats['total_irrigation_l']:,.0f}"])
    tbl = ax_tab.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
                       colWidths=[0.31, 0.27, 0.14, 0.11, 0.17])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 2.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif r <= len(ZONES) and c == 0:
            z = ZONES[r - 1]
            cell.set_facecolor(z.color)
            cell.set_text_props(color=z.text_color, fontweight="bold")
        elif r == len(rows):
            cell.set_facecolor("#e9e9e9")
            cell.set_text_props(fontweight="bold")
    ax_tab.set_title("Zone summary & irrigation water", fontsize=11, fontweight="bold")

    # ---- info box ----
    ax_inf.axis("off")
    loo = info.get("loo")
    loo_txt = (f"LOO-CV RMSE : hybrid {loo['rmse_hybrid']:.1f} % | ML only {loo['rmse_ml']:.1f} %"
               if loo else "LOO-CV RMSE : n/a (need >= 4 samples)")
    lines = [
        f"Samples       : {len(samples)}  (suggested {len(suggested)})   source: {info['source']}",
        f"Trend model   : {info['kind'].upper()} + residual IDW (p=2, smoothed)",
        loo_txt,
        f"Ambient (mean): {info['temp']:.1f} C, {info['rh']:.0f} % RH, VPD {info['vpd']:.2f} kPa",
        f"Mean moisture : {info['mean_moisture']:.1f} %   (range {info['min_moisture']:.0f}-{info['max_moisture']:.0f} %)",
        f"Vigor index   : {info['vigor']:.0f}/100  ({vigor_label(info['vigor'])})  [no pH probe: heuristic]",
        "",
        f"Irrigation    : {stats['total_irrigation_l']:,.0f} L gross  ({stats['total_irrigation_l'] / 1000:.1f} m\u00b3)",
        f"  = {stats['mean_depth_mm_field']:.1f} mm averaged over the whole field",
        f"  target {IRRIGATION_TARGET_PCT:g} % moisture, root depth {info['root_depth_m']:g} m, "
        f"efficiency {stats['efficiency'] * 100:.0f} %",
    ]
    ax_inf.text(0.0, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.6", fc="#f6f6f6", ec="#999999"))
    ax_inf.set_title("Model & field diagnostics", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle("AI-assisted handheld soil monitoring - irrigation zoning", fontsize=14, fontweight="bold")
    if out_path:
        try:
            fig.savefig(out_path, dpi=150)
            LOG.info("Figure saved to %s", out_path)
        except OSError as exc:
            LOG.error("Could not save figure to %s: %s", out_path, exc)
    if show:
        try:
            plt.show()
        except Exception as exc:  # pragma: no cover
            LOG.warning("Could not display the figure: %s", exc)
    return fig


# --------------------------------------------------------------------------------------
# Sample collection (automatic or interactive)
# --------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def sanitize_samples(samples: Sequence[Sample], field: FieldSpec) -> List[Sample]:
    clean: List[Sample] = []
    for i, s in enumerate(samples, 1):
        if not all(math.isfinite(v) for v in (s.x, s.y, s.moisture)):
            LOG.warning("Skipping sample %d: non-finite values.", i)
            continue
        x, y = field.clip_point(s.x, s.y)
        if (x, y) != (s.x, s.y):
            LOG.warning("Sample %d at (%.2f, %.2f) lies outside the field; clamped to (%.2f, %.2f).",
                        i, s.x, s.y, x, y)
            s = replace(s, x=x, y=y)
        clean.append(s)
    return clean


def collect_samples(source, field: FieldSpec, suggested: np.ndarray, interactive: bool,
                    wander_frac: float, rng: np.random.Generator) -> List[Sample]:
    samples: List[Sample] = []

    if not interactive:
        wander = wander_frac * min(field.length, field.width)
        for i, (sx, sy) in enumerate(suggested, 1):
            if isinstance(source, SimulatedSampleSource):
                x, y = field.clip_point(sx + rng.normal(0, wander), sy + rng.normal(0, wander))
            else:
                x, y = float(sx), float(sy)
            smp = source.read_sample(x, y)
            if smp is None:
                LOG.warning("Sample %d at (%.1f, %.1f) failed; skipping.", i, x, y)
                continue
            samples.append(smp)
            print(f"  sample {i:2d} @ ({x:6.1f}, {y:6.1f}) m -> {smp.moisture:5.1f} %  "
                  f"{smp.temp_c:4.0f} C  {smp.rh:3.0f} %RH")
        return samples

    print("\nInteractive sampling. At each prompt:")
    print("   Enter        = go to the suggested spot and sample there")
    print("   'x y'        = you are somewhere else: type your actual coordinates in metres (e.g. 12.5 7)")
    print("   s            = skip the suggested spot      q = finish and compute the map\n")
    idx = 0
    while len(samples) < HARD_MAX_SAMPLES:
        sug = suggested[idx] if idx < len(suggested) else None
        label = (f"[{len(samples) + 1}] suggested ({sug[0]:.1f}, {sug[1]:.1f}) m > " if sug is not None
                 else f"[{len(samples) + 1}] extra sample (x y) or q > ")
        try:
            line = input(label).strip()
        except EOFError:
            break
        low = line.lower()
        if low in {"q", "quit", "done", "finish", "exit"}:
            break
        if low in {"s", "skip"}:
            if sug is not None:
                idx += 1
            continue
        if line == "":
            if sug is None:
                print("  No more suggestions - type 'x y' or 'q'.")
                continue
            x, y = float(sug[0]), float(sug[1])
            idx += 1
        else:
            nums = _NUM_RE.findall(line)
            if len(nums) != 2:
                print("  Please enter exactly two numbers, e.g. '12.5 7', or Enter / s / q.")
                continue
            x, y = float(nums[0]), float(nums[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                print("  Coordinates must be finite numbers.")
                continue
            x2, y2 = field.clip_point(x, y)
            if (x2, y2) != (x, y):
                print(f"  Outside the field - clamped to ({x2:.1f}, {y2:.1f}).")
                x, y = x2, y2
            if sug is not None:
                idx += 1
        smp = source.read_sample(x, y)
        if smp is None:
            print("  Reading failed - try again.")
            continue
        samples.append(smp)
        print(f"  -> {smp.moisture:.1f} % moisture, {smp.temp_c:.0f} C, {smp.rh:.0f} % RH at ({x:.1f}, {y:.1f})")
    return samples


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------
def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:  # pragma: no cover
        return False


def _prompt_float(msg: str, default: float) -> float:
    while True:
        try:
            raw = input(f"{msg} [{default:g}]: ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            v = float(raw)
            if math.isfinite(v) and v > 0:
                return v
        except ValueError:
            pass
        print("  Please enter a positive number.")


def _prompt_texture(default: str = "sandy_loam") -> SoilTexture:
    keys = list(TEXTURES)
    print("Soil texture:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {TEXTURES[k].label}")
    while True:
        try:
            raw = input(f"Choose 1-{len(keys)} or name [{default}]: ").strip()
        except EOFError:
            return TEXTURES[default]
        if raw == "":
            return TEXTURES[default]
        try:
            if raw.isdigit() and 1 <= int(raw) <= len(keys):
                return TEXTURES[keys[int(raw) - 1]]
            return resolve_texture(raw)
        except ValueError as exc:
            print(f"  {exc}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-assisted handheld soil moisture zoning.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--length", type=float, default=None, help="field length X in metres (prompted if omitted in a terminal)")
    p.add_argument("--width", type=float, default=None, help="field width Y in metres (prompted if omitted in a terminal)")
    p.add_argument("--soil", type=str, default=None, help=f"soil texture: {', '.join(TEXTURES)}")
    p.add_argument("--samples", type=int, default=None, help="number of suggested sampling spots (5-10; auto by area)")
    p.add_argument("--port", type=str, default="auto", help="serial port of the HC-05 (COM5, /dev/rfcomm0, ...) or 'auto'")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="HC-05 baud rate")
    p.add_argument("--simulate", action="store_true", help="skip hardware and use the built-in simulator")
    p.add_argument("--interactive", action="store_true", help="enter the actual (x, y) of every sample by hand")
    p.add_argument("--model", choices=["auto", "gpr", "rf"], default="auto", help="baseline ML regressor")
    p.add_argument("--zoning", choices=["threshold", "kmeans"], default="threshold", help="zone delineation method")
    p.add_argument("--root-depth", type=float, default=0.30, help="effective root depth in metres")
    p.add_argument("--efficiency", type=float, default=0.85, help="irrigation application efficiency (0.1-1)")
    p.add_argument("--grid", type=int, default=200, help="grid cells along the longest field side")
    p.add_argument("--wander", type=float, default=0.06, help="simulation: farmer position scatter (fraction of short side)")
    p.add_argument("--seed", type=int, default=7, help="random seed (simulation / models)")
    p.add_argument("--out", type=str, default="soil_zones.png", help="output PNG path ('' to disable)")
    p.add_argument("--no-show", action="store_true", help="do not open a plot window")
    p.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def print_report(field: FieldSpec, texture: SoilTexture, stats: dict, info: dict) -> None:
    print("\n" + "=" * 78)
    print(f" FIELD {field.length:g} x {field.width:g} m  ({field.area:,.0f} m2)   soil: {texture.label}")
    print(f" model: {info['kind'].upper()} + residual IDW   samples: {info['n']}   source: {info['source']}")
    print("=" * 78)
    print(f" {'Zone':<30}{'Area m2':>10}{'Cover %':>9}{'Mean %':>8}{'Water L':>12}")
    for st in stats["zones"]:
        z = ZONES[st["zid"]]
        water = "-" if st["kind"] == "none" else (f"{st['water_l']:,.0f}" if st["kind"] == "irrigate"
                                                  else f"drain ~{st['water_l']:,.0f}")
        mm = "  n/a" if not math.isfinite(st["mean_moisture"]) else f"{st['mean_moisture']:6.1f}"
        print(f" {z.zid + 1}. {z.name + ' (' + z.range_label + ')':<27}{st['area']:>10,.0f}{st['pct']:>9.1f}{mm:>8}{water:>12}")
    print("-" * 78)
    print(f" Irrigation to apply: {stats['total_irrigation_l']:,.0f} L gross "
          f"({stats['total_irrigation_l'] / 1000:.1f} m3, {stats['mean_depth_mm_field']:.1f} mm field-average)")
    print(f" Vigor index: {info['vigor']:.0f}/100 ({vigor_label(info['vigor'])}) | "
          f"ambient {info['temp']:.1f} C, {info['rh']:.0f} %RH, VPD {info['vpd']:.2f} kPa")
    if info.get("loo"):
        print(f" Leave-one-out RMSE: hybrid {info['loo']['rmse_hybrid']:.2f} %, ML-only {info['loo']['rmse_ml']:.2f} %")
    if info.get("truth_rmse"):
        tr = info["truth_rmse"]
        print(f" Simulation check vs hidden truth (RMSE): hybrid {tr['hybrid']:.2f} %, ML-only {tr['ml']:.2f} %, "
              f"plain IDW {tr['idw']:.2f} %")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)-7s %(message)s")

    if args.list_ports:
        if not SERIAL_AVAILABLE:
            print("pyserial is not installed (pip install pyserial).")
            return 1
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports found.")
        for p in ports:
            print(f"{p.device:<18} {p.description}")
        return 0

    if args.no_show:
        matplotlib.use("Agg")

    tty = _stdin_is_tty()
    try:
        length, width = args.length, args.width
        if (length is None or width is None) and tty:
            print("Enter the rectangular field dimensions.")
            length = length if length is not None else _prompt_float("  Length (m)", 60.0)
            width = width if width is not None else _prompt_float("  Width  (m)", 40.0)
        field = FieldSpec(float(length if length is not None else 60.0), float(width if width is not None else 40.0))
        if args.soil is not None:
            texture = resolve_texture(args.soil)
        elif tty or args.interactive:
            texture = _prompt_texture()
        else:
            texture = TEXTURES["sandy_loam"]
        if not (0.05 <= args.root_depth <= 2.0):
            raise ValueError("--root-depth must be between 0.05 and 2.0 m.")
    except (ValueError, TypeError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130

    rng = np.random.default_rng(args.seed)
    suggested = recommend_sampling_points(field, args.samples)
    print(f"\nSuggested stratified sampling spots ({len(suggested)}, serpentine order):")
    for i, (sx, sy) in enumerate(suggested, 1):
        print(f"  {i:2d}. x = {sx:7.2f} m   y = {sy:7.2f} m")

    source = None
    if not args.simulate:
        source = SerialSampleSource.auto_connect(args.port, args.baud)
    if source is None:
        LOG.warning("HC-05 / soil node not available -> running in SIMULATION (offline) mode.")
        source = SimulatedSampleSource(field, texture, args.seed)
    else:
        LOG.info("Live data from %s.", source.port)

    interactive = args.interactive or (isinstance(source, SerialSampleSource) and tty)
    print(f"\nData source: {source.name}\n")
    try:
        raw_samples = collect_samples(source, field, suggested, interactive, args.wander, rng)
    except KeyboardInterrupt:
        print("\nSampling interrupted - using the samples collected so far.")
        raw_samples = []
    finally:
        source.close()

    samples = sanitize_samples(raw_samples, field)
    if not samples:
        print("No valid samples were collected - nothing to map.", file=sys.stderr)
        return 1

    # ---- model ----
    model = HybridMoistureModel(field, model=args.model, random_state=args.seed).fit(samples)
    loo = model.loo_cv()

    xc, yc, dx, dy = field.make_grid(args.grid)
    XX, YY = np.meshgrid(xc, yc)
    query = np.column_stack([XX.ravel(), YY.ravel()])
    pred = model.predict(query)
    moisture = pred["final"].reshape(XX.shape)
    vigor = vigor_index(moisture, pred["temp"].reshape(XX.shape), pred["vpd"].reshape(XX.shape), texture)

    zones, _ = delineate_zones(moisture, args.zoning, seed=args.seed)
    stats = compute_zone_statistics(zones, moisture, vigor, field, dx * dy, texture, args.root_depth, args.efficiency)

    info = {
        "kind": model.kind or "mean", "n": len(samples), "source": source.name, "loo": loo,
        "temp": float(np.mean(pred["temp"])), "rh": float(np.mean(pred["rh"])), "vpd": float(np.mean(pred["vpd"])),
        "mean_moisture": float(moisture.mean()), "min_moisture": float(moisture.min()),
        "max_moisture": float(moisture.max()), "vigor": float(vigor.mean()), "root_depth_m": args.root_depth,
        "truth_rmse": None,
    }
    if isinstance(source, SimulatedSampleSource):
        truth = source.truth_moisture(XX, YY)
        idw_only = idw_interpolate(query, model._xy, model._y, 2.0, model.idw_smoothing_m).reshape(XX.shape)
        info["truth_rmse"] = {
            "hybrid": float(np.sqrt(np.mean((moisture - truth) ** 2))),
            "ml": float(np.sqrt(np.mean((np.clip(pred["ml"], 0, 100).reshape(XX.shape) - truth) ** 2))),
            "idw": float(np.sqrt(np.mean((idw_only - truth) ** 2))),
        }

    print_report(field, texture, stats, info)
    plot_report(field, xc, yc, zones, stats, samples, suggested, info, texture, args.out or None,
                show=not args.no_show)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
