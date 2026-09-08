import json
import os
import time
import io
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from wrf import WrfFile, getvar, interplevel

# ==========================================
# 1. POMOCNÉ & VYPOČETNÍ FUNKCE
# ==========================================

def safe_savefig(fig, out_file, retries=15):
    """Renderuje obrázek v RAM a zapisuje na disk s ošetřením souborových zámků ve Windows."""
    buf = io.BytesIO()
    fig.savefig(buf, format="webp", dpi=140)
    img_bytes = buf.getvalue()
    buf.close()

    for i in range(retries):
        try:
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            return
        except (OSError, PermissionError):
            time.sleep(0.1)

    try:
        tmp_file = out_file.with_suffix(f".tmp_{time.time_ns()}")
        with open(tmp_file, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp_file, out_file)
    except Exception as e:
        print(f"Chyba při zápisu {out_file}: {e}")

def load_custom_cmap(filepath):
    """Načte .ct soubor jako přesnou diskrétní ListedColormap odpovídající kroku v .ct file."""
    if not filepath.exists():
        print(f"Varování: Soubor {filepath.name} neexistuje!")
        return None, None
    vals, colors = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vals.append(float(parts[0]))
                    colors.append((float(parts[1])/255.0, float(parts[2])/255.0, float(parts[3])/255.0))
                except ValueError:
                    continue
    if not vals or len(vals) < 2:
        return None, None

    vals = np.array(vals)
    
    cmap = mcolors.ListedColormap(colors[:-1], name=filepath.stem)
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])

    return cmap, vals

def calc_pressure_level(f, step, level_hpa, var_type):
    p = getvar(f, "pressure", timeidx=step)
    if var_type == "wind":
        u = getvar(f, "ua", timeidx=step)
        v = getvar(f, "va", timeidx=step)
        wspd_kmh = np.sqrt(u**2 + v**2) * 3.6
        return interplevel(wspd_kmh, p, level_hpa)
    elif var_type == "temp":
        tc = getvar(f, "tc", timeidx=step)
        return interplevel(tc, p, level_hpa)

def calc_max_refl10cm(f, step):
    try:
        refl = getvar(f, "REFL_10CM", timeidx=step)
        max_dbz = np.max(np.asarray(refl), axis=0)
    except Exception:
        max_dbz = getvar(f, "mdbz", timeidx=step)

    smoothed = gaussian_filter(max_dbz.astype(np.float32), sigma=0.3, mode='nearest')
    return np.where(smoothed < 4.0, np.nan, smoothed) 

def calc_sirs(f, step):
    try:
        t_surf_c = getvar(f, "TSK", timeidx=step) - 273.15
    except Exception:
        t_surf_c = getvar(f, "T2", timeidx=step) - 273.15

    try:
        qc = np.maximum(getvar(f, "QCLOUD", timeidx=step), 0.0)
        qi = np.maximum(getvar(f, "QICE", timeidx=step), 0.0)
        qs = np.maximum(getvar(f, "QSNOW", timeidx=step), 0.0)
        q_total = qc + qi + qs
        t_c = getvar(f, "tc", timeidx=step)

        has_cloud = q_total >= 1.0e-5
        cloud_any = has_cloud.any(axis=0)

        if not np.any(cloud_any):
            return t_surf_c

        q_total_rev = q_total[::-1, :, :]
        t_c_rev = t_c[::-1, :, :]
        has_cloud_rev = q_total_rev >= 1.0e-5

        top_k_rev = has_cloud_rev.argmax(axis=0)
        ny, nx = cloud_any.shape
        grid_y, grid_x = np.ogrid[:ny, :nx]

        cloud_top_temp = t_c_rev[top_k_rev, grid_y, grid_x]

        t_c_cloud_only = np.where(has_cloud, t_c, np.inf)
        cloud_min_temp = np.min(t_c_cloud_only, axis=0)
        cloud_top_temp = np.minimum(cloud_top_temp, cloud_min_temp)

        return np.where(cloud_any, cloud_top_temp, t_surf_c)
    except Exception:
        return t_surf_c

def calc_total_precip(f, step):
    accum = None
    for name in ('RAINNC', 'RAINC', 'RAINSH'):
        try:
            var = getvar(f, name, timeidx=step)
            arr = np.asarray(var, dtype=np.float32)
            accum = arr if accum is None else accum + arr
        except Exception:
            continue
    return accum if accum is not None else np.zeros_like(getvar(f, "XLAT", timeidx=step))

def calc_delta_precip(f, step, hours_back):
    current_dt = datetime_times[step]
    target_dt = current_dt - timedelta(hours=hours_back)
    
    past_step = 0
    for i in range(step, -1, -1):
        if datetime_times[i] <= target_dt:
            past_step = i
            break
            
    current_p = calc_total_precip(f, step)
    past_p = calc_total_precip(f, past_step)
    
    delta = current_p - past_p
    return np.maximum(delta, 0.0)

def calc_bulk_shear(f, step, target_m):
    u10 = getvar(f, "U10", timeidx=step)
    v10 = getvar(f, "V10", timeidx=step)
    try:
        uvmet = getvar(f, "uvmet", timeidx=step)
        u3d, v3d = uvmet[0], uvmet[1]
    except Exception:
        u3d = getvar(f, "ua", timeidx=step)
        v3d = getvar(f, "va", timeidx=step)

    z3d = getvar(f, "z", timeidx=step)
    ter = getvar(f, "ter", timeidx=step)
    zagl = z3d - ter

    nz, ny, nx = zagl.shape
    idx = np.sum(zagl < target_m, axis=0)
    idx = np.clip(idx, 1, nz - 1)
    y_grid, x_grid = np.ogrid[:ny, :nx]

    z0 = zagl[idx - 1, y_grid, x_grid]
    z1 = zagl[idx, y_grid, x_grid]
    dz = np.maximum(z1 - z0, 1e-5)
    frac = np.clip((target_m - z0) / dz, 0.0, 1.0)

    u_target = u3d[idx - 1, y_grid, x_grid] + frac * (u3d[idx, y_grid, x_grid] - u3d[idx - 1, y_grid, x_grid])
    v_target = v3d[idx - 1, y_grid, x_grid] + frac * (v3d[idx, y_grid, x_grid] - v3d[idx - 1, y_grid, x_grid])

    return np.sqrt((u_target - u10)**2 + (v_target - v10)**2)

def calc_ref_level(f, step, target_m=2000.0):
    try:
        dbz = getvar(f, "REFL_10CM", timeidx=step)
    except Exception:
        dbz = getvar(f, "dbz", timeidx=step)

    z3d = getvar(f, "z", timeidx=step)
    ter = getvar(f, "ter", timeidx=step)
    zagl = z3d - ter

    nz, ny, nx = zagl.shape
    idx = np.sum(zagl < target_m, axis=0)
    idx = np.clip(idx, 1, nz - 1)
    y_grid, x_grid = np.ogrid[:ny, :nx]

    z0 = zagl[idx - 1, y_grid, x_grid]
    z1 = zagl[idx, y_grid, x_grid]
    dz = np.maximum(z1 - z0, 1e-5)
    frac = np.clip((target_m - z0) / dz, 0.0, 1.0)

    raw_2km = dbz[idx - 1, y_grid, x_grid] + frac * (dbz[idx, y_grid, x_grid] - dbz[idx - 1, y_grid, x_grid])
    smoothed = gaussian_filter(np.asarray(raw_2km, dtype=np.float32), sigma=0.3, mode='nearest')
    return np.where(smoothed < 4.0, np.nan, smoothed)

def calc_echotops(f, step, threshold_dbz=18.0):
    try:
        refl = getvar(f, "REFL_10CM", timeidx=step)
    except Exception:
        refl = getvar(f, "dbz", timeidx=step)

    z3d = getvar(f, "z", timeidx=step)
    ter = getvar(f, "ter", timeidx=step)
    z_agl_km = (z3d - ter) / 1000.0

    refl_arr = np.asarray(refl)
    z_agl_arr = np.asarray(z_agl_km)

    nz, ny, nx = refl_arr.shape
    mask = refl_arr >= threshold_dbz
    mask_rev = mask[::-1, :, :]
    has_echo = mask_rev.any(axis=0)

    top_k_rev = mask_rev.argmax(axis=0)
    top_k = (nz - 1) - top_k_rev

    grid_y, grid_x = np.ogrid[:ny, :nx]
    echo_tops_km = z_agl_arr[top_k, grid_y, grid_x]

    return np.where(has_echo, echo_tops_km, np.nan)

def calc_lightning(f, step):
    try:
        qc = np.maximum(np.asarray(getvar(f, "QCLOUD", timeidx=step), dtype=np.float32), 0.0)
        qr = np.maximum(np.asarray(getvar(f, "QRAIN", timeidx=step), dtype=np.float32), 0.0)
        qi = np.maximum(np.asarray(getvar(f, "QICE", timeidx=step), dtype=np.float32), 0.0)
        qs = np.maximum(np.asarray(getvar(f, "QSNOW", timeidx=step), dtype=np.float32), 0.0)
        qg = np.maximum(np.asarray(getvar(f, "QGRAUP", timeidx=step), dtype=np.float32), 0.0)
        temp_c = np.asarray(getvar(f, "tc", timeidx=step), dtype=np.float32)
        height = np.asarray(getvar(f, "z", timeidx=step), dtype=np.float32)
        w = np.asarray(getvar(f, "wa", timeidx=step), dtype=np.float32)
        mdbz = np.nan_to_num(np.asarray(getvar(f, "mdbz", timeidx=step), dtype=np.float32), nan=0.0)
        dbz = np.nan_to_num(np.asarray(getvar(f, "dbz", timeidx=step), dtype=np.float32), nan=0.0)

        q_ice_dense = qi + qs + qg
        in_cloud_core = q_ice_dense > 1.0e-4
        temp_cloud = np.where(in_cloud_core, temp_c, np.inf)
        t_top = np.min(temp_cloud, axis=0)
        has_cloud = np.any(in_cloud_core, axis=0)
        t_top = np.where(has_cloud, t_top, 0.0)

        f_depth = np.clip((-18.0 - t_top) / 40.0, 0.0, 1.0) ** 2.0
        q_liq = qc + qr
        finite = np.isfinite(temp_c) & np.isfinite(height) & np.isfinite(w) & np.isfinite(q_liq) & np.isfinite(qg)
        charge = finite & (temp_c <= -10.0) & (temp_c >= -25.0) & (w > 3.0) & (qg > 2.0e-4)

        tiny = 1.0e-20
        qi_qg = qi + qg
        qs_qg = qs + qg

        with np.errstate(divide='ignore', invalid='ignore'):
            interaction_i = np.where(qi_qg > tiny, np.sqrt(np.maximum(qi * qg, 0.0)) / qi_qg, 0.0)
            interaction_s = np.where(qs_qg > tiny, np.sqrt(np.maximum(qs * qg, 0.0)) / qs_qg, 0.0)

        q_frozen_eff = qg * (interaction_i + interaction_s)
        denom = q_liq + q_frozen_eff

        with np.errstate(divide="ignore", invalid="ignore"):
            epsilon = np.divide(2.0 * np.sqrt(np.maximum(q_liq * q_frozen_eff, 0.0)), denom, out=np.zeros_like(denom, dtype=np.float32), where=denom > tiny)

        epsilon = np.clip(np.nan_to_num(epsilon, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        w_eff = 8.0 * np.tanh(np.maximum(w, 0.0) / 8.0)
        integrand = np.where(charge, epsilon * np.square(w_eff), 0.0)

        z0, z1 = height[:-1], height[1:]
        dz = np.maximum(z1 - z0, 0.0)
        seg_ok = charge[:-1] & charge[1:] & (dz > 0.0)
        seg_dz = np.where(seg_ok, dz, 0.0)

        lpi_int = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * seg_dz, axis=0, dtype=np.float64)
        layer_thickness = np.sum(seg_dz, axis=0, dtype=np.float64)

        lpi_spec = np.zeros_like(lpi_int, dtype=np.float64)
        np.divide(lpi_int, layer_thickness, out=lpi_spec, where=layer_thickness > 10.0)
        lpi_spec = np.nan_to_num(lpi_spec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        has_reflection_core = np.any((dbz >= 35.0) & (temp_c <= -10.0), axis=0)
        dz_full = np.zeros_like(qg)
        dz_full[:-1] = dz
        graupel_path = np.sum(qg * dz_full, axis=0)
        has_min_graupel = graupel_path > 0.12

        valid_cell = has_reflection_core & has_min_graupel & (t_top <= -15.0)
        calibrated = 1.8 * np.power(np.maximum(lpi_spec, 0.0), 1.1) * f_depth * np.clip(mdbz / 45.0, 0.0, 1.0)
        calibrated = np.where(valid_cell, calibrated, 0.0)

        smoothed = gaussian_filter(calibrated.astype(np.float32), sigma=0.6, mode="nearest")
        final_rate = np.where(smoothed < 0.1, 0.0, smoothed)
        return np.clip(final_rate, 0.0, 720.0).astype(np.float32)
    except Exception as e:
        print(f"[LIGHTNING ERROR] step={step}: {e}")
        return np.zeros_like(getvar(f, "XLAT", timeidx=step), dtype=np.float32)

# ==========================================
# 2. NASTAVENÍ SLOŽEK A VSTUPŮ
# ==========================================
BASE_DIR = Path(r"M:\wrfouts\WRFWEB")
WRF_DIR = BASE_DIR / "wrfvystupy"
CT_DIR = BASE_DIR / "colortables"
WEB_DATA_DIR = BASE_DIR / "web_data"

wrf_file = WRF_DIR / "wrfout_d01_2026-07-14_00-00-00.nc"
run_id = "20260714_00z"
output_dir = WEB_DATA_DIR / run_id

# Načtení colortables
cape_cmap, cape_levels = load_custom_cmap(CT_DIR / "cape2.ct")
temp_cmap, temp_levels = load_custom_cmap(CT_DIR / "Temperature.ct")
radar_cmap, radar_levels = load_custom_cmap(CT_DIR / "czrad.ct")
mslp_cmap, mslp_levels = load_custom_cmap(CT_DIR / "tlak.ct")
wind_cmap, wind_levels = load_custom_cmap(CT_DIR / "wind.ct")
sirs_cmap, sirs_levels = load_custom_cmap(CT_DIR / "SIRS1.ct")
shear_cmap, raw_shear_levels = load_custom_cmap(CT_DIR / "Wind_Gust.ct")
echo_cmap, echo_levels = load_custom_cmap(CT_DIR / "echotops.ct")
light_cmap, light_levels = load_custom_cmap(CT_DIR / "lightning.ct")
precip_cmap, precip_levels = load_custom_cmap(CT_DIR / "precip.ct")

if raw_shear_levels is not None:
    shear_levels = np.linspace(0.0, raw_shear_levels.max(), len(raw_shear_levels))
else:
    shear_levels = np.linspace(0.0, 75.0, 60)

f_init = WrfFile(str(wrf_file))
total_steps = f_init.nt

lats = getvar(f_init, "XLAT", timeidx=0)
lons = getvar(f_init, "XLONG", timeidx=0)

print("Načítám časové značky...")
formatted_times = []
datetime_times = []

for step in range(total_steps):
    t_obj = getvar(f_init, "times", timeidx=step)
    t_str = str(t_obj)[:19].replace("_", " ").replace("T", " ")[:16]
    formatted_times.append(t_str)
    dt_full_str = str(t_obj)[:19].replace("T", " ").replace("_", " ")
    datetime_times.append(datetime.strptime(dt_full_str, "%Y-%m-%d %H:%M:%S"))

# ==========================================
# 3. KONFIGURACE VŠECH DATOVÝCH POLÍ
# ==========================================
variables_config = {
    # --- POVRCH & SYNOPTIKA ---
    "temp": {
        "title": "Teplota ve 2m",
        "get_data": lambda f, step: getvar(f, "T2", timeidx=step) - 273.15,
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "extend": "both"
    },
    "dewpoint": {
        "title": "Rosný bod ve 2m",
        "get_data": lambda f, step: getvar(f, "td2", timeidx=step),
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "extend": "both"
    },
    "mslp": {
        "title": "MSLP (Tlak)",
        "get_data": lambda f, step: getvar(f, "slp", timeidx=step),
        "cmap": mslp_cmap,
        "levels": mslp_levels,
        "label": "hPa",
        "ticks": [940, 955, 970, 986, 1001, 1016, 1031, 1046],
        "extend": "both"
    },
    "wind10m": {
        "title": "Vítr v 10m",
        "get_data": lambda f, step: np.sqrt(getvar(f, "U10", timeidx=step)**2 + getvar(f, "V10", timeidx=step)**2) * 3.6,
        "cmap": wind_cmap,
        "levels": wind_levels,
        "label": "km/h",
        "ticks": [0, 20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 280],
        "extend": "max"
    },
    "sat_ir": {
        "title": "Simulovaný IR Satelit",
        "get_data": lambda f, step: calc_sirs(f, step),
        "cmap": sirs_cmap,
        "levels": sirs_levels,
        "label": "°C",
        "ticks": [-90, -80, -70, -65, -60, -55, -50, -45, -40, -30, -10, 0, 10, 30, 50],
        "extend": "both"
    },

    # --- INSTABILITA & STŘIH VĚTRU ---
    "cape": {
        "title": "MUCAPE",
        "get_data": lambda f, step: getvar(f, "cape_2d", timeidx=step)[0],
        "cmap": cape_cmap,
        "levels": cape_levels,
        "label": "J/kg",
        "ticks": [0, 300, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000],
        "extend": "max"
    },
    "cin": {
        "title": "MUCIN",
        "get_data": lambda f, step: np.abs(getvar(f, "cape_2d", timeidx=step)[1]),
        "cmap": "Blues",
        "levels": [0, 10, 25, 50, 100, 150, 200, 300, 400, 500],
        "label": "J/kg",
        "ticks": [0, 25, 50, 100, 200, 300, 500],
        "extend": "max"
    },
    "shear_0_1km": {
        "title": "Bulk Shear 0-1 km",
        "get_data": lambda f, step: calc_bulk_shear(f, step, 1000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },
    "shear_0_3km": {
        "title": "Bulk Shear 0-3 km",
        "get_data": lambda f, step: calc_bulk_shear(f, step, 3000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },
    "shear_0_6km": {
        "title": "Bulk Shear 0-6 km",
        "get_data": lambda f, step: calc_bulk_shear(f, step, 6000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },

    # --- RADAR & KONVEKCE ---
    "radar_max": {
        "title": "Simulovaný radar (Max dBZ)",
        "get_data": lambda f, step: calc_max_refl10cm(f, step),
        "cmap": radar_cmap,
        "levels": radar_levels,
        "label": "dBZ",
        "ticks": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
        "extend": "max"
    },
    "radar_2km": {
        "title": "Odrazivost ve 2 km AGL",
        "get_data": lambda f, step: calc_ref_level(f, step, 2000.0),
        "cmap": radar_cmap,
        "levels": radar_levels,
        "label": "dBZ",
        "ticks": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
        "extend": "max"
    },
    "echotops_18": {
        "title": "Echo Tops 18 dBZ",
        "get_data": lambda f, step: calc_echotops(f, step, 18.0),
        "cmap": echo_cmap,
        "levels": echo_levels,
        "label": "km AGL",
        "ticks": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        "extend": "max"
    },
    "echotops_45": {
        "title": "Echo Tops 45 dBZ",
        "get_data": lambda f, step: calc_echotops(f, step, 45.0),
        "cmap": echo_cmap,
        "levels": echo_levels,
        "label": "km AGL",
        "ticks": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        "extend": "max"
    },
    "lightning": {
        "title": "Blesková aktivita (LPI)",
        "get_data": lambda f, step: calc_lightning(f, step),
        "cmap": light_cmap,
        "levels": light_levels,
        "label": "flashes/h",
        "ticks": [0.1, 1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 360, 720],
        "extend": "max"
    },

    # --- SRÁŽKY ---
    "precip_total": {
        "title": "Celkové srážky",
        "get_data": lambda f, step: calc_total_precip(f, step),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_1h": {
        "title": "Srážky za 1h",
        "get_data": lambda f, step: calc_delta_precip(f, step, 1.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_6h": {
        "title": "Srážky za 6h",
        "get_data": lambda f, step: calc_delta_precip(f, step, 6.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_24h": {
        "title": "Srážky za 24h",
        "get_data": lambda f, step: calc_delta_precip(f, step, 24.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
}

UPPER_LEVELS = [925, 850, 700, 500, 400, 300, 200]

for p_lev in UPPER_LEVELS:
    variables_config[f"wind_{p_lev}hpa"] = {
        "title": f"Vítr v {p_lev} hPa",
        "get_data": lambda f, step, p=p_lev: calc_pressure_level(f, step, p, "wind"),
        "cmap": wind_cmap,
        "levels": wind_levels,
        "label": "km/h",
        "ticks": [0, 20, 40, 60, 80, 100, 130, 160, 200, 240, 280],
        "extend": "max"
    }
    variables_config[f"temp_{p_lev}hpa"] = {
        "title": f"Teplota v {p_lev} hPa",
        "get_data": lambda f, step, p=p_lev: calc_pressure_level(f, step, p, "temp"),
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30],
        "extend": "both"
    }

for var_name in variables_config.keys():
    (output_dir / var_name).mkdir(parents=True, exist_ok=True)

cart_proj = ccrs.PlateCarree()

# ==========================================
# 4. FUNKCE PRO PARALELNÍ ZPRACOVÁNÍ KROKU
# ==========================================
def process_single_step(step):
    f_thread = WrfFile(str(wrf_file))
    time_str = formatted_times[step]
    init_str = formatted_times[0]
    
    dt_init = datetime_times[0]
    dt_valid = datetime_times[step]
    f_hour = int((dt_valid - dt_init).total_seconds() / 3600)
    
    local_max_cape = 0.0

    for var_key, cfg in variables_config.items():
        data_field = cfg["get_data"](f_thread, step)

        if var_key == "cape":
            local_max_cape = float(np.nanmax(data_field))

        fig = plt.figure(figsize=(10.128, 5.7), facecolor='#0f172a')
        
        ax = fig.add_axes([0.02, 0.03, 0.82, 0.88], projection=cart_proj)
        ax.set_facecolor('#1e293b')

        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#94a3b8')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#94a3b8')

        levels = cfg["levels"]
        cmap = cfg["cmap"]

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

        # Načtení konkrétního nastavení extend pro pole (default "max")
        extend_mode = cfg.get("extend", "max")

        cf = ax.contourf(
            lons, lats, data_field,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=cart_proj,
            extend=extend_mode
        )

        if var_key == "mslp":
            iso_levels = np.arange(940, 1060, 2)
            cs = ax.contour(lons, lats, data_field, levels=iso_levels, colors='#e2e8f0', linewidths=0.6, transform=cart_proj)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%d hPa', colors='#ffffff')

        cbar_ax = fig.add_axes([0.85, 0.03, 0.025, 0.88])
        cbar = fig.colorbar(
            cf, cax=cbar_ax,
            ticks=cfg["ticks"],
            spacing='uniform'
        )
        cbar.ax.tick_params(labelsize=9, colors='#f8fafc')
        cbar.set_label(cfg["label"], color='#f8fafc', fontsize=10, fontweight='bold')

        banner_ax = fig.add_axes([0.02, 0.92, 0.955, 0.06], facecolor='#1e293b')
        banner_ax.axis('off')

        banner_ax.text(0.01, 0.5, "WRF ARW 3km", color='#38bdf8', fontsize=10, fontweight='bold', va='center')
        banner_ax.text(0.14, 0.5, f"|  {cfg['title']}", color='#ffffff', fontsize=10, fontweight='bold', va='center')

        time_text = f"Init: {init_str} UTC  |  Valid: {time_str} UTC (+{f_hour:02d}h)"
        banner_ax.text(0.99, 0.5, time_text, color='#f1f5f9', fontsize=8.8, va='center', ha='right', family='monospace')

        out_file = output_dir / var_key / f"{step:03d}.webp"
        safe_savefig(fig, out_file)
        plt.clf()
        plt.close(fig)

    plt.close('all')
    print(f"Done [{step + 1}/{total_steps}]: {time_str}")
    return local_max_cape

# ==========================================
# 5. SPUŠTĚNÍ A METADATA
# ==========================================
if __name__ == "__main__":
    max_workers = min(8, os.cpu_count())
    print(f"Start paralelního zpracování '{run_id}' na {max_workers} worker procesech...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        cape_results = list(executor.map(process_single_step, range(total_steps)))

    max_cape_found = max(cape_results) if cape_results else 0.0

    run_metadata = {
        "run_id": run_id,
        "model": "WRF ARW",
        "init_time": formatted_times[0],
        "end_time": formatted_times[-1],
        "total_steps": total_steps,
        "max_cape": round(max_cape_found, 1),
        "time_steps": formatted_times,
        "fields": list(variables_config.keys())
    }

    json_path = WEB_DATA_DIR / "runs.json"
    all_runs = []

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                all_runs = json.load(jf)
        except Exception:
            all_runs = []

    all_runs = [r for r in all_runs if r.get("run_id") != run_id]
    all_runs.insert(0, run_metadata)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_runs, jf, indent=2, ensure_ascii=False)

    print("\n--- ZPRACOVÁNÍ DOKONČENO ---")
    print(f"Metadata zapsána do: {json_path}")