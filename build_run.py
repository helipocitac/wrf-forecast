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
import matplotlib.patheffects as mpatheffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from wrf import WrfFile, getvar, interplevel
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ==========================================
# 1. POMOCNÉ & VYPOČETNÍ FUNKCE (S KEŠOVÁNÍM)
# ==========================================

def safe_savefig(fig, out_file, retries=20):
    """Renderuje obrázek v RAM a zapíše na disk s ošetřením zámků Windows a úklidem .tmp."""
    buf = io.BytesIO()
    fig.savefig(buf, format="webp", dpi=140)
    img_bytes = buf.getvalue()
    buf.close()

    for _ in range(retries):
        try:
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            return
        except (OSError, PermissionError):
            time.sleep(0.1)

    tmp_file = out_file.with_suffix(f".tmp_{time.time_ns()}")
    try:
        with open(tmp_file, "wb") as f:
            f.write(img_bytes)
        
        for _ in range(retries):
            try:
                if out_file.exists():
                    try:
                        out_file.unlink()
                    except Exception:
                        pass
                os.replace(tmp_file, out_file)
                return
            except (OSError, PermissionError):
                time.sleep(0.15)
    except Exception as e:
        print(f"Chyba při zápisu {out_file.name}: {e}")
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass

def load_custom_cmap(filepath):
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

def get_cached(f, step, cache, var_name):
    """Zajistí, že se každá 3D/2D proměnná z WRF načte v daném čase maximálně JEDNOU."""
    if var_name not in cache:
        cache[var_name] = getvar(f, var_name, timeidx=step)
    return cache[var_name]

def draw_value_grid(ax, lons, lats, data, num_x=12, num_y=7, fmt="{:.0f}"):
    data_arr = np.asarray(data)
    lons_arr = np.asarray(lons)
    lats_arr = np.asarray(lats)
    ny, nx = data_arr.shape

    y_indices = np.linspace(int(ny * 0.08), int(ny * 0.92), num_y, dtype=int)
    x_indices = np.linspace(int(nx * 0.08), int(nx * 0.92), num_x, dtype=int)

    for iy in y_indices:
        for ix in x_indices:
            val = data_arr[iy, ix]
            if not np.isfinite(val):
                continue

            lon_val = lons_arr[iy, ix]
            lat_val = lats_arr[iy, ix]

            txt = ax.text(
                lon_val, lat_val, fmt.format(val),
                transform=ccrs.PlateCarree(),
                fontsize=7.5,
                fontweight='bold',
                ha='center', va='center',
                color='#0f172a',
                zorder=30
            )
            txt.set_path_effects([
                mpatheffects.withStroke(linewidth=1.8, foreground='#ffffff', alpha=0.9)
            ])

def calc_pressure_uv(f, step, cache, level_hpa):
    p = get_cached(f, step, cache, "pressure")
    u = get_cached(f, step, cache, "ua")
    v = get_cached(f, step, cache, "va")
    u_p = interplevel(u, p, level_hpa)
    v_p = interplevel(v, p, level_hpa)
    return u_p, v_p

def draw_streamlines(ax, lons, lats, u, v, color='white', linewidth=0.8, density=1.2):
    """Vykreslí husté plynulé proudnice větru (streamlines)."""
    u_arr = np.squeeze(np.asarray(u, dtype=np.float32))
    v_arr = np.squeeze(np.asarray(v, dtype=np.float32))
    lat_arr = np.squeeze(np.asarray(lats, dtype=np.float32))
    lon_arr = np.squeeze(np.asarray(lons, dtype=np.float32))

    u_arr = np.nan_to_num(u_arr, nan=0.0)
    v_arr = np.nan_to_num(v_arr, nan=0.0)

    ny, nx = u_arr.shape
    stride = 2 if nx > 150 else 1

    u_sub = u_arr[::stride, ::stride]
    v_sub = v_arr[::stride, ::stride]
    lat_sub = lat_arr[::stride, ::stride]
    lon_sub = lon_arr[::stride, ::stride]

    sub_ny, sub_nx = u_sub.shape
    if sub_ny < 2 or sub_nx < 2:
        return

    x_1d = np.linspace(float(np.nanmin(lon_sub)), float(np.nanmax(lon_sub)), sub_nx)
    y_1d = np.linspace(float(np.nanmin(lat_sub)), float(np.nanmax(lat_sub)), sub_ny)

    try:
        ax.streamplot(
            x_1d, y_1d, u_sub, v_sub,
            transform=ccrs.PlateCarree(),
            color=color,
            linewidth=linewidth,
            density=density, # Ponechána vysoká hustota podle přání
            arrowstyle='->',
            arrowsize=1.2,
            minlength=0.2,
            zorder=25
        )
    except Exception as e:
        print(f"[STREAMLINE ERROR]: {e}")

def calc_pressure_level(f, step, cache, level_hpa, var_type):
    p = get_cached(f, step, cache, "pressure")
    if var_type == "wind":
        u = get_cached(f, step, cache, "ua")
        v = get_cached(f, step, cache, "va")
        wspd_kmh = np.sqrt(u**2 + v**2) * 3.6
        return interplevel(wspd_kmh, p, level_hpa)
    elif var_type == "temp":
        tc = get_cached(f, step, cache, "tc")
        return interplevel(tc, p, level_hpa)

def calc_rh2(f, step, cache):
    try:
        rh2 = get_cached(f, step, cache, "rh2")
        return np.clip(np.asarray(rh2, dtype=np.float32), 0.0, 100.0)
    except Exception:
        t2 = get_cached(f, step, cache, "T2") - 273.15
        td2 = get_cached(f, step, cache, "td2")
        e = 6.112 * np.exp((17.67 * td2) / (td2 + 243.5))
        es = 6.112 * np.exp((17.67 * t2) / (t2 + 243.5))
        return np.clip(100.0 * (e / es), 0.0, 100.0)

def calc_pressure_rh(f, step, cache, level_hpa):
    p = get_cached(f, step, cache, "pressure")
    rh = get_cached(f, step, cache, "rh")
    return interplevel(rh, p, level_hpa)

def calc_pwat(f, step, cache):
    """Srážitelná voda v celém sloupci atmosféry (PWAT) v mm."""
    try:
        pw = get_cached(f, step, cache, "pw")
        return np.maximum(np.asarray(pw, dtype=np.float32), 0.0)
    except Exception as e:
        print(f"[PWAT ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_cloud_cover_layer(f, step, cache, layer_type="total"):
    """
    Vypočítá pokrytí oblačností (%) pro celou atmosféru nebo konkrétní vrstvu.
    - low:  >= 800 hPa (~0 až 2 km AGL)
    - mid:  800 až 400 hPa (~2 až 6 km AGL)
    - high: < 400 hPa (~nad 6 km AGL)
    """
    try:
        cldfra = get_cached(f, step, cache, "CLDFRA")
        cld_arr = np.clip(np.asarray(cldfra, dtype=np.float32), 0.0, 1.0)

        if layer_type != "total":
            p = get_cached(f, step, cache, "pressure")
            p_arr = np.asarray(p, dtype=np.float32)

            if layer_type == "low":
                mask = p_arr >= 800.0
            elif layer_type == "mid":
                mask = (p_arr < 800.0) & (p_arr >= 400.0)
            elif layer_type == "high":
                mask = p_arr < 400.0
            else:
                mask = np.ones_like(p_arr, dtype=bool)

            cld_arr = np.where(mask, cld_arr, 0.0)

        # Náhodné překrývání vrstev (random overlap)
        layer_cld = (1.0 - np.prod(1.0 - cld_arr, axis=0)) * 100.0
        return np.clip(layer_cld, 0.0, 100.0)
    except Exception as e:
        print(f"[{layer_type.upper()} CLOUD ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_freezing_level(f, step, cache):
    """Výška hladiny 0 °C над terénem (m AGL)."""
    try:
        tc = get_cached(f, step, cache, "tc")
        z3d = get_cached(f, step, cache, "z")
        ter = get_cached(f, step, cache, "ter")
        zagl = z3d - ter
        fz_lvl = interplevel(zagl, tc, 0.0)
        return np.maximum(np.nan_to_num(fz_lvl, nan=0.0), 0.0)
    except Exception as e:
        print(f"[FREEZING LEVEL ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_gusts_universal(f, step, cache):
    u10 = get_cached(f, step, cache, "U10")
    v10 = get_cached(f, step, cache, "V10")
    wspd10 = np.sqrt(u10**2 + v10**2)
    
    try:
        pblh = np.asarray(get_cached(f, step, cache, "PBLH"), dtype=np.float32)
        try:
            uvmet = get_cached(f, step, cache, "uvmet")
            u3d, v3d = uvmet[0], uvmet[1]
        except Exception:
            u3d = get_cached(f, step, cache, "ua")
            v3d = get_cached(f, step, cache, "va")
            
        wspd3d = np.sqrt(u3d**2 + v3d**2)
        z3d = get_cached(f, step, cache, "z")
        ter = get_cached(f, step, cache, "ter")
        zagl = z3d - ter
        
        target_h = np.clip(pblh, 200.0, 1500.0)
        
        nz, ny, nx = zagl.shape
        idx = np.sum(zagl < target_h, axis=0)
        idx = np.clip(idx, 1, nz - 1)
        y_grid, x_grid = np.ogrid[:ny, :nx]
        
        z0 = zagl[idx - 1, y_grid, x_grid]
        z1 = zagl[idx, y_grid, x_grid]
        dz = np.maximum(z1 - z0, 1e-5)
        frac = np.clip((target_h - z0) / dz, 0.0, 1.0)
        
        wspd_pbl = wspd3d[idx - 1, y_grid, x_grid] + frac * (wspd3d[idx, y_grid, x_grid] - wspd3d[idx - 1, y_grid, x_grid])
        gust_ms = np.maximum(1.28 * wspd10, wspd10 + 0.5 * np.maximum(wspd_pbl - wspd10, 0.0))
    except Exception:
        gust_ms = wspd10 * 1.35
        
    return gust_ms * 3.6

def calc_max_refl10cm(f, step, cache):
    try:
        refl = get_cached(f, step, cache, "REFL_10CM")
        max_dbz = np.max(np.asarray(refl), axis=0)
    except Exception:
        max_dbz = get_cached(f, step, cache, "mdbz")

    smoothed = gaussian_filter(max_dbz.astype(np.float32), sigma=0.3, mode='nearest')
    return np.where(smoothed < 4.0, np.nan, smoothed) 

def calc_sirs(f, step, cache):
    try:
        t_surf_c = get_cached(f, step, cache, "TSK") - 273.15
    except Exception:
        t_surf_c = get_cached(f, step, cache, "T2") - 273.15

    try:
        qc = np.maximum(get_cached(f, step, cache, "QCLOUD"), 0.0)
        qi = np.maximum(get_cached(f, step, cache, "QICE"), 0.0)
        qs = np.maximum(get_cached(f, step, cache, "QSNOW"), 0.0)
        q_total = qc + qi + qs
        t_c = get_cached(f, step, cache, "tc")

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

def calc_total_precip(f, step, cache):
    accum = None
    for name in ('RAINNC', 'RAINC', 'RAINSH'):
        try:
            var = get_cached(f, step, cache, name)
            arr = np.asarray(var, dtype=np.float32)
            accum = arr if accum is None else accum + arr
        except Exception:
            continue
    return accum if accum is not None else np.zeros_like(get_cached(f, step, cache, "XLAT"))

def calc_delta_precip(f, step, cache, hours_back):
    current_dt = datetime_times[step]
    target_dt = current_dt - timedelta(hours=hours_back)
    
    past_step = 0
    for i in range(step, -1, -1):
        if datetime_times[i] <= target_dt:
            past_step = i
            break
            
    current_p = calc_total_precip(f, step, cache)
    past_cache = {}
    past_p = calc_total_precip(f, past_step, past_cache)
    
    delta = current_p - past_p
    return np.maximum(delta, 0.0)

def calc_snow_depth(f, step, cache):
    try:
        snowh = get_cached(f, step, cache, "SNOWH")
        return np.maximum(np.asarray(snowh, dtype=np.float32) * 100.0, 0.0)
    except Exception:
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_total_snow_acc(f, step, cache):
    try:
        snownc = get_cached(f, step, cache, "SNOWNC")
        return np.maximum(np.asarray(snownc, dtype=np.float32), 0.0)
    except Exception:
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_delta_snow_acc(f, step, cache, hours_back):
    current_dt = datetime_times[step]
    target_dt = current_dt - timedelta(hours=hours_back)
    
    past_step = 0
    for i in range(step, -1, -1):
        if datetime_times[i] <= target_dt:
            past_step = i
            break
            
    current_s = calc_total_snow_acc(f, step, cache)
    past_cache = {}
    past_s = calc_total_snow_acc(f, past_step, past_cache)
    
    delta = current_s - past_s
    return np.maximum(delta, 0.0)

def calc_bulk_shear(f, step, cache, target_m):
    u10 = get_cached(f, step, cache, "U10")
    v10 = get_cached(f, step, cache, "V10")
    try:
        uvmet = get_cached(f, step, cache, "uvmet")
        u3d, v3d = uvmet[0], uvmet[1]
    except Exception:
        u3d = get_cached(f, step, cache, "ua")
        v3d = get_cached(f, step, cache, "va")

    z3d = get_cached(f, step, cache, "z")
    ter = get_cached(f, step, cache, "ter")
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

def calc_ref_level(f, step, cache, target_m=2000.0):
    try:
        dbz = get_cached(f, step, cache, "REFL_10CM")
    except Exception:
        dbz = get_cached(f, step, cache, "dbz")

    z3d = get_cached(f, step, cache, "z")
    ter = get_cached(f, step, cache, "ter")
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

def calc_echotops(f, step, cache, threshold_dbz=18.0):
    try:
        refl = get_cached(f, step, cache, "REFL_10CM")
    except Exception:
        refl = get_cached(f, step, cache, "dbz")

    z3d = get_cached(f, step, cache, "z")
    ter = get_cached(f, step, cache, "ter")
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

def calc_sbcape(f, step, cache):
    try:
        cape_2d = get_cached(f, step, cache, "cape_2d")
        cape_arr = np.asarray(cape_2d, dtype=np.float32)
        sbcape = cape_arr[0] if cape_arr.ndim == 3 else cape_arr
        return np.maximum(np.nan_to_num(sbcape, nan=0.0), 0.0)
    except Exception as e:
        print(f"[SBCAPE ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_mlcape(f, step, cache):
    try:
        cape_3d = get_cached(f, step, cache, "cape_3d")
        cape_arr = np.asarray(cape_3d, dtype=np.float32)

        if cape_arr.ndim == 4:
            cape_field = cape_arr[0]
            num_levels = min(4, cape_field.shape[0])
            mlcape = np.mean(cape_field[:num_levels, :, :], axis=0)
        elif cape_arr.ndim == 3:
            mlcape = cape_arr[0]
        else:
            mlcape = cape_arr

        return np.maximum(np.nan_to_num(mlcape, nan=0.0), 0.0)
    except Exception as e:
        print(f"[MLCAPE ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_lapse_rate_700_500(f, step, cache):
    try:
        p = get_cached(f, step, cache, "pressure")
        tc = get_cached(f, step, cache, "tc")
        z = get_cached(f, step, cache, "z")

        t700 = interplevel(tc, p, 700.0)
        t500 = interplevel(tc, p, 500.0)
        z700 = interplevel(z, p, 700.0)
        z500 = interplevel(z, p, 500.0)

        dz_km = (z500 - z700) / 1000.0
        dt = t700 - t500

        lapse_rate = dt / dz_km
        return np.asarray(lapse_rate, dtype=np.float32)
    except Exception as e:
        print(f"[LAPSE RATE ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_lifted_index(f, step, cache):
    try:
        p = get_cached(f, step, cache, "pressure")
        tc = get_cached(f, step, cache, "tc")
        t2 = get_cached(f, step, cache, "T2") - 273.15
        td2 = get_cached(f, step, cache, "td2")
        psfc = get_cached(f, step, cache, "PSFC") / 100.0

        t500_env = interplevel(tc, p, 500.0)

        t_k = t2 + 273.15
        e = 6.112 * np.exp((17.67 * td2) / (td2 + 243.5))
        q = (0.622 * e) / (psfc - 0.378 * e)
        theta_e = (t_k + (2500000.0 / 1004.0) * q) * ((1000.0 / psfc) ** 0.286)

        t_parcel_500_k = (theta_e / ((1000.0 / 500.0) ** 0.286)) - 15.0
        t_parcel_500_c = t_parcel_500_k - 273.15

        li = t500_env - t_parcel_500_c
        return np.asarray(li, dtype=np.float32)
    except Exception as e:
        print(f"[LIFTED INDEX ERROR]: {e}")
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

def calc_lightning(f, step, cache):
    try:
        qc = np.maximum(np.asarray(get_cached(f, step, cache, "QCLOUD"), dtype=np.float32), 0.0)
        qr = np.maximum(np.asarray(get_cached(f, step, cache, "QRAIN"), dtype=np.float32), 0.0)
        qi = np.maximum(np.asarray(get_cached(f, step, cache, "QICE"), dtype=np.float32), 0.0)
        qs = np.maximum(np.asarray(get_cached(f, step, cache, "QSNOW"), dtype=np.float32), 0.0)
        qg = np.maximum(np.asarray(get_cached(f, step, cache, "QGRAUP"), dtype=np.float32), 0.0)
        temp_c = np.asarray(get_cached(f, step, cache, "tc"), dtype=np.float32)
        height = np.asarray(get_cached(f, step, cache, "z"), dtype=np.float32)
        w = np.asarray(get_cached(f, step, cache, "wa"), dtype=np.float32)
        mdbz = np.nan_to_num(np.asarray(get_cached(f, step, cache, "mdbz"), dtype=np.float32), nan=0.0)
        dbz = np.nan_to_num(np.asarray(get_cached(f, step, cache, "dbz"), dtype=np.float32), nan=0.0)

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
        return np.zeros_like(get_cached(f, step, cache, "XLAT"), dtype=np.float32)

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
snow_cmap, snow_levels = load_custom_cmap(CT_DIR / "snow.ct")
lapse_cmap, lapse_levels = load_custom_cmap(CT_DIR / "lapse_rate.ct")
li_cmap, li_levels = load_custom_cmap(CT_DIR / "lifted_index.ct")
rh_cmap, rh_levels = load_custom_cmap(CT_DIR / "rh.ct")
cld_cmap, cld_levels = load_custom_cmap(CT_DIR / "cloudcover.ct")
iso_cmap, iso_levels = load_custom_cmap(CT_DIR / "izoterma.ct")

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

# Pre-clipping geometrií Cartopy pro bleskový rendering
lon_min, lon_max = float(lons.min()), float(lons.max())
lat_min, lat_max = float(lats.min()), float(lats.max())
BOUNDS = [lon_min, lon_max, lat_min, lat_max]

print("Příprava kartografických geometrií v RAM...")
STATES_GEOMS = list(cfeature.STATES.with_scale('10m').intersecting_geometries(BOUNDS))
BORDERS_GEOMS = list(cfeature.BORDERS.with_scale('10m').intersecting_geometries(BOUNDS))
COAST_GEOMS = list(cfeature.COASTLINE.with_scale('10m').intersecting_geometries(BOUNDS))

# ==========================================
# 3. KONFIGURACE VŠECH DATOVÝCH POLÍ
# ==========================================
variables_config = {
    "temp": {
        "title": "Teplota ve 2m",
        "get_data": lambda f, step, c: get_cached(f, step, c, "T2") - 273.15,
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "extend": "both",
        "draw_grid": True
    },
    "dewpoint": {
        "title": "Rosný bod ve 2m",
        "get_data": lambda f, step, c: get_cached(f, step, c, "td2"),
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "extend": "both",
        "draw_grid": True
    },
    "rh2m": {
        "title": "Relativní vlhkost ve 2m",
        "get_data": lambda f, step, c: calc_rh2(f, step, c),
        "cmap": rh_cmap,
        "levels": rh_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
    },
    "mslp": {
        "title": "MSLP (Tlak)",
        "get_data": lambda f, step, c: get_cached(f, step, c, "slp"),
        "cmap": mslp_cmap,
        "levels": mslp_levels,
        "label": "hPa",
        "ticks": [940, 955, 970, 986, 1001, 1016, 1031, 1046],
        "extend": "both"
    },
    "pwat": {
        "title": "Srážitelná voda (PWAT)",
        "get_data": lambda f, step, c: calc_pwat(f, step, c),
        "cmap": rh_cmap,
        "levels": rh_levels,
        "label": "mm",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "max"
    },
    "cloud_cover": {
        "title": "Celková oblačnost",
        "get_data": lambda f, step, c: calc_cloud_cover_layer(f, step, c, "total"),
        "cmap": cld_cmap,
        "levels": cld_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
    },
    "cloud_cover_low": {
        "title": "Nízká oblačnost (>800 hPa)",
        "get_data": lambda f, step, c: calc_cloud_cover_layer(f, step, c, "low"),
        "cmap": cld_cmap,
        "levels": cld_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
    },
    "cloud_cover_mid": {
        "title": "Střední oblačnost (800-400 hPa)",
        "get_data": lambda f, step, c: calc_cloud_cover_layer(f, step, c, "mid"),
        "cmap": cld_cmap,
        "levels": cld_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
    },
    "cloud_cover_high": {
        "title": "Vysoká oblačnost (<400 hPa)",
        "get_data": lambda f, step, c: calc_cloud_cover_layer(f, step, c, "high"),
        "cmap": cld_cmap,
        "levels": cld_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
    },
    "freezing_level": {
        "title": "Nulová izoterma (0 °C)",
        "get_data": lambda f, step, c: calc_freezing_level(f, step, c),
        "cmap": iso_cmap,
        "levels": iso_levels,
        "label": "m AGL",
        "ticks": [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000],
        "extend": "max"
    },
    "wind10m": {
        "title": "Vítr v 10m",
        "get_data": lambda f, step, c: np.sqrt(get_cached(f, step, c, "U10")**2 + get_cached(f, step, c, "V10")**2) * 3.6,
        "get_uv": lambda f, step, c: (get_cached(f, step, c, "U10"), get_cached(f, step, c, "V10")),
        "cmap": wind_cmap,
        "levels": wind_levels,
        "label": "km/h",
        "ticks": [0, 20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 280],
        "extend": "max"
    },
    "GUST": {
        "title": "Nárazy větru v 10m",
        "get_data": lambda f, step, c: calc_gusts_universal(f, step, c),
        "cmap": wind_cmap,
        "levels": wind_levels,
        "label": "km/h",
        "ticks": [0, 20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 280],
        "extend": "max"
    },
    "sat_ir": {
        "title": "Simulovaný IR Satelit",
        "get_data": lambda f, step, c: calc_sirs(f, step, c),
        "cmap": sirs_cmap,
        "levels": sirs_levels,
        "label": "°C",
        "ticks": [-90, -80, -70, -65, -60, -55, -50, -45, -40, -30, -10, 0, 10, 30, 50],
        "extend": "both"
    },
    "cape": {
        "title": "MUCAPE",
        "get_data": lambda f, step, c: get_cached(f, step, c, "cape_2d")[0],
        "cmap": cape_cmap,
        "levels": cape_levels,
        "label": "J/kg",
        "ticks": [0, 300, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000],
        "extend": "max"
    },
    "sbcape": {
        "title": "SBCAPE (Surface-Based)",
        "get_data": lambda f, step, c: calc_sbcape(f, step, c),
        "cmap": cape_cmap,
        "levels": cape_levels,
        "label": "J/kg",
        "ticks": [0, 300, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000],
        "extend": "max"
    },
    "mlcape": {
        "title": "MLCAPE (Mixed-Layer)",
        "get_data": lambda f, step, c: calc_mlcape(f, step, c),
        "cmap": cape_cmap,
        "levels": cape_levels,
        "label": "J/kg",
        "ticks": [0, 300, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000],
        "extend": "max"
    },
    "cin": {
        "title": "MUCIN",
        "get_data": lambda f, step, c: np.abs(get_cached(f, step, c, "cape_2d")[1]),
        "cmap": "Blues",
        "levels": [0, 10, 25, 50, 100, 150, 200, 300, 400, 500],
        "label": "J/kg",
        "ticks": [0, 25, 50, 100, 200, 300, 500],
        "extend": "max"
    },
    "shear_0_1km": {
        "title": "Bulk Shear 0-1 km",
        "get_data": lambda f, step, c: calc_bulk_shear(f, step, c, 1000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },
    "shear_0_3km": {
        "title": "Bulk Shear 0-3 km",
        "get_data": lambda f, step, c: calc_bulk_shear(f, step, c, 3000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },
    "shear_0_6km": {
        "title": "Bulk Shear 0-6 km",
        "get_data": lambda f, step, c: calc_bulk_shear(f, step, c, 6000.0),
        "cmap": shear_cmap,
        "levels": shear_levels,
        "label": "m/s",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70],
        "extend": "max"
    },
    "lapse_rate_700_500": {
        "title": "Lapse Rate (700-500 hPa)",
        "get_data": lambda f, step, c: calc_lapse_rate_700_500(f, step, c),
        "cmap": lapse_cmap,
        "levels": lapse_levels,
        "label": "°C/km",
        "ticks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5],
        "extend": "both"
    },
    "lifted_index": {
        "title": "Lifted Index (LI)",
        "get_data": lambda f, step, c: calc_lifted_index(f, step, c),
        "cmap": li_cmap,
        "levels": li_levels,
        "label": "°C",
        "ticks": [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "extend": "both"
    },
    "radar_max": {
        "title": "Simulovaný radar (Max dBZ)",
        "get_data": lambda f, step, c: calc_max_refl10cm(f, step, c),
        "cmap": radar_cmap,
        "levels": radar_levels,
        "label": "dBZ",
        "ticks": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
        "extend": "max"
    },
    "radar_2km": {
        "title": "Odrazivost ve 2 km AGL",
        "get_data": lambda f, step, c: calc_ref_level(f, step, c, 2000.0),
        "cmap": radar_cmap,
        "levels": radar_levels,
        "label": "dBZ",
        "ticks": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
        "extend": "max"
    },
    "echotops_18": {
        "title": "Echo Tops 18 dBZ",
        "get_data": lambda f, step, c: calc_echotops(f, step, c, 18.0),
        "cmap": echo_cmap,
        "levels": echo_levels,
        "label": "km AGL",
        "ticks": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        "extend": "max"
    },
    "echotops_45": {
        "title": "Echo Tops 45 dBZ",
        "get_data": lambda f, step, c: calc_echotops(f, step, c, 45.0),
        "cmap": echo_cmap,
        "levels": echo_levels,
        "label": "km AGL",
        "ticks": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        "extend": "max"
    },
    "lightning": {
        "title": "Blesková aktivita (LPI)",
        "get_data": lambda f, step, c: calc_lightning(f, step, c),
        "cmap": light_cmap,
        "levels": light_levels,
        "label": "flashes/h",
        "ticks": [0.1, 1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 360, 720],
        "extend": "max"
    },
    "precip_total": {
        "title": "Celkové srážky",
        "get_data": lambda f, step, c: calc_total_precip(f, step, c),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_1h": {
        "title": "Srážky za 1h",
        "get_data": lambda f, step, c: calc_delta_precip(f, step, c, 1.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_6h": {
        "title": "Srážky za 6h",
        "get_data": lambda f, step, c: calc_delta_precip(f, step, c, 6.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "precip_24h": {
        "title": "Srážky za 24h",
        "get_data": lambda f, step, c: calc_delta_precip(f, step, c, 24.0),
        "cmap": precip_cmap,
        "levels": precip_levels,
        "label": "mm",
        "ticks": [0.1, 1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 500],
        "extend": "max"
    },
    "snow_depth": {
        "title": "Celková sněhová pokrývka",
        "get_data": lambda f, step, c: calc_snow_depth(f, step, c),
        "cmap": snow_cmap,
        "levels": snow_levels,
        "label": "cm",
        "ticks": [0.1, 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400],
        "extend": "max"
    },
    "snow_acc_1h": {
        "title": "Nový sníh za 1h",
        "get_data": lambda f, step, c: calc_delta_snow_acc(f, step, c, 1.0),
        "cmap": snow_cmap,
        "levels": snow_levels,
        "label": "cm",
        "ticks": [0.1, 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400],
        "extend": "max"
    },
    "snow_acc_6h": {
        "title": "Nový sníh za 6h",
        "get_data": lambda f, step, c: calc_delta_snow_acc(f, step, c, 6.0),
        "cmap": snow_cmap,
        "levels": snow_levels,
        "label": "cm",
        "ticks": [0.1, 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400],
        "extend": "max"
    },
    "snow_acc_24h": {
        "title": "Nový sníh za 24h",
        "get_data": lambda f, step, c: calc_delta_snow_acc(f, step, c, 24.0),
        "cmap": snow_cmap,
        "levels": snow_levels,
        "label": "cm",
        "ticks": [0.1, 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400],
        "extend": "max"
    },
    "snow_acc_total": {
        "title": "Nový sníh celkem",
        "get_data": lambda f, step, c: calc_total_snow_acc(f, step, c),
        "cmap": snow_cmap,
        "levels": snow_levels,
        "label": "cm",
        "ticks": [0.1, 1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400],
        "extend": "max"
    },
}

UPPER_LEVELS = [925, 850, 700, 500, 400, 300, 200]

for p_lev in UPPER_LEVELS:
    variables_config[f"wind_{p_lev}hpa"] = {
        "title": f"Vítr v {p_lev} hPa",
        "get_data": lambda f, step, c, p=p_lev: calc_pressure_level(f, step, c, p, "wind"),
        "get_uv": lambda f, step, c, p=p_lev: calc_pressure_uv(f, step, c, p),
        "cmap": wind_cmap,
        "levels": wind_levels,
        "label": "km/h",
        "ticks": [0, 20, 40, 60, 80, 100, 130, 160, 200, 240, 280],
        "extend": "max"
    }
    variables_config[f"temp_{p_lev}hpa"] = {
        "title": f"Teplota v {p_lev} hPa",
        "get_data": lambda f, step, c, p=p_lev: calc_pressure_level(f, step, c, p, "temp"),
        "cmap": temp_cmap,
        "levels": temp_levels,
        "label": "°C",
        "ticks": [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30],
        "extend": "both",
        "draw_grid": True
    }
    variables_config[f"rh_{p_lev}hpa"] = {
        "title": f"Relativní vlhkost v {p_lev} hPa",
        "get_data": lambda f, step, c, p=p_lev: calc_pressure_rh(f, step, c, p),
        "cmap": rh_cmap,
        "levels": rh_levels,
        "label": "%",
        "ticks": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "extend": "neither"
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
    
    # Keš pro zamezení opakovaných readů z NetCDF
    step_cache = {}
    local_max_cape = 0.0

    # Vytvoření plátna pouze JEDNOU pro celý časový krok
    fig = plt.figure(figsize=(10.128, 5.7), facecolor='#0f172a')
    ax = fig.add_axes([0.02, 0.03, 0.82, 0.88], projection=cart_proj)
    banner_ax = fig.add_axes([0.02, 0.92, 0.955, 0.06], facecolor='#1e293b')

    for var_key, cfg in variables_config.items():
        data_field = cfg["get_data"](f_thread, step, step_cache)

        if var_key == "cape":
            local_max_cape = float(np.nanmax(data_field))

        # Čištění os
        ax.cla()
        banner_ax.cla()
        banner_ax.axis('off')

        # Vždy vytvoříme čistou osu colorbaru s plnou výškou (88 %)
        if 'cbar_ax' in locals() and cbar_ax in fig.axes:
            cbar_ax.remove()
        cbar_ax = fig.add_axes([0.85, 0.03, 0.025, 0.88])

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=cart_proj)
        ax.set_facecolor('#1e293b')

        # Vykreslení zrychlených předpřipravených geometrií
        ax.add_geometries(STATES_GEOMS, crs=cart_proj, linewidth=0.5, edgecolor='#334155', facecolor='none', zorder=14)
        ax.add_geometries(BORDERS_GEOMS, crs=cart_proj, linewidth=1.0, edgecolor='#020617', facecolor='none', zorder=15)
        ax.add_geometries(COAST_GEOMS, crs=cart_proj, linewidth=1.0, edgecolor='#020617', facecolor='none', zorder=15)

        levels = cfg["levels"]
        cmap = cfg["cmap"]
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)
        extend_mode = cfg.get("extend", "max")

        cf = ax.contourf(
            lons, lats, data_field,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=cart_proj,
            extend=extend_mode
        )
        ax.autoscale(False)

        if var_key == "mslp":
            iso_levels = np.arange(940, 1060, 2)
            cs = ax.contour(lons, lats, data_field, levels=iso_levels, colors='#e2e8f0', linewidths=0.6, transform=cart_proj)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%d hPa', colors='#ffffff')

        if "get_uv" in cfg:
            try:
                u_field, v_field = cfg["get_uv"](f_thread, step, step_cache)
                draw_streamlines(ax, lons, lats, u_field, v_field, color='white', linewidth=0.8, density=1.2)
            except Exception as e:
                print(f"Chyba vykreslení proudnic u {var_key}: {e}")

        if cfg.get("draw_grid", False):
            draw_value_grid(ax, lons, lats, data_field, num_x=12, num_y=7, fmt="{:.0f}")

        cbar = fig.colorbar(cf, cax=cbar_ax, ticks=cfg["ticks"], spacing='uniform')
        cbar.ax.tick_params(labelsize=9, colors='#f8fafc')
        cbar.set_label(cfg["label"], color='#f8fafc', fontsize=10, fontweight='bold')

        banner_ax.text(0.01, 0.5, "WRF ARW", color='#38bdf8', fontsize=10, fontweight='bold', va='center')
        banner_ax.text(0.12, 0.5, f"|  {cfg['title']}", color='#ffffff', fontsize=10, fontweight='bold', va='center')

        time_text = f"Init: {init_str} UTC  |  Valid: {time_str} UTC (+{f_hour:02d}h)"
        banner_ax.text(0.99, 0.5, time_text, color='#f1f5f9', fontsize=8.8, va='center', ha='right', family='monospace')

        out_file = output_dir / var_key / f"{step:03d}.webp"
        safe_savefig(fig, out_file)

    plt.close(fig)
    print(f"Done [{step + 1}/{total_steps}]: {time_str}")
    return local_max_cape

# ==========================================
# 5. SPUŠTĚNÍ A METADATA
# ==========================================
if __name__ == "__main__":
    max_workers = min(12, os.cpu_count())
    print(f"Start zrychleného paralelního zpracování '{run_id}' na {max_workers} worker procesech...")

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