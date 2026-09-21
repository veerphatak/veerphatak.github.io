"""
Physics analysis script: trajectory_physics_open_channel_full_report.py

Purpose
-------
Computes the full open-channel trajectory physics report from processed trajectories, particle geometry and flow properties.

Inputs
------
Processed trajectories, particle-property dictionaries and flow-property settings.

Outputs
-------
Full metric tables, summary plots and derived physics quantities.

Methodological notes
--------------------
Particle motion is converted from image-plane kinematics into hydraulic descriptors such as Reynolds number, slip ratio and drag-related metrics using the stated geometric approximations.

Appendix use
------------
This file is included as supporting code for the dissertation/report. Comments are
restricted to methodological choices, assumptions and processing stages so that the
appendix remains readable without duplicating every line of executable logic.
"""

import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUTPUT_ROOT = Path('C:\\IP Work\\Open_Channel_Physics_Reports_3_and_4.5_Inchps')
WORKSPACE = Path('C:\\IP Work\\New project')
BATCHES = [{'label': '3_Inchps', 'flow_speed_in_s': 3.0, 'source_dirs': [Path('C:\\IP Work\\3_Inchps_Rod_Disc_Binarised\\rods'), Path('C:\\IP Work\\3_Inchps_Rod_Disc_Binarised\\discs')]}, {'label': '4.5_Inchps', 'flow_speed_in_s': 4.5, 'source_dirs': [Path('C:\\IP Work\\4.5_Inchps_rod_binarised_widthlocked_5mm_fullbatch_v2'), Path('C:\\IP Work\\4.5_Inchps_Discs_Binarised'), Path('C:\\IP Work\\4.5_Inchps_Spheres_Binarised')]}]
PX_PER_MM = 4.0
MM_PER_PX = 1.0 / PX_PER_MM
FPS = 240.0
RHO_F = 1000.0
MU = 0.001
G = 9.81
NU = MU / RHO_F
RHO_PP = 905.0
RHO_ACRYLIC = 1180.0
DISC_DIAMETER_MM = 20.0
DISC_THICKNESS_MM = 1.5
ROD_DIAMETER_MM = 5.0
ROD_LENGTH_MM = 25.0
SMOOTHING_WINDOW = 9
MIN_POINTS = 12
SAVE_PER_TRIAL_KINEMATICS = True

def find_col(df, candidates):
    """Handle the find col step used by this script."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def safe_scalar(value):
    """Handle the safe scalar step used by this script."""
    if value is None:
        return np.nan
    if isinstance(value, (np.floating, float, np.integer, int)):
        if np.isfinite(value):
            return float(value)
        return np.nan
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan

def infer_particle_info(folder_name: str):
    """Handle the infer particle info step used by this script."""
    name = folder_name.lower()
    if '10mm' in name and 'sphere' in name:
        return {'particle_type': '10 mm sphere', 'shape': 'sphere', 'material': 'polypropylene', 'density': RHO_PP, 'diameter_m': 0.01}
    if '14mm' in name and 'sphere' in name:
        return {'particle_type': '14 mm sphere', 'shape': 'sphere', 'material': 'polypropylene', 'density': RHO_PP, 'diameter_m': 0.014}
    if 'disc' in name:
        return {'particle_type': 'disc', 'shape': 'disc', 'material': 'acrylic', 'density': RHO_ACRYLIC, 'diameter_m': DISC_DIAMETER_MM / 1000.0, 'thickness_m': DISC_THICKNESS_MM / 1000.0}
    if 'rod' in name:
        return {'particle_type': 'rod', 'shape': 'rod', 'material': 'acrylic', 'density': RHO_ACRYLIC, 'diameter_m': ROD_DIAMETER_MM / 1000.0, 'length_m': ROD_LENGTH_MM / 1000.0}
    return {'particle_type': 'unknown', 'shape': 'sphere', 'material': 'unknown', 'density': RHO_PP, 'diameter_m': 0.01}

def geometry_properties(info):
    """Handle the geometry properties step used by this script."""
    shape = info['shape']
    if shape == 'sphere':
        d = info['diameter_m']
        r = d / 2.0
        volume = 4.0 / 3.0 * math.pi * r ** 3
        area = math.pi * r ** 2
        lref = d
        aspect_ratio = 1.0
    elif shape == 'disc':
        d = info['diameter_m']
        t = info['thickness_m']
        r = d / 2.0
        volume = math.pi * r ** 2 * t
        area = math.pi * r ** 2
        lref = d
        aspect_ratio = d / t if t > 0 else np.nan
    elif shape == 'rod':
        d = info['diameter_m']
        length = info['length_m']
        r = d / 2.0
        volume = math.pi * r ** 2 * length
        area = d * length
        lref = d
        aspect_ratio = length / d if d > 0 else np.nan
    else:
        raise ValueError(f'Unknown shape: {shape}')
    d_eq = (6.0 * volume / math.pi) ** (1.0 / 3.0)
    return {'volume_m3': volume, 'projected_area_m2': area, 'lref_m': lref, 'equivalent_diameter_m': d_eq, 'shape_aspect_ratio': aspect_ratio}

def smooth_series(arr, window=9):
    """Handle the smooth series step used by this script."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 5:
        return arr.copy()
    window = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if window < 3:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')

def load_tracks(csv_path: Path, fps_fallback: float=FPS):
    """Load the tracks used by this script."""
    df = pd.read_csv(csv_path)
    frame_col = find_col(df, ['out_idx', 'frame_idx', 'frame', 'frame_index', 'clip_frame'])
    time_col = find_col(df, ['t_rel_s', 'time_s', 't_s'])
    x_mm_col = find_col(df, ['x_kalman_mm', 'x_mm'])
    y_mm_col = find_col(df, ['y_kalman_mm', 'y_mm'])
    if x_mm_col and y_mm_col:
        out = pd.DataFrame({'frame': df[frame_col] if frame_col else np.arange(len(df)), 't_s': df[time_col] if time_col else np.nan, 'x_mm': pd.to_numeric(df[x_mm_col], errors='coerce'), 'y_mm': pd.to_numeric(df[y_mm_col], errors='coerce')})
    else:
        x_px_col = find_col(df, ['cx_kalman_raw_px', 'cx_raw_px', 'track_x', 'x'])
        y_px_col = find_col(df, ['cy_kalman_raw_px', 'cy_raw_px', 'track_y', 'y'])
        if x_px_col is None or y_px_col is None:
            raise ValueError(f'No usable position columns found in {csv_path.name}')
        out = pd.DataFrame({'frame': df[frame_col] if frame_col else np.arange(len(df)), 't_s': df[time_col] if time_col else np.nan, 'x_mm': pd.to_numeric(df[x_px_col], errors='coerce') * MM_PER_PX, 'y_mm': pd.to_numeric(df[y_px_col], errors='coerce') * MM_PER_PX})
    out['frame'] = pd.to_numeric(out['frame'], errors='coerce')
    fps_fallback = float(fps_fallback) if np.isfinite(fps_fallback) and float(fps_fallback) > 0 else float(FPS)
    if out['t_s'].isna().all():
        out['t_s'] = out['frame'] / fps_fallback
    else:
        out['t_s'] = pd.to_numeric(out['t_s'], errors='coerce')
        mask = out['t_s'].isna()
        out.loc[mask, 't_s'] = out.loc[mask, 'frame'] / fps_fallback
    out = out.dropna(subset=['frame', 't_s', 'x_mm', 'y_mm']).copy()
    out['frame'] = out['frame'].astype(int)
    out = out.sort_values('frame').drop_duplicates('frame')
    if len(out) < MIN_POINTS:
        raise ValueError(f'Too few valid points in {csv_path.name}: {len(out)}')
    return out.reset_index(drop=True)

def pca_direction(x, y):
    """Handle the pca direction step used by this script."""
    p = np.column_stack([x, y])
    p = p - p.mean(axis=0)
    _, _, vh = np.linalg.svd(p, full_matrices=False)
    v = vh[0]
    return v / np.linalg.norm(v)

def compute_kinematics(tracks):
    """Compute the kinematics used by this script."""
    t = tracks['t_s'].to_numpy(dtype=float)
    x_m = tracks['x_mm'].to_numpy(dtype=float) / 1000.0
    y_m = tracks['y_mm'].to_numpy(dtype=float) / 1000.0
    x_s = smooth_series(x_m, SMOOTHING_WINDOW)
    y_s = smooth_series(y_m, SMOOTHING_WINDOW)
    vx = np.gradient(x_s, t)
    vy = np.gradient(y_s, t)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    denom = np.power(vx ** 2 + vy ** 2, 1.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(denom > 1e-12, np.abs(vx * ay - vy * ax) / denom, np.nan)
    a_normal = speed ** 2 * curvature
    return pd.DataFrame({'frame': tracks['frame'].to_numpy(), 't_s': t, 'x_m': x_s, 'y_m': y_s, 'x_mm': x_s * 1000.0, 'y_mm': y_s * 1000.0, 'vx_m_s': vx, 'vy_m_s': vy, 'vx_mm_s': vx * 1000.0, 'vy_mm_s': vy * 1000.0, 'speed_m_s': speed, 'speed_mm_s': speed * 1000.0, 'ax_m_s2': ax, 'ay_m_s2': ay, 'curvature_1_m': curvature, 'a_normal_m_s2': a_normal})

def trajectory_metrics(kin):
    """Handle the trajectory metrics step used by this script."""
    x = kin['x_m'].to_numpy()
    y = kin['y_m'].to_numpy()
    dx = np.diff(x)
    dy = np.diff(y)
    path_length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    net_dx = x[-1] - x[0]
    net_dy = y[-1] - y[0]
    net_displacement = np.sqrt(net_dx ** 2 + net_dy ** 2)
    tortuosity = path_length / net_displacement if net_displacement > 1e-12 else np.nan
    return (path_length, net_displacement, tortuosity, net_dx, net_dy)

def define_regions(kin):
    """Handle the define regions step used by this script."""
    n = len(kin)
    if n < 10:
        return (kin.copy(), kin.copy())
    late_start = max(0, int(0.6 * n))
    mid_start = int(0.2 * n)
    mid_end = max(mid_start + 1, int(0.8 * n))
    late = kin.iloc[late_start:].copy()
    middle = kin.iloc[mid_start:mid_end].copy()
    return (late, middle)

def infer_terminal_velocity(late_segment):
    """Handle the infer terminal velocity step used by this script."""
    x_end = late_segment['x_m'].to_numpy()
    y_end = late_segment['y_m'].to_numpy()
    t_end = late_segment['t_s'].to_numpy()
    if len(late_segment) >= 5:
        vdir = pca_direction(x_end, y_end)
        disp = np.array([x_end[-1] - x_end[0], y_end[-1] - y_end[0]])
        if np.dot(vdir, disp) < 0:
            vdir = -vdir
        s_end = x_end * vdir[0] + y_end * vdir[1]
        return abs(np.polyfit(t_end, s_end, 1)[0])
    return abs(np.nanmean(late_segment['speed_m_s']))

def effective_drag_coefficient(rho_p, volume_m3, projected_area_m2, terminal_velocity_m_s):
    """Handle the effective drag coefficient step used by this script."""
    if abs(rho_p - RHO_F) <= 1e-12 or terminal_velocity_m_s <= 1e-12:
        return np.nan
    return 2.0 * abs(rho_p - RHO_F) * volume_m3 * G / (RHO_F * projected_area_m2 * terminal_velocity_m_s ** 2)

def analyse_trial(trial_dir: Path, batch_label: str, flow_speed_in_s: float, analysis_fps: float=FPS):
    """Handle the analyse trial step used by this script."""
    info = infer_particle_info(trial_dir.name)
    geom = geometry_properties(info)
    rho_p = info['density']
    flow_speed_m_s = flow_speed_in_s * 0.0254
    tracks = load_tracks(trial_dir / 'tracks_processed.csv', fps_fallback=analysis_fps)
    kin = compute_kinematics(tracks)
    late, middle = define_regions(kin)
    terminal_velocity_m_s = infer_terminal_velocity(late)
    mean_speed_m_s = safe_scalar(np.nanmean(kin['speed_m_s']))
    max_speed_m_s = safe_scalar(np.nanmax(kin['speed_m_s']))
    mean_curvature = safe_scalar(np.nanmean(kin['curvature_1_m']))
    mean_normal_acc = safe_scalar(np.nanmean(kin['a_normal_m_s2']))
    mean_streamwise_velocity_m_s = safe_scalar(np.nanmean(middle['vx_m_s']))
    mean_vertical_velocity_m_s = safe_scalar(np.nanmean(middle['vy_m_s']))
    mean_abs_streamwise_velocity_m_s = safe_scalar(np.nanmean(np.abs(middle['vx_m_s'])))
    mean_abs_vertical_velocity_m_s = safe_scalar(np.nanmean(np.abs(middle['vy_m_s'])))
    path_length_m, net_disp_m, tortuosity, net_dx_m, net_dy_m = trajectory_metrics(kin)
    duration_s = safe_scalar(kin['t_s'].iloc[-1] - kin['t_s'].iloc[0])
    net_streamwise_velocity_m_s = net_dx_m / duration_s if duration_s and duration_s > 0 else np.nan
    net_vertical_velocity_m_s = net_dy_m / duration_s if duration_s and duration_s > 0 else np.nan
    volume_m3 = geom['volume_m3']
    projected_area_m2 = geom['projected_area_m2']
    lref_m = geom['lref_m']
    d_eq_m = geom['equivalent_diameter_m']
    shape_aspect_ratio = geom['shape_aspect_ratio']
    particle_mass_kg = rho_p * volume_m3
    displaced_fluid_mass_kg = RHO_F * volume_m3
    buoyant_mass_kg = (rho_p - RHO_F) * volume_m3
    weight_n = particle_mass_kg * G
    buoyancy_n = displaced_fluid_mass_kg * G
    apparent_weight_n = buoyant_mass_kg * G
    density_ratio = rho_p / RHO_F
    submerged_specific_gravity = (rho_p - RHO_F) / RHO_F
    re_terminal = RHO_F * terminal_velocity_m_s * d_eq_m / MU if terminal_velocity_m_s > 1e-12 else np.nan
    re_mean = RHO_F * mean_speed_m_s * d_eq_m / MU if np.isfinite(mean_speed_m_s) and mean_speed_m_s > 1e-12 else np.nan
    cd_eff = effective_drag_coefficient(rho_p, volume_m3, projected_area_m2, terminal_velocity_m_s)
    if np.isfinite(mean_normal_acc) and np.isfinite(mean_speed_m_s) and (mean_speed_m_s > 1e-12):
        lift_force_est_n = particle_mass_kg * mean_normal_acc
        cl_eff = 2.0 * lift_force_est_n / (RHO_F * projected_area_m2 * mean_speed_m_s ** 2)
    else:
        lift_force_est_n = np.nan
        cl_eff = np.nan
    archimedes = G * abs(rho_p - RHO_F) / RHO_F * d_eq_m ** 3 / NU ** 2
    galileo = math.sqrt(archimedes) if np.isfinite(archimedes) and archimedes >= 0 else np.nan
    tau_p_stokes_s = rho_p * d_eq_m ** 2 / (18.0 * MU)
    stokes_number_flow = tau_p_stokes_s * flow_speed_m_s / d_eq_m if d_eq_m > 1e-12 else np.nan
    froude_mean = mean_speed_m_s / math.sqrt(G * d_eq_m) if d_eq_m > 1e-12 and np.isfinite(mean_speed_m_s) else np.nan
    froude_terminal = terminal_velocity_m_s / math.sqrt(G * d_eq_m) if d_eq_m > 1e-12 else np.nan
    mean_speed_to_flow_ratio = mean_speed_m_s / flow_speed_m_s if flow_speed_m_s > 1e-12 else np.nan
    terminal_to_flow_ratio = terminal_velocity_m_s / flow_speed_m_s if flow_speed_m_s > 1e-12 else np.nan
    streamwise_advective_ratio = abs(net_streamwise_velocity_m_s) / flow_speed_m_s if flow_speed_m_s > 1e-12 else np.nan
    vertical_transport_ratio = abs(net_vertical_velocity_m_s) / flow_speed_m_s if flow_speed_m_s > 1e-12 else np.nan
    slip_velocity_mag_m_s = abs(flow_speed_m_s - abs(net_streamwise_velocity_m_s)) if np.isfinite(net_streamwise_velocity_m_s) else np.nan
    summary = {'trial': trial_dir.name, 'batch_label': batch_label, 'flow_speed_in_s': flow_speed_in_s, 'flow_speed_m_s': flow_speed_m_s, 'particle_type': info['particle_type'], 'shape': info['shape'], 'material': info['material'], 'density_kg_m3': rho_p, 'density_ratio_rho_p_over_rho_f': density_ratio, 'submerged_specific_gravity': submerged_specific_gravity, 'diameter_m': info.get('diameter_m', np.nan), 'length_m': info.get('length_m', np.nan), 'thickness_m': info.get('thickness_m', np.nan), 'shape_aspect_ratio': shape_aspect_ratio, 'equivalent_diameter_m': d_eq_m, 'volume_m3': volume_m3, 'projected_area_m2': projected_area_m2, 'particle_mass_kg': particle_mass_kg, 'displaced_fluid_mass_kg': displaced_fluid_mass_kg, 'buoyant_mass_kg': buoyant_mass_kg, 'weight_N': weight_n, 'buoyancy_N': buoyancy_n, 'apparent_weight_N': apparent_weight_n, 'n_points': len(kin), 'duration_s': duration_s, 'trajectory_length_mm': path_length_m * 1000.0, 'net_displacement_mm': net_disp_m * 1000.0, 'net_streamwise_displacement_mm': net_dx_m * 1000.0, 'net_vertical_displacement_mm': net_dy_m * 1000.0, 'tortuosity': tortuosity, 'mean_speed_mm_s': mean_speed_m_s * 1000.0 if np.isfinite(mean_speed_m_s) else np.nan, 'max_speed_mm_s': max_speed_m_s * 1000.0 if np.isfinite(max_speed_m_s) else np.nan, 'terminal_velocity_mm_s': terminal_velocity_m_s * 1000.0, 'settling_velocity_mm_s': terminal_velocity_m_s * 1000.0, 'mean_streamwise_velocity_mm_s': mean_streamwise_velocity_m_s * 1000.0 if np.isfinite(mean_streamwise_velocity_m_s) else np.nan, 'mean_vertical_velocity_mm_s': mean_vertical_velocity_m_s * 1000.0 if np.isfinite(mean_vertical_velocity_m_s) else np.nan, 'mean_abs_streamwise_velocity_mm_s': mean_abs_streamwise_velocity_m_s * 1000.0 if np.isfinite(mean_abs_streamwise_velocity_m_s) else np.nan, 'mean_abs_vertical_velocity_mm_s': mean_abs_vertical_velocity_m_s * 1000.0 if np.isfinite(mean_abs_vertical_velocity_m_s) else np.nan, 'net_streamwise_velocity_mm_s': net_streamwise_velocity_m_s * 1000.0 if np.isfinite(net_streamwise_velocity_m_s) else np.nan, 'net_vertical_velocity_mm_s': net_vertical_velocity_m_s * 1000.0 if np.isfinite(net_vertical_velocity_m_s) else np.nan, 'slip_velocity_mm_s': slip_velocity_mag_m_s * 1000.0 if np.isfinite(slip_velocity_mag_m_s) else np.nan, 'mean_curvature_1_m': mean_curvature, 'mean_normal_acc_m_s2': mean_normal_acc, 'Re': re_terminal, 'Re_terminal': re_terminal, 'Re_mean': re_mean, 'Cd_eff': cd_eff, 'Cl_eff': cl_eff, 'lift_force_est_N': lift_force_est_n, 'Archimedes_Ar': archimedes, 'Galileo_Ga': galileo, 'tau_p_stokes_s': tau_p_stokes_s, 'Stokes_number_flow': stokes_number_flow, 'Froude_mean': froude_mean, 'Froude_terminal': froude_terminal, 'mean_speed_to_flow_ratio': mean_speed_to_flow_ratio, 'terminal_to_flow_ratio': terminal_to_flow_ratio, 'streamwise_advective_ratio': streamwise_advective_ratio, 'vertical_transport_ratio': vertical_transport_ratio}
    kin['batch_label'] = batch_label
    kin['trial'] = trial_dir.name
    kin['flow_speed_in_s'] = flow_speed_in_s
    return (summary, kin)

def theoretical_cd_sphere(reynolds):
    """Handle the theoretical cd sphere step used by this script."""
    reynolds = np.asarray(reynolds, dtype=float)
    cd = np.full_like(reynolds, np.nan, dtype=float)
    low = reynolds < 1000
    high = ~low
    with np.errstate(divide='ignore', invalid='ignore'):
        cd[low] = 24.0 / reynolds[low] * (1.0 + 0.15 * reynolds[low] ** 0.687)
    cd[high] = 0.44
    return cd

def aggregate_with_std(df, group_col, metrics):
    """Handle the aggregate with std step used by this script."""
    grouped = df.groupby(group_col, dropna=False)
    out = grouped.agg(n_trials=('trial', 'count'), **{f'{metric}_mean': (metric, 'mean') for metric in metrics}, **{f'{metric}_std': (metric, lambda s: np.nanstd(s, ddof=1) if len(s.dropna()) > 1 else np.nan) for metric in metrics}).reset_index()
    return out

def plot_by_group(df, group_col, col, ylabel, path, title_suffix=None):
    """Handle the plot by group step used by this script."""
    agg = df.groupby(group_col, dropna=False)[col].agg(['mean', 'std']).reset_index()
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=['mean'])
    if agg.empty:
        return
    x = np.arange(len(agg))
    y = agg['mean'].to_numpy(dtype=float)
    yerr = agg['std'].fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=4, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(agg[group_col], rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    title = f"{ylabel} by {group_col.replace('_', ' ')}"
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_by_trial(df, col, ylabel, path, title_suffix=None):
    """Handle the plot by trial step used by this script."""
    sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col]).copy()
    if sub.empty:
        return
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, sub[col].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(sub['trial'], rotation=75, ha='right')
    ax.set_ylabel(ylabel)
    title = f'{ylabel} by trial'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_drag_curve(df, path, title_suffix=None):
    """Handle the plot drag curve step used by this script."""
    sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Re', 'Cd_eff']).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    re_grid = np.logspace(-1, 4, 400)
    ax.plot(re_grid, theoretical_cd_sphere(re_grid), label='Theoretical sphere drag')
    if 'particle_flow_group' not in sub.columns:
        sub['particle_flow_group'] = sub['particle_type']
    for group in sub['particle_flow_group'].dropna().unique():
        grp = sub[sub['particle_flow_group'] == group]
        ax.scatter(grp['Re'], grp['Cd_eff'], label=group)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds number, Re')
    ax.set_ylabel('Effective drag coefficient, Cd')
    title = 'Experimental effective drag vs theoretical sphere drag'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def write_equations_reference(out_dir: Path):
    """Write the equations reference used by this script."""
    text = '# Open-Channel Transport Equations Used\n\nThis report keeps the original trajectory, drag, lift, and Reynolds-number calculations and\nadds the following extra quantities for plastic-particle transport interpretation.\n\n## Added transport / scaling equations\n\n- Density ratio:\n  `rho_p / rho_f`\n\n- Submerged specific gravity:\n  `(rho_p - rho_f) / rho_f`\n\n- Equivalent spherical diameter:\n  `d_eq = (6 V / pi)^(1/3)`\n\n- Particle mass:\n  `m_p = rho_p V`\n\n- Displaced-fluid mass:\n  `m_f = rho_f V`\n\n- Buoyant mass:\n  `m_b = (rho_p - rho_f) V`\n\n- Weight:\n  `W = rho_p V g`\n\n- Buoyancy:\n  `F_b = rho_f V g`\n\n- Apparent weight:\n  `W_app = (rho_p - rho_f) V g`\n\n- Terminal Reynolds number:\n  `Re_terminal = rho_f U_terminal d_eq / mu`\n\n- Mean-speed Reynolds number:\n  `Re_mean = rho_f U_mean d_eq / mu`\n\n- Archimedes number:\n  `Ar = g * |rho_p - rho_f| / rho_f * d_eq^3 / nu^2`\n\n- Galileo number:\n  `Ga = sqrt(Ar)`\n\n- Stokes response time:\n  `tau_p = rho_p d_eq^2 / (18 mu)`\n\n- Flow-based Stokes number:\n  `St = tau_p U_flow / d_eq`\n\n- Mean Froude number:\n  `Fr_mean = U_mean / sqrt(g d_eq)`\n\n- Terminal Froude number:\n  `Fr_terminal = U_terminal / sqrt(g d_eq)`\n\n- Mean-speed to flow-speed ratio:\n  `U_mean / U_flow`\n\n- Terminal-speed to flow-speed ratio:\n  `U_terminal / U_flow`\n\n- Streamwise advective ratio:\n  `|U_x,net| / U_flow`\n\n- Vertical transport ratio:\n  `|U_y,net| / U_flow`\n\n- Slip-velocity magnitude:\n  `|U_flow - |U_x,net||`\n\n## Notes\n\n- `U_terminal` is estimated from a PCA-aligned fit over the late-stage part of the trajectory.\n- `U_x,net` and `U_y,net` are net streamwise and vertical transport velocities over the full run.\n- For non-spherical particles, the drag calculation still uses the nominal projected area from the\n  existing script, while the added dimensionless groups use `d_eq` for cross-shape scaling.\n'
    (out_dir / 'equations_reference.md').write_text(text, encoding='utf-8')

def collect_trial_dirs(source_dirs):
    """Handle the collect trial dirs step used by this script."""
    trial_dirs = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for child in sorted(source_dir.iterdir()):
            if child.is_dir() and (child / 'tracks_processed.csv').exists():
                trial_dirs.append(child)
    return trial_dirs

def process_batch(batch_def, out_dir: Path):
    """Handle the process batch step used by this script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / 'processed_kinematics'
    if SAVE_PER_TRIAL_KINEMATICS:
        processed_dir.mkdir(exist_ok=True)
    analysis_fps = float(batch_def.get('analysis_fps', FPS))
    metrics = ['terminal_velocity_mm_s', 'mean_speed_mm_s', 'mean_streamwise_velocity_mm_s', 'slip_velocity_mm_s', 'tortuosity', 'Re', 'Cd_eff', 'Cl_eff', 'Archimedes_Ar', 'Galileo_Ga', 'Stokes_number_flow', 'Froude_terminal', 'terminal_to_flow_ratio', 'streamwise_advective_ratio']
    rows = []
    logs = []
    trial_dirs = collect_trial_dirs(batch_def['source_dirs'])
    for trial_dir in trial_dirs:
        try:
            summary, kin = analyse_trial(trial_dir=trial_dir, batch_label=batch_def['label'], flow_speed_in_s=batch_def['flow_speed_in_s'], analysis_fps=analysis_fps)
            rows.append(summary)
            logs.append({'trial': trial_dir.name, 'status': 'processed', 'message': ''})
            if SAVE_PER_TRIAL_KINEMATICS:
                kin.to_csv(processed_dir / f'{trial_dir.name}_kinematics.csv', index=False)
            print(f"Processed: {batch_def['label']} / {trial_dir.name}")
        except Exception as exc:
            logs.append({'trial': trial_dir.name, 'status': 'skipped', 'message': str(exc)})
            print(f"Skipped {batch_def['label']} / {trial_dir.name}: {exc}")
    log_df = pd.DataFrame(logs)
    log_df.to_csv(out_dir / 'processing_log.csv', index=False)
    if not rows:
        raise RuntimeError(f"No valid trial folders processed for batch {batch_def['label']}.")
    trials = pd.DataFrame(rows).sort_values(['particle_type', 'trial']).reset_index(drop=True)
    trials['particle_flow_group'] = trials['particle_type'] + ' @ ' + trials['batch_label']
    by_particle = aggregate_with_std(trials, 'particle_type', metrics)
    by_shape = aggregate_with_std(trials, 'shape', metrics)
    trials.to_csv(out_dir / 'full_physics_results_by_trial.csv', index=False)
    by_particle.to_csv(out_dir / 'full_physics_results_by_particle.csv', index=False)
    by_shape.to_csv(out_dir / 'full_physics_results_by_shape.csv', index=False)
    plot_by_group(trials, 'particle_type', 'terminal_velocity_mm_s', 'Terminal velocity (mm/s)', out_dir / 'terminal_velocity_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'mean_speed_mm_s', 'Mean speed (mm/s)', out_dir / 'mean_speed_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'slip_velocity_mm_s', 'Slip-velocity magnitude (mm/s)', out_dir / 'slip_velocity_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'Archimedes_Ar', 'Archimedes number, Ar', out_dir / 'archimedes_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'Stokes_number_flow', 'Flow-based Stokes number', out_dir / 'stokes_number_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'terminal_to_flow_ratio', 'Terminal/flow speed ratio (-)', out_dir / 'terminal_to_flow_ratio_by_particle.png', batch_def['label'])
    plot_by_group(trials, 'particle_type', 'Cd_eff', 'Effective drag coefficient, Cd', out_dir / 'drag_coefficient_by_particle.png', batch_def['label'])
    plot_by_trial(trials, 'terminal_velocity_mm_s', 'Terminal velocity (mm/s)', out_dir / 'terminal_velocity_by_trial.png', batch_def['label'])
    plot_by_trial(trials, 'mean_streamwise_velocity_mm_s', 'Mean streamwise velocity (mm/s)', out_dir / 'streamwise_velocity_by_trial.png', batch_def['label'])
    plot_by_trial(trials, 'slip_velocity_mm_s', 'Slip-velocity magnitude (mm/s)', out_dir / 'slip_velocity_by_trial.png', batch_def['label'])
    plot_by_trial(trials, 'Cd_eff', 'Effective drag coefficient, Cd', out_dir / 'drag_coefficient_by_trial.png', batch_def['label'])
    plot_by_trial(trials, 'Stokes_number_flow', 'Flow-based Stokes number', out_dir / 'stokes_number_by_trial.png', batch_def['label'])
    plot_by_trial(trials, 'tortuosity', 'Tortuosity (-)', out_dir / 'tortuosity_by_trial.png', batch_def['label'])
    plot_drag_curve(trials, out_dir / 'drag_curve_comparison.png', batch_def['label'])
    return {'trials': trials, 'by_particle': by_particle, 'by_shape': by_shape, 'processing_log': log_df}

def build_combined_report(batch_results, out_dir: Path):
    """Build the combined report used by this script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = pd.concat([result['trials'] for result in batch_results], ignore_index=True)
    trials = trials.sort_values(['batch_label', 'particle_type', 'trial']).reset_index(drop=True)
    trials['particle_flow_group'] = trials['particle_type'] + ' @ ' + trials['batch_label']
    metrics = ['terminal_velocity_mm_s', 'mean_speed_mm_s', 'mean_streamwise_velocity_mm_s', 'slip_velocity_mm_s', 'tortuosity', 'Re', 'Cd_eff', 'Cl_eff', 'Archimedes_Ar', 'Galileo_Ga', 'Stokes_number_flow', 'Froude_terminal', 'terminal_to_flow_ratio', 'streamwise_advective_ratio']
    by_particle_flow = aggregate_with_std(trials, 'particle_flow_group', metrics)
    by_shape_flow = aggregate_with_std(trials, 'batch_label', metrics)
    combined_log = pd.concat([result['processing_log'] for result in batch_results], ignore_index=True)
    trials.to_csv(out_dir / 'full_physics_results_by_trial.csv', index=False)
    by_particle_flow.to_csv(out_dir / 'full_physics_results_by_particle_flow.csv', index=False)
    by_shape_flow.to_csv(out_dir / 'full_physics_results_by_flow.csv', index=False)
    combined_log.to_csv(out_dir / 'processing_log.csv', index=False)
    plot_by_group(trials, 'particle_flow_group', 'terminal_velocity_mm_s', 'Terminal velocity (mm/s)', out_dir / 'terminal_velocity_by_particle_flow.png', 'combined')
    plot_by_group(trials, 'particle_flow_group', 'mean_speed_mm_s', 'Mean speed (mm/s)', out_dir / 'mean_speed_by_particle_flow.png', 'combined')
    plot_by_group(trials, 'particle_flow_group', 'slip_velocity_mm_s', 'Slip-velocity magnitude (mm/s)', out_dir / 'slip_velocity_by_particle_flow.png', 'combined')
    plot_by_group(trials, 'particle_flow_group', 'Stokes_number_flow', 'Flow-based Stokes number', out_dir / 'stokes_number_by_particle_flow.png', 'combined')
    plot_by_group(trials, 'particle_flow_group', 'terminal_to_flow_ratio', 'Terminal/flow speed ratio (-)', out_dir / 'terminal_to_flow_ratio_by_particle_flow.png', 'combined')
    plot_drag_curve(trials, out_dir / 'drag_curve_comparison.png', 'combined')

def main():
    """Run the main entry point for this script."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_equations_reference(OUTPUT_ROOT)
    batch_summaries = []
    batch_results = []
    for batch_def in BATCHES:
        batch_out = OUTPUT_ROOT / batch_def['label']
        result = process_batch(batch_def, batch_out)
        batch_results.append(result)
        batch_summaries.append({'label': batch_def['label'], 'flow_speed_in_s': batch_def['flow_speed_in_s'], 'n_processed': int((result['processing_log']['status'] == 'processed').sum()), 'n_skipped': int((result['processing_log']['status'] == 'skipped').sum()), 'output_dir': str(batch_out)})
    build_combined_report(batch_results, OUTPUT_ROOT / 'combined')
    manifest = {'output_root': str(OUTPUT_ROOT), 'batches': batch_summaries}
    (OUTPUT_ROOT / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    script_path = WORKSPACE / 'trajectory_physics_open_channel_full_report.py'
    if script_path.exists():
        (OUTPUT_ROOT / script_path.name).write_text(script_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Saved physics reports to: {OUTPUT_ROOT}')

# Direct execution entry point used when the script is run outside the batch drivers.
if __name__ == '__main__':
    main()
